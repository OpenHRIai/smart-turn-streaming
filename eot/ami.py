"""AMI Meeting Corpus access for Stage A2: annotations (NXT XML) and headset audio.

Corpus: https://groups.inf.ed.ac.uk/ami/corpus/ (CC BY 4.0; the licence text ships in the
annotations zip as LICENCE.txt and is kept with the data). Layout used here:

    data/ami/annotations/words/<meeting>.<A-E>.words.xml      word alignments per speaker
    data/ami/annotations/segments/<meeting>.<A-E>.segments.xml transcriber segments (word ranges)
    data/ami/annotations/corpusResources/meetings.xml          speaker letter -> headset channel, camera
    data/ami/audio/<meeting>/<meeting>.Headset-<n>.wav         per-speaker close-talking audio (16 kHz)

Words with ``punc="true"`` are punctuation (zero duration) and are not words for the rules;
they are kept on the utterance transcript so ``a_text`` reads naturally. Words without timings
(a few, see 00README) are dropped and counted.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from eot.turn_rules import Segment, Speaker, Utterance, Word, words_to_vad

ANN_DIR = Path("data/ami/annotations")
AUDIO_DIR = Path("data/ami/audio")
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
NS = {"nite": "http://nite.sourceforge.net/"}

# Kaldi AMI recipe splits (egs/ami/s5/local/split_{dev,eval}.orig); everything else is train.
DEV_MEETINGS = ("ES2011a", "ES2011b", "ES2011c", "ES2011d", "IS1008a", "IS1008b", "IS1008c", "IS1008d",
                "TS3004a", "TS3004b", "TS3004c", "TS3004d", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011")
EVAL_MEETINGS = ("ES2004a", "ES2004b", "ES2004c", "ES2004d", "IS1009a", "IS1009b", "IS1009c", "IS1009d",
                 "TS3003a", "TS3003b", "TS3003c", "TS3003d", "EN2002a", "EN2002b", "EN2002c", "EN2002d")


def split_of(meeting: str) -> str:
    if meeting in DEV_MEETINGS:
        return "dev"
    if meeting in EVAL_MEETINGS:
        return "test"
    return "train"


@dataclass
class MeetingSpeaker:
    meeting: str
    letter: str          # nxt_agent A..E
    channel: int         # headset channel
    camera: str
    global_name: str     # participant id across meetings
    role: str


def meetings(ann_dir: Path = ANN_DIR) -> dict[str, list[MeetingSpeaker]]:
    root = ET.parse(ann_dir / "corpusResources" / "meetings.xml").getroot()
    out: dict[str, list[MeetingSpeaker]] = {}
    for m in root.findall("meeting"):
        obs = m.get("observation")
        spk = [MeetingSpeaker(obs, s.get("nxt_agent"), int(s.get("channel")), s.get("camera") or "",
                              s.get("global_name") or "", s.get("role") or "") for s in m.findall("speaker")]
        out[obs] = sorted(spk, key=lambda s: s.letter)
    return out


_ID_RANGE = re.compile(r"id\(([^)]+)\)")


def read_words(meeting: str, letter: str, ann_dir: Path = ANN_DIR) -> tuple[list[Word], list[tuple[str, Word]], int]:
    """(timed words, all tokens incl. punctuation with their nite ids, n dropped for missing times)."""
    path = ann_dir / "words" / f"{meeting}.{letter}.words.xml"
    if not path.exists():
        return [], [], 0
    root = ET.parse(path).getroot()
    words: list[Word] = []
    tokens: list[tuple[str, Word]] = []
    dropped = 0
    for w in root.findall("w"):
        text = (w.text or "").strip()
        st, en = w.get("starttime"), w.get("endtime")
        if st is None or en is None or st in ("", "nan") or en in ("", "nan"):
            dropped += 1
            continue
        try:
            s, e = float(st), float(en)
        except ValueError:
            dropped += 1
            continue
        tok = Word(text, s, e)
        tokens.append((w.get(f"{{{NS['nite']}}}id"), tok))
        if w.get("punc") != "true" and text:
            words.append(tok)
    return words, tokens, dropped


def read_segments(meeting: str, letter: str, ann_dir: Path = ANN_DIR) -> list[tuple[float, float, str, str]]:
    """(transcriber_start, transcriber_end, first word id, last word id) per segment."""
    path = ann_dir / "segments" / f"{meeting}.{letter}.segments.xml"
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    out = []
    for s in root.findall("segment"):
        child = s.find("nite:child", NS)
        if child is None:
            continue
        ids = _ID_RANGE.findall(child.get("href", ""))
        if not ids:
            continue
        out.append((float(s.get("transcriber_start")), float(s.get("transcriber_end")), ids[0], ids[-1]))
    return out


def speaker_data(meeting: str, ms: MeetingSpeaker, ann_dir: Path = ANN_DIR) -> tuple[Speaker, dict]:
    """A ``Speaker`` for the rules: VAD from word alignments, utterances from transcriber segments."""
    words, tokens, dropped = read_words(meeting, ms.letter, ann_dir)
    idx = {i: k for k, (i, _) in enumerate(tokens)}
    utts: list[Utterance] = []
    for st, en, first, last in read_segments(meeting, ms.letter, ann_dir):
        if first not in idx or last not in idx:
            continue
        toks = [t for _, t in tokens[idx[first]: idx[last] + 1]]
        timed = [t for t in toks if t.end > t.start]
        text = " ".join(t.text for t in toks)
        if not timed:
            continue
        utts.append(Utterance(min(t.start for t in timed), max(t.end for t in timed), text, timed))
    vad: list[Segment] = words_to_vad(words)
    info = dict(n_words=len(words), n_dropped=dropped, n_utts=len(utts), speech_s=round(sum(s.end - s.start for s in vad), 1))
    return Speaker(ms.letter, vad, utts), info


def headset_url(meeting: str, channel: int) -> str:
    return f"{MIRROR}/{meeting}/audio/{meeting}.Headset-{channel}.wav"


def headset_path(meeting: str, channel: int, audio_dir: Path = AUDIO_DIR) -> Path:
    return audio_dir / meeting / f"{meeting}.Headset-{channel}.wav"
