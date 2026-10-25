"""Stage A2: AMI end-of-turn events with audio-derived VAD (the deployed task).

    .venv/bin/python scripts/a2_ami_build.py --split dev test [--workers 4] [--out data/a2/ami]

AMI's word alignments are contiguous inside a transcriber segment (each word ends where the next
starts), so pauses *inside* a segment are invisible to them and alignment-only mining
(``a2_ami_mine.py``) only finds the long between-segment silences. Here the VAD comes from the
headset audio (Silero, as Pipecat's VAD would see it), gated by the transcript so that crosstalk
from the other headsets does not count as this speaker's speech:

    keep a Silero region if it overlaps one of the speaker's own transcribed segments (expanded by
    GATE_PAD_S) or any of their timed words.

The rules (``eot.turn_rules``) then run per speaker against all the others. Outputs per meeting
under ``--out/<meeting>/``: ``vad_<L>.json`` (gated regions), ``events.jsonl`` (every silence incl.
excluded, with the audio path and headset channel) and a manifest parquet per split with the
labelled events only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import ami  # noqa: E402
from eot import turn_rules as R  # noqa: E402
from eot.turn_rules import Segment, Speaker  # noqa: E402

GATE_PAD_S = 0.3
_DET = None


def _init():
    global _DET
    import torch
    torch.set_num_threads(1)
    from eot.offsets import OffsetDetector
    _DET = OffsetDetector(min_silence_ms=100, min_speech_ms=100, speech_pad_ms=30)


def gate(regions: list[Segment], spk: Speaker, pad: float = GATE_PAD_S) -> list[Segment]:
    spans = [(u.start - pad, u.end + pad) for u in spk.utts]
    words = [(w.start, w.end) for u in spk.utts for w in u.words]
    out = []
    for r in regions:
        if any(r.start < e and r.end > s for s, e in spans) or any(r.start < e and r.end > s for s, e in words):
            out.append(r)
    return out


def build_meeting(args: tuple) -> dict:
    meeting, spks, out_dir = args
    import soundfile as sf
    speakers: list[Speaker] = []
    meta: dict[str, dict] = {}
    for ms in spks:
        wav = ami.headset_path(meeting, ms.channel)
        if not wav.exists():
            continue
        x, sr = sf.read(wav, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            import torch, torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        spk_tr, info = ami.speaker_data(meeting, ms)
        raw = [Segment(a, b) for a, b in _DET.segments(x)]
        regs = gate(raw, spk_tr)
        speakers.append(Speaker(ms.letter, regs, spk_tr.utts))
        meta[ms.letter] = dict(channel=ms.channel, audio=str(wav), duration_s=round(len(x) / 16000, 2),
                               vad_raw=len(raw), vad_gated=len(regs),
                               speech_raw_s=round(sum(b - a for a, b in ((r.start, r.end) for r in raw)), 1),
                               speech_gated_s=round(sum(r.end - r.start for r in regs), 1), **info)
    if not speakers:
        return dict(meeting=meeting, n=0, events=[])
    evs = R.find_all(speakers, include_excluded=True)
    d = Path(out_dir) / meeting
    d.mkdir(parents=True, exist_ok=True)
    for s in speakers:
        (d / f"vad_{s.name}.json").write_text(json.dumps([[round(r.start, 3), round(r.end, 3)] for r in s.vad]))
    (d / "speakers.json").write_text(json.dumps(meta, indent=1))
    rows = []
    with open(d / "events.jsonl", "w") as f:
        for e in evs:
            row = dict(meeting=meeting, split=ami.split_of(meeting), speaker=e.speaker,
                       channel=meta[e.speaker]["channel"], audio=meta[e.speaker]["audio"], t=round(e.t, 3),
                       label=e.label, reason=e.reason, gap_s=e.gap_s, b_onset_s=e.b_onset_s, b_speaker=e.b_speaker,
                       a_turn_s=e.a_turn_s, a_turn_words=e.a_turn_words, a_text=e.a_text[-400:], b_text=e.b_text[:200])
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    return dict(meeting=meeting, n=len(rows), events=rows, meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="*", default=["dev", "test"], choices=["dev", "test", "train"])
    ap.add_argument("--meetings", nargs="*", default=[])
    ap.add_argument("--out", default="data/a2/ami")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ms = ami.meetings()
    want = [m for m in sorted(ms) if ami.split_of(m) in a.split or m in a.meetings]
    jobs = [(m, ms[m], a.out) for m in want]
    all_rows = []
    with Pool(a.workers, initializer=_init) as pool:
        for res in pool.imap_unordered(build_meeting, jobs):
            c = Counter(r["reason"] for r in res["events"])
            print(f"{res['meeting']}: {res['n']} silences, complete {c['complete']}, incomplete {c['incomplete']}", flush=True)
            all_rows += res["events"]
    out = Path(a.out)
    # Manifests cover every meeting built so far under --out, not only this run (so a partial
    # rebuild of a few meetings does not shrink them).
    all_rows = [json.loads(line) for f in sorted(out.glob("*/events.jsonl")) for line in open(f)]
    df = pd.DataFrame(all_rows)
    for split in sorted(df.split.unique()):
        lab = df[(df.split == split) & df.label.notna()].drop(columns=["reason"])
        lab.to_parquet(out / f"manifest_{split}.parquet", index=False)
    print("\n| split | meetings | complete | incomplete | no_floor | short_turn | collision | long_silence | tail |")
    print("|---|---|---|---|---|---|---|---|---|")
    for split in sorted(df.split.unique()):
        g = df[df.split == split]
        c = Counter(g.reason)
        print(f"| {split} | {g.meeting.nunique()} | {c['complete']} | {c['incomplete']} | {c['no_floor']} | "
              f"{c['short_turn']} | {c['collision']} | {c['long_silence']} | {c['tail']} |")
    lab = df[df.label.notna()]
    inc, com = lab[lab.label == "incomplete"], lab[lab.label == "complete"]
    print(f"\nlabelled {len(lab)}; complete fraction {lab.label.eq('complete').mean():.3f}")
    print("incomplete gap_s quantiles:", inc.gap_s.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
          " frac < 0.5 s:", round((inc.gap_s < 0.5).mean(), 3), " frac >= 1.5 s:", round((inc.gap_s >= 1.5).mean(), 3))
    print("complete b_onset_s quantiles:", com.b_onset_s.quantile([.1, .25, .5, .75]).round(2).to_dict(),
          " frac <= 0.2 s:", round((com.b_onset_s <= 0.2).mean(), 3), " frac <= 0.5 s:", round((com.b_onset_s <= 0.5).mean(), 3))


if __name__ == "__main__":
    main()
