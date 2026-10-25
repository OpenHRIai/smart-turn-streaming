"""End-of-turn labelling rules over one multi-party conversation (Stage A2).

A port of OpenVideoFDB's WS9 rules (``ovfdb/endpoint/rules.py``, dyadic) generalised to any
number of participants: **A = the speaker whose silence is judged**, **others = everyone else**.
Pure functions: VAD regions and word-aligned utterances in, ``SilenceEvent`` list out. No media.

A *silence* is the end ``t`` of an A VAD region whose gap to A's next region is at least
``MIN_GAP`` (a deployed endpointer is asked at every VAD stop). Its label is what happened next:

* ``complete``   --- some other participant took a real turn (not a backchannel) within
                     ``MAX_RESPONSE_S`` and A let them keep it for ``MIN_YIELD_S``.
* ``incomplete`` --- A resumed within ``MAX_PAUSE_S`` before anyone else said more than a
                     backchannel.
* excluded       --- anything else, with a ``reason`` (collision, long_silence, no_floor,
                     short_turn, tail). Kept for the mid-turn negatives and for reporting.

Differences from the dyadic version are confined to "others": the floor test, the next real
onset and the turn cuts all take the union of the other participants' speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_GAP = 0.2          # s: shortest A silence that counts as a VAD stop
FLOOR_S = 1.0          # s before t in which no other participant may be speaking (non-backchannel)
MIN_TURN_WORDS = 2     # A's turn so far must have >= this many words ...
MIN_TURN_S = 1.0       # ... or >= this much speech
MAX_RESPONSE_S = 3.0   # s: another participant's turn must start within this of t for "complete"
MIN_YIELD_S = 1.0      # s: A must stay quiet this long after the other's onset, else collision
MAX_PAUSE_S = 5.0      # s: A must resume within this for "incomplete"
CONTEXT_S = 8.0        # s of A transcript kept as evidence (= model audio context)
TURN_GAP = 0.6         # silence that still joins two VAD regions into one turn
BACKCHANNEL_MAX_DUR = 1.0
BACKCHANNEL_MAX_WORDS = 2   # strictly fewer than this many words

COMPLETE = "complete"
INCOMPLETE = "incomplete"


@dataclass
class Segment:
    start: float
    end: float


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Utterance:
    start: float
    end: float
    transcript: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Speaker:
    name: str
    vad: list[Segment]
    utts: list[Utterance]


@dataclass
class SilenceEvent:
    speaker: str
    t: float                     # A's VAD region end, file seconds
    label: str | None            # COMPLETE | INCOMPLETE | None (excluded)
    reason: str                  # label name, or the exclusion reason
    gap_s: float | None          # a_resume - t (None: A never speaks again)
    b_onset_s: float | None      # next other non-backchannel onset - t (None: none)
    b_speaker: str | None        # who took the floor (complete) / spoke next
    a_turn_s: float              # speech in A's current turn up to t
    a_turn_words: int
    a_text: str = ""
    b_text: str = ""

    @property
    def endpoint_bool(self) -> bool | None:
        return None if self.label is None else self.label == COMPLETE


# --- helpers ------------------------------------------------------------------

def n_words(u: Utterance) -> int:
    return len(u.words) if u.words else len(u.transcript.split())


def is_backchannel(u: Utterance) -> bool:
    return (u.end - u.start) < BACKCHANNEL_MAX_DUR and n_words(u) < BACKCHANNEL_MAX_WORDS


def merge_regions(vad: list[Segment], gap: float = TURN_GAP) -> list[list[Segment]]:
    runs: list[list[Segment]] = []
    for seg in sorted(vad, key=lambda s: s.start):
        if runs and seg.start - runs[-1][-1].end < gap:
            runs[-1].append(seg)
        else:
            runs.append([seg])
    return runs


@dataclass
class Turn:
    start: float
    end: float
    segs: list[Segment]


def speaker_turns(vad: list[Segment], other_real_onsets: list[float]) -> list[Turn]:
    """0.6 s-merged VAD runs, split where another participant's real utterance starts."""
    cuts = sorted(other_real_onsets)
    turns: list[Turn] = []
    for run in merge_regions(vad):
        cur: list[Segment] = []
        for seg in run:
            if cur and any(cur[-1].end <= c <= seg.start for c in cuts):
                turns.append(Turn(cur[0].start, cur[-1].end, cur))
                cur = []
            cur.append(seg)
        if cur:
            turns.append(Turn(cur[0].start, cur[-1].end, cur))
    return turns


def _text_in(utts: list[Utterance], t0: float, t1: float) -> str:
    parts = []
    for u in utts:
        if u.start < t1 and u.end > t0:
            if u.words:
                parts.extend(w.text for w in u.words if t0 <= w.start < t1)
            else:
                parts.append(u.transcript)
    return " ".join(parts).strip()


def words_to_vad(words: list[Word], gap: float = MIN_GAP) -> list[Segment]:
    """VAD regions from word alignments: words closer than ``gap`` join one region."""
    regs: list[Segment] = []
    for w in sorted(words, key=lambda w: w.start):
        if w.end <= w.start:
            continue
        if regs and w.start - regs[-1].end < gap:
            regs[-1].end = max(regs[-1].end, w.end)
        else:
            regs.append(Segment(w.start, w.end))
    return regs


# --- rules ----------------------------------------------------------------------

def find_silences(a: Speaker, others: list[Speaker], *, min_gap: float = MIN_GAP, floor_s: float = FLOOR_S,
                  max_response_s: float = MAX_RESPONSE_S, min_yield_s: float = MIN_YIELD_S,
                  max_pause_s: float = MAX_PAUSE_S, include_excluded: bool = False) -> list[SilenceEvent]:
    """Every silence of ``a``, labelled by what ``a`` and the others did next."""
    regs = sorted(a.vad, key=lambda s: s.start)
    real = sorted(((u, o.name) for o in others for u in o.utts if not is_backchannel(u)),
                  key=lambda x: x[0].start)
    a_turns = speaker_turns(a.vad, [u.start for u, _ in real])
    out: list[SilenceEvent] = []
    for i, reg in enumerate(regs):
        t = reg.end
        nxt = regs[i + 1] if i + 1 < len(regs) else None
        if nxt is not None and nxt.start - t < min_gap:
            continue
        gap = round(nxt.start - t, 3) if nxt is not None else None
        b_next = next(((u, name) for u, name in real if u.start > t), None)
        b_on = round(b_next[0].start - t, 3) if b_next is not None else None
        turn = next((tr for tr in a_turns if tr.start <= reg.start and tr.end >= t), None)
        turn_segs = [s for s in (turn.segs if turn else [reg]) if s.end <= t + 1e-6]
        a_turn_s = sum(s.end - s.start for s in turn_segs)
        t0 = turn_segs[0].start if turn_segs else reg.start
        a_turn_words = sum(1 for u in a.utts for w in u.words if t0 <= w.start < t)
        ev = SilenceEvent(speaker=a.name, t=t, label=None, reason="", gap_s=gap, b_onset_s=b_on,
                          b_speaker=b_next[1] if b_next else None,
                          a_turn_s=round(a_turn_s, 3), a_turn_words=a_turn_words,
                          a_text=_text_in(a.utts, t - CONTEXT_S, t),
                          b_text=b_next[0].transcript if b_next is not None else "")
        b_floor = any(u.start < t and u.end > t - floor_s for u, _ in real)
        if b_floor:
            ev.reason = "no_floor"
        elif a_turn_words < MIN_TURN_WORDS and a_turn_s < MIN_TURN_S:
            ev.reason = "short_turn"
        elif b_on is not None and b_on <= max_response_s and (gap is None or gap > b_on):
            if gap is not None and gap - b_on < min_yield_s:
                ev.reason = "collision"
            else:
                ev.label, ev.reason = COMPLETE, COMPLETE
        elif gap is not None and gap <= max_pause_s and (b_on is None or b_on > gap):
            ev.label, ev.reason = INCOMPLETE, INCOMPLETE
        elif gap is None and b_on is None:
            ev.reason = "tail"
        else:
            ev.reason = "long_silence"
        if ev.label is not None or include_excluded:
            out.append(ev)
    return out


def find_all(speakers: list[Speaker], **kw) -> list[SilenceEvent]:
    """Every speaker in turn against all the others, time-ordered."""
    evs: list[SilenceEvent] = []
    for i, a in enumerate(speakers):
        evs += find_silences(a, speakers[:i] + speakers[i + 1:], **kw)
    return sorted(evs, key=lambda e: (e.t, e.speaker))


def summarize(events: list[SilenceEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.reason] = counts.get(e.reason, 0) + 1
    return dict(sorted(counts.items()))
