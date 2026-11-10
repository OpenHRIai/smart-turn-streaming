"""CANDOR end-of-turn test: windows on eot-bench's scoring grid, from conversation half B.

    .venv/bin/python scripts/candor_eot_items.py \
        --events ../smart-turn-multimodal-private/runs/candor/bids/events_all.jsonl \
        --out ../OpenVideoFDB/data/candor/eot_test/items.parquet

Events come from the WS9/A2 turn rules over Silero VAD and AWS words (``eot/turn_rules.py`` in
smart-turn-multimodal-private, branch ``candor-bids``): ``complete`` = the partner took a real turn
within 3 s and kept it, ``incomplete`` = the same speaker resumed within 5 s. Only conversations in
half B of OpenVideoFDB's split (md5 of the conversation id, odd) are used, so half A stays free for
training. A seeded sample of conversations, then of events per conversation.

Each event becomes one span, scored every 100 ms as eot-bench does: a turn end (``eot``) for 1.5 s,
eot-bench's own eot span length, with the audio after the partner's onset replaced by room tone, as
the AMI items are cut; a pause (``hold``) until the speaker resumes. One row per window:

    item_id, event_id, conv, participant, audio (path on the ``candor`` volume), label (eot | hold),
    span_dur, t_off (VAD stop, conversation seconds), kind ('tau'), value (= silence so far),
    end_s, cut_s (NaN = none), p_end, gap_s, b_onset_s

Uses the CANDOR Corpus from BetterUp Labs (Reece, Cooney et al., Science Advances 2023,
https://doi.org/10.1126/sciadv.adf3197). CANDOR Corpus License v2.1: no redistribution of the data or
of anything derived from it. The output
belongs under ``OpenVideoFDB/data/candor`` (git-ignored) or on the ``candor`` volume, keyed by
conversation id so removals can be honoured.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

STEP = 0.1
EOT_SPAN_S = 1.5
MAX_HOLD_S = 5.0
_EPS = 1e-6


def half_of(conv: str) -> int:
    """OpenVideoFDB's conversation split (cue_confirm.py): 0 = half A, 1 = half B."""
    return int(hashlib.md5(str(conv).encode()).hexdigest(), 16) % 2


def time_grid(start: float, end: float, step: float = STEP) -> list[float]:
    """eot_harness.io._time_grid."""
    n = max(0, int(math.floor((end - start) / step + _EPS)))
    values = [round(start + i * step, 6) for i in range(n + 1)]
    if values and values[-1] < end - _EPS:
        values.append(round(end, 6))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-conv", type=int, default=200)
    ap.add_argument("--ends-per-conv", type=int, default=10)
    ap.add_argument("--holds-per-conv", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cols = ["conv", "participant", "t", "label", "gap_s", "b_onset_s", "a_turn_s"]
    ev = pd.read_json(a.events, lines=True)[cols]
    ev = ev[ev.label.isin(["complete", "incomplete"])]
    ev = ev[ev.conv.map(half_of) == 1]
    ev = ev[(ev.label == "complete") | (ev.gap_s <= MAX_HOLD_S)]
    convs = sorted(ev.conv.unique())
    chosen = sorted(rng.choice(convs, size=min(a.n_conv, len(convs)), replace=False).tolist())
    print(f"half B: {len(convs)} conversations with labelled events; sampled {len(chosen)}")

    picked = []
    for conv in chosen:
        g = ev[ev.conv == conv]
        for label, n in (("complete", a.ends_per_conv), ("incomplete", a.holds_per_conv)):
            h = g[g.label == label]
            if len(h):
                picked.append(h.iloc[np.sort(rng.choice(len(h), size=min(n, len(h)), replace=False))])
    ev = pd.concat(picked, ignore_index=True)

    rows = []
    for e in ev.itertuples(index=False):
        eot = e.label == "complete"
        dur = EOT_SPAN_S if eot else float(e.gap_s)
        event_id = f"{e.conv}/{e.participant}/{e.t:.3f}"
        for tau in time_grid(0.0, dur):
            cut = e.t + e.b_onset_s if (eot and tau > e.b_onset_s) else np.nan
            rows.append(dict(event_id=event_id, conv=e.conv, participant=e.participant,
                             audio=f"proc/{e.conv}/{e.participant}/audio16.flac", label="eot" if eot else "hold",
                             span_dur=dur, t_off=float(e.t), kind="tau", value=float(tau), end_s=float(e.t + tau),
                             cut_s=cut, p_end=int(eot), gap_s=float(e.gap_s), b_onset_s=float(e.b_onset_s)))
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", np.arange(len(df)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    spans = df.drop_duplicates("event_id")
    print(f"{len(df)} windows, {spans.conv.nunique()} conversations, {spans.participant.nunique()} speakers; "
          f"spans: {int((spans.label == 'eot').sum())} eot, {int((spans.label == 'hold').sum())} hold "
          f"(hold length median {spans[spans.label == 'hold'].span_dur.median():.2f} s); "
          f"eot windows cut at the partner's onset: {int(df.cut_s.notna().sum())} -> {a.out}")


if __name__ == "__main__":
    main()
