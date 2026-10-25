"""Extra negatives for B1: AMI long thinking pauses that the rules excluded.

    .venv/bin/python scripts/b1_items_extra.py --items data/b1/items.parquet --out data/b1/items_extra.parquet

The WS9 rules exclude a silence as ``long_silence`` when the speaker resumed after more than
MAX_PAUSE_S (5 s) with nobody else taking the floor first. From the deployed model's point of view
these are incomplete pauses that happen to be long, and they are exactly the slice where false
alarms concentrate (pauses >= 1.5 s). Here each one yields tau windows at every tau (all < gap) with
``p_end = 0``, plus the pre-offset and mid windows with ``p_eager = 0``. Item ids continue from the
main items table so the two feature caches concatenate. Train: 4 taus / 3 deltas sampled; dev and
test: full sets (they are reported separately and do not enter the main AMI test tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.windows import DELTAS, MIN_SPEECH_BEFORE_S, TAUS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/b1/items.parquet")
    ap.add_argument("--events-dir", default="data/a2/ami")
    ap.add_argument("--out", default="data/b1/items_extra.parquet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n0 = len(pd.read_parquet(a.items, columns=["item_id"]))
    rows = []
    n_ev = 0
    for f in sorted(Path(a.events_dir).glob("*/events.jsonl")):
        for line in open(f):
            e = json.loads(line)
            if e["reason"] != "long_silence" or e["gap_s"] is None:
                continue
            if e["b_onset_s"] is not None and e["b_onset_s"] <= e["gap_s"]:
                continue        # somebody else spoke first: not a pure thinking pause
            n_ev += 1
            full = e["split"] != "train"
            base = dict(source="ami", split=e["split"], t_off=e["t"], meeting=e["meeting"], speaker=e["speaker"],
                        channel=int(e["channel"]), audio=e["audio"], gap_s=e["gap_s"], b_onset_s=e["b_onset_s"],
                        a_turn_s=e["a_turn_s"], extra="long_pause")
            taus = list(TAUS) if full else sorted(rng.choice(TAUS, 4, replace=False).tolist())
            for t in taus:
                rows.append(dict(base, kind="tau", value=float(t), end_s=e["t"] + t, cut_s=np.nan, p_end=0, p_eager=-1))
            deltas = [d for d in DELTAS if e["a_turn_s"] >= d + MIN_SPEECH_BEFORE_S]
            if not full and len(deltas) > 3:
                deltas = sorted(rng.choice(deltas, 3, replace=False).tolist())
            for d in deltas:
                rows.append(dict(base, kind="pre", value=float(d), end_s=e["t"] - d, cut_s=np.nan, p_end=0, p_eager=0))
            lo, hi = e["t"] - e["a_turn_s"] + 1.0, e["t"] - max(DELTAS) - 0.2
            if hi > lo:
                m = float(rng.uniform(lo, hi))
                rows.append(dict(base, kind="mid", value=m, end_s=m, cut_s=np.nan, p_end=0, p_eager=0))
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", np.arange(n0, n0 + len(df)))
    df.to_parquet(a.out, index=False)
    print(f"{n_ev} long-pause events -> {len(df)} items (ids {n0}..{n0 + len(df) - 1}) -> {a.out}")
    print(df.groupby(["split", "kind"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
