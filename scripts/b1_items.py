"""Stage A3: enumerate the training/evaluation windows ("items") from both sources.

    .venv/bin/python scripts/b1_items.py --out data/b1/items.parquet [--std-train-tts 30000] [--seed 0]

One row per 8 s window to be featurised. Columns:

    item_id, source ('std' | 'ami'), split ('train' | 'dev' | 'test'), kind ('tau' | 'pre' | 'mid'),
    value (tau or delta in s; mid: the window end in file seconds), end_s (window end, file seconds),
    t_off (the acoustic offset / VAD stop), cut_s (audio after this time is replaced by room tone;
    NaN = none), p_end, p_eager (-1 = undefined), and the source reference:
      std: shard, row, clip_id, language, dataset, synthetic
      ami: meeting, speaker, channel, audio, gap_s, b_onset_s, a_turn_s

Sampling (seeded):
  std train : every human-recorded clip + ``--std-train-tts`` TTS clips; per clip 4 of the 9 taus,
              3 of the 6 deltas, 1 mid window.
  std test  : ``--std-test-clips`` clips stratified by label; the full tau and delta sets + 1 mid.
  ami train : every labelled silence; 4 taus (tau <= gap for incompletes), 3 deltas, 1 mid.
  ami dev/test: every labelled silence; full sets.
For AMI completes a tau window that runs past the next speaker's onset is cut there
(``cut_s = t + b_onset_s``) and continued with room tone, so the window shows "nobody has answered
yet", which is the situation a deployed model is in; it never hears the other person's reply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.windows import DELTAS, EAGER_HORIZON_S, MIN_SPEECH_BEFORE_S, TAUS  # noqa: E402

HUMAN_STD = {"liva_1", "midcentury_1", "mundo_1", "human_5", "human_convcollector_1"}


def std_items(off: pd.DataFrame, split: str, full: bool, rng: np.random.Generator) -> list[dict]:
    rows = []
    for r in off.itertuples(index=False):
        label = int(r.endpoint_bool)
        base = dict(source="std", split=split, t_off=r.t_off, cut_s=np.nan, shard=r.shard, row=int(r.row),
                    clip_id=r.id, language=r.language, dataset=r.dataset, synthetic=bool(r.synthetic))
        taus = TAUS if full else sorted(rng.choice(TAUS, 4, replace=False).tolist())
        for t in taus:
            rows.append(dict(base, kind="tau", value=float(t), end_s=r.t_off + t, p_end=label, p_eager=-1))
        deltas = [d for d in DELTAS if r.t_off - d >= r.t_on + MIN_SPEECH_BEFORE_S]
        if not full and len(deltas) > 3:
            deltas = sorted(rng.choice(deltas, 3, replace=False).tolist())
        for d in deltas:
            rows.append(dict(base, kind="pre", value=float(d), end_s=r.t_off - d, p_end=0,
                             p_eager=int(label == 1 and d <= EAGER_HORIZON_S)))
        lo, hi = r.t_on + 1.0, r.t_off - max(DELTAS) - 0.2
        if hi > lo:
            e = float(rng.uniform(lo, hi))
            rows.append(dict(base, kind="mid", value=e, end_s=e, p_end=0, p_eager=0))
    return rows


def ami_items(man: pd.DataFrame, split: str, full: bool, rng: np.random.Generator) -> list[dict]:
    rows = []
    for r in man.itertuples(index=False):
        label = int(r.label == "complete")
        base = dict(source="ami", split=split, t_off=r.t, meeting=r.meeting, speaker=r.speaker, channel=int(r.channel),
                    audio=r.audio, gap_s=r.gap_s, b_onset_s=r.b_onset_s, a_turn_s=r.a_turn_s)
        if label:
            valid = list(TAUS)
        else:
            valid = [t for t in TAUS if t < r.gap_s]          # the silence must still be going on
        taus = valid if (full or len(valid) <= 4) else sorted(rng.choice(valid, 4, replace=False).tolist())
        for t in taus:
            cut = r.t + r.b_onset_s if (label and r.b_onset_s is not None and t > r.b_onset_s) else np.nan
            rows.append(dict(base, kind="tau", value=float(t), end_s=r.t + t, cut_s=cut, p_end=label, p_eager=-1))
        deltas = [d for d in DELTAS if r.a_turn_s >= d + MIN_SPEECH_BEFORE_S]
        if not full and len(deltas) > 3:
            deltas = sorted(rng.choice(deltas, 3, replace=False).tolist())
        for d in deltas:
            rows.append(dict(base, kind="pre", value=float(d), end_s=r.t - d, cut_s=np.nan, p_end=0,
                             p_eager=int(label == 1 and d <= EAGER_HORIZON_S)))
        lo, hi = r.t - r.a_turn_s + 1.0, r.t - max(DELTAS) - 0.2
        if hi > lo:
            e = float(rng.uniform(lo, hi))
            rows.append(dict(base, kind="mid", value=e, end_s=e, cut_s=np.nan, p_end=0, p_eager=0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/b1/items.parquet")
    ap.add_argument("--std-train-tts", type=int, default=30000)
    ap.add_argument("--std-test-clips", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    tr = pd.read_parquet("data/a1/train_offsets.parquet")
    tr = tr[tr.error.isna() & tr.t_off.notna()]
    human = tr[tr.dataset.isin(HUMAN_STD)]
    tts = tr[~tr.dataset.isin(HUMAN_STD)].sample(a.std_train_tts, random_state=a.seed)
    std_tr = pd.concat([human, tts])
    te = pd.read_parquet("data/a1/test_offsets.parquet")
    te = te[te.error.isna() & te.t_off.notna()]
    per = a.std_test_clips // 2
    std_te = pd.concat([te[te.endpoint_bool].sample(per, random_state=a.seed),
                        te[~te.endpoint_bool].sample(per, random_state=a.seed)])

    rows = std_items(std_tr, "train", False, rng) + std_items(std_te, "test", True, rng)
    for split in ("train", "dev", "test"):
        man = pd.read_parquet(f"data/a2/ami/manifest_{split}.parquet")
        rows += ami_items(man, split, split != "train", rng)
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", np.arange(len(df)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)

    print(f"{len(df)} items -> {a.out}\n")
    print("| source | split | clips/events | tau | pre | mid | p_end=1 | p_eager=1 |")
    print("|---|---|---|---|---|---|---|---|")
    for (s, sp), g in df.groupby(["source", "split"]):
        n_src = g.clip_id.nunique() if s == "std" else g.groupby(["meeting", "speaker", "t_off"]).ngroups
        k = g.kind.value_counts()
        print(f"| {s} | {sp} | {n_src} | {k.get('tau', 0)} | {k.get('pre', 0)} | {k.get('mid', 0)} | "
              f"{int((g.p_end == 1).sum())} | {int((g.p_eager == 1).sum())} |")
    print(f"\nAMI complete tau windows cut at the next speaker's onset: {int(df.cut_s.notna().sum())}")


if __name__ == "__main__":
    main()
