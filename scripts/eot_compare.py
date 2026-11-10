"""Compare end-of-turn models on eot-bench-format predictions, with a paired bootstrap over groups.

    PYTHONPATH=../eot-bench ../eot-bench/.venv/bin/python scripts/eot_compare.py \
        --pred detector=a/predictions.parquet --pred head=b/predictions.parquet --ref detector \
        [--group-sep /] [--n-boot 2000] [--out report.md]

Point estimates are eot-bench's own (``compute_metrics_from_predictions`` + ``get_pareto_operating_points``,
first-threshold-crossing scoring, pauses 0.2-5 s). The bootstrap resamples groups (eot-bench: turns; CANDOR
with ``--group-sep /``: conversations, the first part of the span id) with the same draw for every model and
re-optimises the policy in each draw, so differences are paired.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from eot_harness import metrics as M

H = dict(min_hold_span_duration=0.2, max_hold_span_duration=5.0)
TH = np.round(np.arange(0, 1.0001, 0.01), 2)
DL = np.array(M.DEFAULT_ACTION_DELAYS)
TO = np.array(M.DEFAULT_TIMEOUTS)


def matrices(pred: pd.DataFrame, groups: list, group_of) -> tuple:
    """Per policy (threshold, action delay, timeout) and group: cutoff counts and latency sums."""
    spans = M.span_table_from_predictions(pred, **H)
    rows = M._prediction_rows_for_spans(pred, spans)
    idx = {g: i for i, g in enumerate(groups)}
    h = spans[spans.label == "hold"].reset_index(drop=True)
    e = spans[spans.label == "eot"].reset_index(drop=True)
    hd = h.span_dur.to_numpy(float)
    hg = h.id.map(group_of).map(idx).to_numpy()
    eg = e.id.map(group_of).map(idx).to_numpy()
    C, L = [], []
    for th in TH:
        cross = M._first_crossing_times(rows, threshold=float(th))
        hf, ef = M._fire_times_from_crossing(h, cross), M._fire_times_from_crossing(e, cross)
        for d in DL:
            mc = np.isfinite(hf) & (hd > np.maximum(d, hf) + 1e-9)
            eft = np.maximum(d, ef)
            for t in TO:
                if t + 1e-9 < d:
                    continue
                cut = (hd > t + 1e-9) | mc
                lat = np.where(np.isfinite(ef) & (eft <= t + 1e-9), eft, t)
                C.append(np.bincount(hg, cut, len(groups)))
                L.append(np.bincount(eg, lat, len(groups)))
    return (np.array(C, np.float32), np.array(L, np.float32), np.bincount(hg, minlength=len(groups)),
            np.bincount(eg, minlength=len(groups)), len(h), len(e))


def best_cutoff(C, L, Nh, Ne, W, budget):
    cr = (C @ W.T) / (Nh @ W.T)
    la = (L @ W.T) / (Ne @ W.T)
    cr[la > budget + 1e-9] = np.inf
    return cr.min(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path/to/predictions.parquet")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--group-sep", default=None, help="group = span id up to the first separator (default: the id)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Comparison")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    preds = {k: M.load_predictions(v) for k, v in (p.split("=", 1) for p in a.pred)}
    group_of = (lambda i: str(i).split(a.group_sep)[0]) if a.group_sep else (lambda i: str(i))
    ids = {k: set(p.id) for k, p in preds.items()}
    assert all(s == ids[a.ref] for s in ids.values()), "models were not scored on the same spans"
    groups = sorted({group_of(i) for i in ids[a.ref]})

    out = [f"# {a.title}", ""]
    point = ["| model | AUC | AP | false cutoffs @300 ms | @600 ms | latency @5% cutoffs | @10% |", "|---|---|---|---|---|---|---|"]
    mats = {}
    for name, pred in preds.items():
        trade, summary = M.compute_metrics_from_predictions(pred, **H)
        lat = M.get_pareto_operating_points(trade, "mean_latency", (0.3, 0.6))
        cut = M.get_pareto_operating_points(trade, "cutoff_rate", (0.05, 0.10))

        def cell(df, i, col, scale, fmt, unit):
            r = df.iloc[i]
            return format(r[col] * scale, fmt) + unit if bool(r["feasible"]) else "–"

        point.append(f"| {name} | {summary['auc']:.3f} | {summary['ap']:.3f} | {cell(lat, 0, 'cutoff_rate', 100, '.1f', '%')} | "
                     f"{cell(lat, 1, 'cutoff_rate', 100, '.1f', '%')} | {cell(cut, 0, 'mean_latency', 1000, '.0f', ' ms')} | "
                     f"{cell(cut, 1, 'mean_latency', 1000, '.0f', ' ms')} |")
        mats[name] = matrices(pred, groups, group_of)
    nh, ne = mats[a.ref][4], mats[a.ref][5]
    out += [f"{nh} pauses (0.2-5 s) and {ne} turn ends in {len(groups)} groups; {a.n_boot} paired bootstrap draws over groups.", "",
            *point, ""]

    rng = np.random.default_rng(a.seed)
    W = rng.multinomial(len(groups), np.ones(len(groups)) / len(groups), a.n_boot).astype(np.float32)
    one = np.ones((1, len(groups)), np.float32)
    boot = ["| model | budget | false cutoffs (point, own check) | 95% interval | minus " + a.ref + " | 95% interval | P(better) |",
            "|---|---|---|---|---|---|---|"]
    ref = {b: best_cutoff(*mats[a.ref][:4], W, b) for b in (0.3, 0.6)}
    for name in preds:
        C, L, Nh, Ne = mats[name][:4]
        for b in (0.3, 0.6):
            p = best_cutoff(C, L, Nh, Ne, one, b)[0]
            x = best_cutoff(C, L, Nh, Ne, W, b)
            if not np.isfinite(p):
                boot.append(f"| {name} | {int(b * 1000)} ms | – | | | | |")
                continue
            lo, hi = np.percentile(x, [2.5, 97.5]) * 100
            if name == a.ref:
                boot.append(f"| {name} | {int(b * 1000)} ms | {p * 100:.1f}% | {lo:.1f}–{hi:.1f}% | | | |")
                continue
            d = (x - ref[b]) * 100
            dl, dh = np.percentile(d, [2.5, 97.5])
            boot.append(f"| {name} | {int(b * 1000)} ms | {p * 100:.1f}% | {lo:.1f}–{hi:.1f}% | {d.mean():+.1f} | "
                        f"{dl:+.1f} to {dh:+.1f} | {(d < 0).mean():.3f} |")
    out += boot
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
