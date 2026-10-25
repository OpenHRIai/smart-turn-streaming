"""Stage C3: a monotone calibration of the heads' P(complete) for the clip-end regime.

    .venv/bin/python scripts/c3_calibrate.py --run runs/b1/knob_ami07_auc --out runs/c3/calibration.json

The heads were trained with a class-weighted loss, so their probabilities sit above the frozen
model's at the same ranking: at threshold 0.5 on smart-turn-data test they have more recall and
more false positives than Smart Turn v3.2 (B1 regression guard). Pipecat's official benchmark fixes
the threshold at 0.5, so for that comparison (and for users who try 0.5 first) this fits Platt
scaling, ``p' = sigmoid(a * logit(p) + b)``, on the **training** split's clip-end windows (τ = 0 and
0.2 s) and reports the effect on the test split. A monotone map: every AUC and every fitted table is
unchanged (a table threshold θ maps to sigmoid(a * logit(θ) + b)).

Inference re-runs the heads on the cached frozen-encoder features (data/b1/feats.npz), not on audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.model import Heads, tau_features  # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(z[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics(p, y, th=0.5):
    from sklearn.metrics import roc_auc_score
    pred = p > th
    return dict(auc=float(roc_auc_score(y, p)), acc=float((pred == (y == 1)).mean()),
                recall=float(pred[y == 1].mean()), fpr=float(pred[y == 0].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/b1/knob_ami07_auc")
    ap.add_argument("--items", default="data/b1/items.parquet")
    ap.add_argument("--feats", default="data/b1/feats.npz")
    ap.add_argument("--fit-taus", default="0.0,0.2")
    ap.add_argument("--out", default="runs/c3/calibration.json")
    a = ap.parse_args()
    fit_taus = [float(t) for t in a.fit_taus.split(",")]

    it = pd.read_parquet(a.items)
    std = it[(it.source == "std") & (it.kind == "tau")]
    sel = std[std.value.isin(fit_taus + [2.5, 1.2])]
    idx = sel.index.values
    print(f"{len(idx)} smart-turn-data tau windows (train {int((sel.split == 'train').sum())}, test {int((sel.split == 'test').sum())})")

    z = np.load(a.feats)
    pooled = z["pooled_v32"][idx]; mel = z["mel20"][idx]; pb = z["p_v32"][idx].astype(np.float32)
    tau = tau_features(sel.kind.values, sel.value.values)
    ck = torch.load(Path(a.run) / "heads.pt", map_location="cpu")
    cfg = ck["config"]
    model = Heads(use_mel=not cfg.get("no_mel", False), use_tau=not cfg.get("no_tau", False), dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ck["state"]); model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 8192):
            o = model(torch.from_numpy(pooled[i:i + 8192].astype(np.float32)), torch.from_numpy(mel[i:i + 8192].astype(np.float32)),
                      torch.from_numpy(tau[i:i + 8192]))
            out.append(o[:, 0].numpy())
    zlog = np.concatenate(out).astype(np.float64)             # raw logit of p_end
    p = sigmoid(zlog)
    y = sel.p_end.values.astype(int)
    tr = (sel.split == "train").values & sel.value.isin(fit_taus).values
    te = (sel.split == "test").values

    a_, b_ = fit_platt(zlog[tr], y[tr])
    pc = sigmoid(a_ * zlog + b_)
    print(f"Platt on {int(tr.sum())} train windows at tau {fit_taus}: a = {a_:.4f}, b = {b_:.4f}")
    rows = ["| τ | model | AUC | acc@0.5 | recall@0.5 | FPR@0.5 |", "|---|---|---|---|---|---|"]
    for t in sorted(set(fit_taus + [1.2, 2.5])):
        m = te & (sel.value.values == t)
        for name, pp in (("heads, raw", p[m]), ("heads, calibrated", pc[m]), ("frozen v3.2", pb[m])):
            r = metrics(pp, y[m])
            rows.append(f"| {t:g} | {name} | {r['auc']:.3f} | {r['acc']:.3f} | {r['recall']:.3f} | {r['fpr']:.3f} |")
    table = "\n".join(rows)
    print(table)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(a=a_, b=b_, fit="platt on smart-turn-data v3.2 train tau windows", fit_taus=fit_taus, run=a.run,
                   n_fit=int(tr.sum()), test_table_md=table), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
