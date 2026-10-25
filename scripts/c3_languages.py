"""Stage C3: per-language behaviour of the heads on smart-turn-data v3.2 test, from the B1 predictions.

    .venv/bin/python scripts/c3_languages.py --run runs/b1/knob_ami07_auc --out runs/c3/languages.md

Two questions per language: does the ranking hold at the clip end (AUC at τ = 0, the regime of
pipecat's official benchmark), and does the decision hold through silence after an unfinished
utterance (false-alarm rate on incomplete clips padded with their own room tone to τ = 2.5 s, at the
threshold that gives 5 % false alarms at τ = 0 over the whole test set, per model). The frozen Smart
Turn v3.2 is scored the same way from its own probability. No new inference: the heads' predictions are
``predictions.npz`` of the B1 run, the frozen model's are ``p_v32`` in the feature cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANG = {"eng": "English", "spa": "Spanish", "rus": "Russian", "por": "Portuguese", "nld": "Dutch", "deu": "German",
        "fra": "French", "hin": "Hindi", "pol": "Polish", "zho": "Chinese", "ind": "Indonesian", "nor": "Norwegian",
        "kor": "Korean", "tur": "Turkish", "fin": "Finnish", "ben": "Bengali", "vie": "Vietnamese", "ukr": "Ukrainian",
        "ara": "Arabic", "jpn": "Japanese", "ita": "Italian", "dan": "Danish", "mar": "Marathi"}


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/b1/knob_ami07_auc")
    ap.add_argument("--items", default="data/b1/items.parquet")
    ap.add_argument("--feats", default="data/b1/feats.npz")
    ap.add_argument("--far0", type=float, default=0.05, help="false-alarm rate at tau 0 that sets each model's threshold")
    ap.add_argument("--out", default="runs/c3/languages.md")
    a = ap.parse_args()

    it = pd.read_parquet(a.items)
    z = np.load(Path(a.run) / "predictions.npz")
    idx = z["test_std"]
    d = it.iloc[idx].copy()
    d["p"] = z["p_test_std"][:, 0]
    d["pb"] = np.load(a.feats)["p_v32"][idx].astype(np.float32)
    d = d[d.kind == "tau"]
    taus = (0.0, 0.2, 1.2, 2.5)

    # thresholds at matched false-alarm rate at tau 0, over all languages
    inc0 = d[(d.value == 0.0) & (d.p_end == 0)]
    th = {"p": float(np.quantile(inc0.p, 1 - a.far0)), "pb": float(np.quantile(inc0.pb, 1 - a.far0))}

    rows = []
    for lang, g in d.groupby("language"):
        r = {"lang": lang, "n_clips": g[g.value == 0.0].shape[0]}
        for t in taus:
            gt = g[g.value == t]
            r[f"auc_{t}"] = auc(gt.p_end, gt.p); r[f"aucb_{t}"] = auc(gt.p_end, gt.pb)
        inc = g[g.p_end == 0]
        for t in (0.0, 2.5):
            gt = inc[inc.value == t]
            r[f"far_{t}"] = float((gt.p >= th["p"]).mean()); r[f"farb_{t}"] = float((gt.pb >= th["pb"]).mean())
        com = g[(g.p_end == 1) & (g.value == 0.0)]
        r["rec_0"] = float((com.p >= th["p"]).mean()); r["recb_0"] = float((com.pb >= th["pb"]).mean())
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("n_clips", ascending=False)

    # whole-set row
    def whole(col, colb, t, lab=None):
        gt = d[d.value == t] if lab is None else d[(d.value == t) & (d.p_end == lab)]
        return gt

    out = [f"# C3: per-language behaviour on smart-turn-data v3.2 test (heads `{a.run}` vs frozen Smart Turn v3.2)", "",
           f"{len(R)} languages, {int(R.n_clips.sum())} clips. AUC on windows ending τ seconds after the acoustic offset (clips padded "
           "with their own room tone). FAR: false-alarm rate on incomplete clips at the threshold that gives "
           f"{a.far0:.0%} false alarms at τ = 0 over the whole set (heads θ = {th['p']:.3f}, frozen θ = {th['pb']:.3f}); recall at that "
           "threshold at τ = 0. Each cell: heads / frozen.", "",
           "| language | clips | AUC τ 0 | AUC τ 0.2 | AUC τ 1.2 | AUC τ 2.5 | recall τ 0 | FAR τ 0 | FAR τ 2.5 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        out.append(f"| {LANG.get(r['lang'], r['lang'])} | {int(r['n_clips'])} | "
                   + " | ".join(f"{r[f'auc_{t}']:.3f} / {r[f'aucb_{t}']:.3f}" for t in taus)
                   + f" | {r['rec_0']:.2f} / {r['recb_0']:.2f} | {r['far_0.0']:.3f} / {r['farb_0.0']:.3f} | {r['far_2.5']:.3f} / {r['farb_2.5']:.3f} |")
    # all languages
    allr = ["all", int(R.n_clips.sum())]
    cells = []
    for t in taus:
        gt = d[d.value == t]; cells.append(f"{auc(gt.p_end, gt.p):.3f} / {auc(gt.p_end, gt.pb):.3f}")
    com = d[(d.p_end == 1) & (d.value == 0.0)]; inc = d[d.p_end == 0]
    cells.append(f"{(com.p >= th['p']).mean():.2f} / {(com.pb >= th['pb']).mean():.2f}")
    for t in (0.0, 2.5):
        gt = inc[inc.value == t]; cells.append(f"{(gt.p >= th['p']).mean():.3f} / {(gt.pb >= th['pb']).mean():.3f}")
    out.append(f"| **all** | {allr[1]} | " + " | ".join(cells) + " |")
    out += ["", "Reading guide: a FAR at τ 2.5 well above the FAR at τ 0 means the model drifts towards 'complete' as the "
            "silence after an unfinished utterance grows; the heads were trained to hold, the frozen model was not."]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out) + "\n")
    R.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
