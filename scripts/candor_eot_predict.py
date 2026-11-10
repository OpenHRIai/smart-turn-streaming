"""CANDOR end-of-turn test: eot-bench-format predictions for Smart Turn, smart-turn-streaming and a Parakeet head.

    modal volume get candor eot_test/whisper.npz  ../OpenVideoFDB/data/candor/eot_test/whisper.npz
    modal volume get candor eot_test/parakeet.npz ../OpenVideoFDB/data/candor/eot_test/parakeet.npz
    PYTHONPATH=. ../eot-bench/.venv/bin/python scripts/candor_eot_predict.py \
        --dir ../OpenVideoFDB/data/candor/eot_test --parakeet-heads runs/parakeet/pk_ami07_auc/heads.pt

Inputs: ``items.parquet`` (scripts/candor_eot_items.py), ``whisper.npz`` (modal_features.py --what candor:
Smart Turn v3.2's own P(complete) ``p_v32`` and the released heads' inputs ``pooled_v32`` + ``mel20``) and
``parakeet.npz`` (modal_parakeet_features.py --what candor). Every model scores the same window at every grid
point: the 8 s ending at that point (Smart Turn, smart-turn-streaming), or ending 0-160 ms earlier at a
streaming chunk boundary (Parakeet). So smart-turn-streaming gets a freshly computed encoder vector at every
point, which its live version refreshes less often (not measured how much that matters).

Writes ``<dir>/predictions/<model>/predictions.parquet`` with eot-bench's columns (one span per event,
``id`` = conversation/participant/time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eot.model import tau_features


def rows(items: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"id": items.event_id.values, "language": "en", "span_index": 0,
                         "timestamp": (items.t_off + items.value).values, "silence_dur": items.value.round(6).values,
                         "p_eot": p.astype(np.float64), "label": items.label.values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--parakeet-heads", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    items = pd.read_parquet(d / "items.parquet")
    tau = tau_features(items.kind.values, items.value.values)
    out = {}

    w = np.load(d / "whisper.npz")
    assert (w["item_id"] == items.item_id.values).all()
    out["smart_turn_v32"] = w["p_v32"]
    import onnxruntime as ort
    from eot.hub import model_paths
    sess = ort.InferenceSession(model_paths()[1])
    out["smart_turn_streaming"] = sess.run(None, {"pooled": w["pooled_v32"].astype(np.float32),
                                                  "mel20": w["mel20"].astype(np.float32), "tau_feat": tau})[0][:, 0]

    if a.parakeet_heads:
        import torch
        from eot.model import Heads
        z = np.load(d / "parakeet.npz")
        assert (z["item_id"] == items.item_id.values).all()
        ckpt = torch.load(a.parakeet_heads, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        heads = Heads(d_pooled=512, d_mel=512, use_mel=not cfg.get("no_mel", False), use_tau=not cfg.get("no_tau", False),
                      dropout=cfg.get("dropout", 0.1))
        heads.load_state_dict(ckpt["state"])
        heads.eval()
        with torch.inference_mode():
            logits = heads(torch.from_numpy(z["pooled"]), torch.from_numpy(z["last"]), torch.from_numpy(tau))
        out[f"parakeet_{Path(a.parakeet_heads).parent.name}"] = torch.sigmoid(logits[:, 0]).numpy()

    for name, p in out.items():
        path = d / "predictions" / name / "predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows(items, p).to_parquet(path, index=False)
        print(f"{name}: {len(p)} rows, mean p {p.mean():.3f} -> {path}")


if __name__ == "__main__":
    main()
