"""Stage A1 baseline: how the current audio-only model scores tau-padded and pre-offset windows.

    .venv/bin/python scripts/a1_tau_baseline.py --parquet-dir data/smart_turn_data/test \
        --offsets data/a1/test_offsets.parquet --model ../smart-turn-multimodal/smart-turn-multimodal-audio.onnx \
        --n 600 --out data/a1/tau_baseline_test.csv

For a stratified sample of clips (half complete, half incomplete) it builds every ``tau`` and
``pre`` window with ``eot.windows``, scores it with the ONNX graph (Whisper features exactly as
``train.py``), and prints mean P(complete) per label and AUC per tau / delta. This is the number
the tau-aware model (Stage B1) has to improve: today the score should drift upwards with tau on
incomplete clips, which is the failure seen on Seamless.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import windows as W  # noqa: E402
from eot.offsets import SR  # noqa: E402


def load_clip(shard_path: Path, row: int) -> np.ndarray:
    import soundfile as sf
    pf = pq.ParquetFile(shard_path)
    # locate the row group holding `row`
    start = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if row < start + n:
            tbl = pf.read_row_group(rg, columns=["audio"])
            audio = tbl.column("audio")[row - start].as_py()
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                import torch, torchaudio
                x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
            return x.astype(np.float32)
        start += n
    raise IndexError(row)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(a.model, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    off = pd.read_parquet(a.offsets)
    off = off[off.error.isna() & off.t_off.notna()]
    rng = np.random.default_rng(a.seed)
    per = a.n // 2
    pick = pd.concat([off[off.endpoint_bool].sample(min(per, int(off.endpoint_bool.sum())), random_state=a.seed),
                      off[~off.endpoint_bool].sample(min(per, int((~off.endpoint_bool).sum())), random_state=a.seed)])
    shards = {p.name: p for p in Path(a.parquet_dir).rglob("*.parquet")}

    def score(wins: list[np.ndarray]) -> np.ndarray:
        feats = fe(wins, sampling_rate=SR, return_tensors="np", padding="max_length",
                   max_length=8 * SR, truncation=True, do_normalize=True).input_features.astype(np.float32)
        out = []
        for i in range(0, len(feats), 16):
            out.append(sess.run(None, {inp: feats[i:i + 16]})[0].reshape(-1))
        return np.concatenate(out)

    rows = []
    for k, (_, r) in enumerate(pick.iterrows()):
        x = load_clip(shards[r.shard], int(r.row))
        items = [dict(kind="tau", value=t) for t in W.TAUS] + \
                [dict(kind="pre", value=d) for d in W.DELTAS if r.t_off - d >= r.t_on + W.MIN_SPEECH_BEFORE_S]
        items.append(dict(kind="clip", value=0.0))
        wins = []
        for it in items:
            if it["kind"] == "clip":
                wins.append(W.cut_window(x, SR, len(x) / SR, r.t_off, rng))  # the clip as stored (last 8 s)
            else:
                wins.append(W.variant(x, SR, r.t_off, it["kind"], it["value"], rng))
        ps = score(wins)
        for it, p in zip(items, ps):
            rows.append(dict(id=r.id, label=int(r.endpoint_bool), language=r.language, dataset=r.dataset,
                             synthetic=bool(r.synthetic), trailing_s=r.trailing_s, kind=it["kind"],
                             value=it["value"], p=float(p)))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(pick)} clips", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n## current model vs tau (windows ending tau s after the acoustic offset)\n")
    print("| window | n | mean P, complete | mean P, incomplete | AUC | recall @0.5 | FAR @0.5 |")
    print("|---|---|---|---|---|---|---|")
    g = df[df.kind == "clip"]
    def line(name, g):
        c, i = g[g.label == 1].p, g[g.label == 0].p
        print(f"| {name} | {len(g)} | {c.mean():.3f} | {i.mean():.3f} | {auc(g.label.values, g.p.values):.3f} | "
              f"{(c >= 0.5).mean():.2f} | {(i >= 0.5).mean():.2f} |")
    line("stored clip (last 8 s)", g)
    for t in W.TAUS:
        line(f"tau = {t:.1f} s", df[(df.kind == "tau") & (df.value == t)])
    print("\n## pre-offset windows (ending delta s before the offset; label = clip label)\n")
    print("| delta | n | mean P, complete clips | mean P, incomplete clips | AUC |")
    print("|---|---|---|---|---|")
    for d in W.DELTAS:
        g = df[(df.kind == "pre") & (df.value == d)]
        if len(g):
            print(f"| {d:.1f} s | {len(g)} | {g[g.label == 1].p.mean():.3f} | {g[g.label == 0].p.mean():.3f} | "
                  f"{auc(g.label.values, g.p.values):.3f} |")


if __name__ == "__main__":
    main()
