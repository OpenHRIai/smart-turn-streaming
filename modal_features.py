"""Stage B1 feature cache on Modal: frozen-encoder features for every item of data/b1/items.parquet.

    modal volume put smart-turn-v2 data/b1/items.parquet data/b1/items.parquet
    modal volume put smart-turn-v2 data/models/smart-turn-v3.2-gpu.pooled.onnx data/models/smart-turn-v3.2-gpu.pooled.onnx
    modal volume put smart-turn-v2 data/models/smart-turn-multimodal-audio.pooled.onnx data/models/smart-turn-multimodal-audio.pooled.onnx
    modal run --detach modal_features.py [--max-containers 8]
    modal volume get smart-turn-v2 data/b1/feats.npz data/b1/feats.npz

Work is split into tasks: one per smart-turn-data parquet shard, and one per group of AMI headset
files. Each task cuts the 8 s windows (``eot.windows`` semantics, room tone after ``cut_s``),
computes Whisper log-mel on the GPU exactly as the trainer does (``WhisperFeatureExtractor``,
``do_normalize=True``), runs both frozen backbones through ONNX Runtime (CUDA) and writes an
``.npz`` with, per item: ``item_id``, ``pooled_v32`` / ``pooled_mm`` (fp16, 384), ``mel20``
(fp16, 20x80), ``p_v32`` / ``p_mm`` (fp32). ``merge`` concatenates the task files into one
``feats.npz`` ordered by item_id.
"""

from __future__ import annotations

import modal

app = modal.App("smart-turn-v2-features")
volume = modal.Volume.from_name("smart-turn-v2")
VOL = "/vol"
BATCH = 96

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0", "torchaudio==2.9.0", "transformers==4.48.2", "onnxruntime-gpu==1.23.2",
        "numpy==2.3.4", "soundfile", "pyarrow", "pandas",
    )
    .add_local_python_source("eot")
)

MODELS = {"v32": "data/models/smart-turn-v3.2-gpu.pooled.onnx", "mm": "data/models/smart-turn-multimodal-audio.pooled.onnx"}


def _sessions():
    import onnxruntime as ort
    from eot.features import FeatureSession
    avail = ort.get_available_providers()
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    return {k: FeatureSession(f"{VOL}/{p}", providers=prov) for k, p in MODELS.items()}, prov


def _featurize(wins, fe, sess, device):
    import numpy as np
    import torch
    from eot.features import mel_summary
    feats = fe(wins, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=8 * 16000,
               truncation=True, do_normalize=True, device=device).input_features
    feats = feats.float().cpu().numpy()
    out = {"mel20": mel_summary(feats).astype(np.float16)}
    for k, s in sess.items():
        p, pooled = s.run(feats)
        out[f"p_{k}"] = p.astype(np.float32)
        out[f"pooled_{k}"] = pooled.astype(np.float16)
    return out


def _cut(x, sr, end_s, t_off, cut_s, rng):
    """Window ending at end_s; audio after cut_s (if any) is replaced by room tone from [t_off, cut_s]."""
    import math
    from eot.windows import cut_window
    if cut_s is not None and not (isinstance(cut_s, float) and math.isnan(cut_s)):
        x = x[: int(round(cut_s * sr))]
    return cut_window(x, sr, end_s, t_off, rng)


def _save(path, parts):
    import numpy as np
    import os
    keys = parts[0].keys()
    arr = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arr)
    return len(arr["item_id"])


@app.function(image=image, gpu="L4", volumes={VOL: volume}, timeout=3 * 3600, memory=16384, max_containers=8)
def featurize_std(shard: str, items_path: str = "data/b1/items.parquet", out_dir: str = "data/b1/feats") -> str:
    import io
    import time
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from transformers import WhisperFeatureExtractor
    t0 = time.time()
    items = pd.read_parquet(f"{VOL}/{items_path}")
    items = items[(items.source == "std") & (items.shard == shard)].sort_values("row")
    if items.empty:
        return f"{shard}: no items"
    split = items.split.iloc[0]
    path = f"{VOL}/data/smart_turn_data/{split}/data/{shard}"
    pf = pq.ParquetFile(path)
    by_row = {r: g for r, g in items.groupby("row")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = WhisperFeatureExtractor(chunk_length=8)
    sess, prov = _sessions()
    rng = np.random.default_rng(0)
    parts, wins, ids = [], [], []

    def flush():
        if wins:
            out = _featurize(wins, fe, sess, device)
            out["item_id"] = np.array(ids, dtype=np.int64)
            parts.append(out)
            wins.clear(); ids.clear()

    row = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        need = [r for r in range(row, row + n) if r in by_row]
        if need:
            tbl = pf.read_row_group(rg, columns=["audio"])
            col = tbl.column("audio")
            for r in need:
                audio = col[r - row].as_py()
                x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr != 16000:
                    import torchaudio
                    x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
                for it in by_row[r].itertuples(index=False):
                    wins.append(_cut(x, 16000, float(it.end_s), float(it.t_off), it.cut_s, rng))
                    ids.append(int(it.item_id))
                    if len(wins) >= BATCH:
                        flush()
        row += n
    flush()
    n = _save(f"{VOL}/{out_dir}/std_{shard}.npz", parts)
    volume.commit()
    return f"{shard}: {n} windows, {time.time() - t0:.0f} s, {prov[0]}"


@app.function(image=image, gpu="L4", volumes={VOL: volume}, timeout=3 * 3600, memory=16384, max_containers=8)
def featurize_ami(audio_paths: list[str], items_path: str = "data/b1/items.parquet", out_dir: str = "data/b1/feats") -> str:
    import time
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import WhisperFeatureExtractor
    t0 = time.time()
    items = pd.read_parquet(f"{VOL}/{items_path}")
    items = items[(items.source == "ami") & items.audio.isin(audio_paths)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = WhisperFeatureExtractor(chunk_length=8)
    sess, prov = _sessions()
    rng = np.random.default_rng(0)
    parts, wins, ids = [], [], []

    def flush():
        if wins:
            out = _featurize(wins, fe, sess, device)
            out["item_id"] = np.array(ids, dtype=np.int64)
            parts.append(out)
            wins.clear(); ids.clear()

    for ap, g in items.groupby("audio"):
        x, sr = sf.read(f"{VOL}/{ap}", dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        for it in g.sort_values("end_s").itertuples(index=False):
            wins.append(_cut(x, 16000, float(it.end_s), float(it.t_off), it.cut_s, rng))
            ids.append(int(it.item_id))
            if len(wins) >= BATCH:
                flush()
    flush()
    tag = audio_paths[0].split("/")[-1].split(".")[0]
    n = _save(f"{VOL}/{out_dir}/ami_{tag}_{len(audio_paths)}.npz", parts)
    volume.commit()
    return f"ami {tag}+{len(audio_paths) - 1}: {n} windows, {time.time() - t0:.0f} s, {prov[0]}"


@app.function(image=image, volumes={VOL: volume}, timeout=3600, memory=65536, cpu=4)
def merge(out_dir: str = "data/b1/feats", out_file: str = "data/b1/feats.npz") -> str:
    import glob
    import numpy as np
    files = sorted(glob.glob(f"{VOL}/{out_dir}/*.npz"))
    parts = [dict(np.load(f)) for f in files]
    keys = parts[0].keys()
    arr = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.argsort(arr["item_id"])
    arr = {k: v[order] for k, v in arr.items()}
    np.savez(f"{VOL}/{out_file}", **arr)
    volume.commit()
    ids = arr["item_id"]
    return f"merged {len(files)} files: {len(ids)} items, unique {len(np.unique(ids))}, max id {ids.max()}"


@app.local_entrypoint()
def main(what: str = "all", max_containers: int = 8, ami_group: int = 6, items: str = "data/b1/items.parquet",
         out_dir: str = "data/b1/feats", out_file: str = "data/b1/feats.npz"):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    tbl = pq.read_table(items)
    out = []
    if what in ("all", "std") and "shard" in tbl.column_names:
        std = tbl.filter(pc.equal(tbl["source"], "std"))
        shards = sorted(set(std["shard"].to_pylist()))
        if shards:
            out += list(featurize_std.map(shards, kwargs=dict(items_path=items, out_dir=out_dir)))
    if what in ("all", "ami"):
        ami = tbl.filter(pc.equal(tbl["source"], "ami"))
        paths = sorted(set(ami["audio"].to_pylist()))
        groups = [paths[i:i + ami_group] for i in range(0, len(paths), ami_group)]
        if groups:
            out += list(featurize_ami.map(groups, kwargs=dict(items_path=items, out_dir=out_dir)))
    if what in ("all", "std", "ami", "merge"):
        out.append(merge.remote(out_dir, out_file))
    print("\n".join(out))
