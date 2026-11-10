"""Parakeet Realtime EOU encoder features on Modal, for the windows of data/b1/items.parquet.

    modal run modal_parakeet_features.py --items /path/to/local/items.parquet

For now this only measures: it processes one smart-turn-data shard and one group of AMI files and
prints time and size per window, so the cost of the full run is known before it is launched.

Each 8 s window (``eot.windows`` semantics, room tone after ``cut_s``) is trimmed by a seeded random
0 to 160 ms at its end, because in a live stream the newest encoder chunk is up to one chunk old.
A single forward pass over the 777 mel frames that remain then gives the same output as the
chunk-by-chunk streaming encoder (checked to within 7e-5, on values around 0.15). The frame count is
9 + 16 * 48, so the last output frame falls on a streaming chunk boundary.

Per window: ``item_id``, ``last`` (fp16, 13x512, the final 1.04 s of encoder frames), ``pooled``
(fp16, 512, the mean of all 98 frames) and ``jitter`` (int16, samples trimmed from the end).
"""

from __future__ import annotations

import modal

app = modal.App("parakeet-eou-features")
volume = modal.Volume.from_name("smart-turn-v2")
VOL = "/vol"
MODEL = "/model/parakeet_realtime_eou_120m-v1.nemo"
OUT_DIR = "data/parakeet/feats"
BATCH = 128
WINDOW_SAMPLES = 128000
CHUNK_SAMPLES = 2560        # 160 ms
N_FRAMES = 777              # 9 + 16 * 48
SEGMENT_SAMPLES = (N_FRAMES - 1) * 160 + 200    # a frame reads 200 samples past its centre
LAST_FRAMES = 13

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .uv_pip_install("nemo-toolkit[asr]==3.0.0", "torch==2.15.0", "torchaudio==2.11.1", "soundfile", "pyarrow", "pandas",
                    "huggingface_hub")
    .run_commands("python -c \"from huggingface_hub import hf_hub_download as d; "
                  "d('nvidia/parakeet_realtime_eou_120m-v1', 'parakeet_realtime_eou_120m-v1.nemo', local_dir='/model')\"")
    .add_local_python_source("eot")
)


class Extractor:
    """Collects windows into batches, encodes them and keeps the features and stage timings."""

    def __init__(self):
        import time
        import torch
        import nemo.collections.asr as nemo_asr
        t = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nemo_asr.models.ASRModel.restore_from(MODEL, map_location=self.device).eval()
        self.times = {"load_model": time.time() - t, "read_and_cut": 0.0, "encode": 0.0, "save": 0.0}
        self.parts, self.segments, self.jitters, self.ids = [], [], [], []

    def add(self, window, item_id: int) -> None:
        import numpy as np
        jitter = int(np.random.default_rng(item_id).integers(0, CHUNK_SAMPLES))
        end = WINDOW_SAMPLES - jitter
        self.segments.append(window[end - SEGMENT_SAMPLES: end])
        self.jitters.append(jitter)
        self.ids.append(item_id)
        if len(self.ids) >= BATCH:
            self.flush()

    def flush(self) -> None:
        import time
        import numpy as np
        import torch
        if not self.ids:
            return
        t = time.time()
        with torch.inference_mode():
            signal = torch.from_numpy(np.stack(self.segments)).to(self.device)
            lengths = torch.full((len(self.ids),), SEGMENT_SAMPLES, device=self.device)
            mel, _ = self.model.preprocessor(input_signal=signal, length=lengths)
            frames = torch.full((len(self.ids),), N_FRAMES, device=self.device)
            encoded, _ = self.model.encoder(audio_signal=mel[:, :, :N_FRAMES], length=frames)   # (B, 512, 98)
            last = encoded[:, :, -LAST_FRAMES:].transpose(1, 2).half().cpu().numpy()
            pooled = encoded.mean(-1).half().cpu().numpy()
        self.parts.append({"item_id": np.array(self.ids, np.int64), "last": last, "pooled": pooled,
                           "jitter": np.array(self.jitters, np.int16)})
        self.segments, self.jitters, self.ids = [], [], []
        self.times["encode"] += time.time() - t

    def finish(self, name: str, t0: float, out_dir: str = f"{VOL}/{OUT_DIR}", vol=None) -> dict:
        import os
        import time
        import numpy as np
        self.flush()
        t = time.time()
        arrays = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{name}.npz"
        np.savez(path, **arrays)
        (vol or volume).commit()
        self.times["save"] = time.time() - t
        return {"name": name, "windows": len(arrays["item_id"]), "total_s": round(time.time() - t0, 1),
                "stages_s": {k: round(v, 1) for k, v in self.times.items()}, "bytes": os.path.getsize(path),
                "device": self.device, "finite": bool(np.isfinite(arrays["last"]).all()),
                "shapes": {k: list(v.shape) for k, v in arrays.items()},
                "mean_abs": round(float(np.abs(arrays["last"].astype(np.float32)).mean()), 4)}


def _cut(x, end_s, t_off, cut_s, rng):
    """Window ending at end_s; audio after cut_s (if any) is replaced by room tone from [t_off, cut_s]."""
    import math
    from eot.windows import cut_window
    if cut_s is not None and not (isinstance(cut_s, float) and math.isnan(cut_s)):
        x = x[: int(round(cut_s * 16000))]
    return cut_window(x, 16000, end_s, t_off, rng)


def _to_mono_16k(x, sr):
    import torch
    import torchaudio
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, volumes={VOL: volume}, timeout=3 * 3600, max_containers=8)
def featurize_std(shard: str, items_path: str = "data/b1/items.parquet") -> dict:
    import io
    import time
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import soundfile as sf
    t0 = time.time()
    extractor = Extractor()
    items = pd.read_parquet(f"{VOL}/{items_path}")
    items = items[(items.source == "std") & (items.shard == shard)].sort_values("row")
    parquet = pq.ParquetFile(f"{VOL}/data/smart_turn_data/{items.split.iloc[0]}/data/{shard}")
    by_row = {r: g for r, g in items.groupby("row")}
    rng = np.random.default_rng(0)
    row = 0
    for group in range(parquet.num_row_groups):
        n = parquet.metadata.row_group(group).num_rows
        need = [r for r in range(row, row + n) if r in by_row]
        if need:
            column = parquet.read_row_group(group, columns=["audio"]).column("audio")
            for r in need:
                t = time.time()
                x, sr = sf.read(io.BytesIO(column[r - row].as_py()["bytes"]), dtype="float32", always_2d=False)
                x = _to_mono_16k(x, sr)
                windows = [(_cut(x, float(i.end_s), float(i.t_off), i.cut_s, rng), int(i.item_id))
                           for i in by_row[r].itertuples(index=False)]
                extractor.times["read_and_cut"] += time.time() - t
                for window, item_id in windows:
                    extractor.add(window, item_id)
        row += n
    return extractor.finish(f"std_{shard}", t0)


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, volumes={VOL: volume}, timeout=3 * 3600, max_containers=8)
def featurize_ami(audio_paths: list[str], items_path: str = "data/b1/items.parquet") -> dict:
    import time
    import numpy as np
    import pandas as pd
    import soundfile as sf
    t0 = time.time()
    extractor = Extractor()
    items = pd.read_parquet(f"{VOL}/{items_path}")
    items = items[(items.source == "ami") & items.audio.isin(audio_paths)]
    rng = np.random.default_rng(0)
    for path, group in items.groupby("audio"):
        t = time.time()
        x, sr = sf.read(f"{VOL}/{path}", dtype="float32", always_2d=False)
        x = _to_mono_16k(x, sr)
        extractor.times["read_and_cut"] += time.time() - t
        for i in group.sort_values("end_s").itertuples(index=False):
            t = time.time()
            window = _cut(x, float(i.end_s), float(i.t_off), i.cut_s, rng)
            extractor.times["read_and_cut"] += time.time() - t
            extractor.add(window, int(i.item_id))
    tag = audio_paths[0].split("/")[-1].split(".")[0]
    return extractor.finish(f"ami_{tag}_{len(audio_paths)}", t0)


CANDOR = "/candor"
candor_volume = modal.Volume.from_name("candor")
CANDOR_OUT = f"{CANDOR}/eot_test/parakeet"


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, volumes={CANDOR: candor_volume}, timeout=3 * 3600,
              max_containers=8)
def featurize_candor(convs: list[str], items_path: str = "eot_test/items.parquet") -> dict:
    """CANDOR test windows (scripts/candor_eot_items.py) of these conversations. Everything stays on the candor volume."""
    import time
    import numpy as np
    import pandas as pd
    import soundfile as sf
    t0 = time.time()
    extractor = Extractor()
    items = pd.read_parquet(f"{CANDOR}/{items_path}")
    items = items[items.conv.isin(convs)]
    rng = np.random.default_rng(0)
    for path, group in items.groupby("audio"):
        t = time.time()
        x, sr = sf.read(f"{CANDOR}/{path}", dtype="float32", always_2d=False)
        x = _to_mono_16k(x, sr)
        extractor.times["read_and_cut"] += time.time() - t
        for i in group.sort_values("end_s").itertuples(index=False):
            t = time.time()
            window = _cut(x, float(i.end_s), float(i.t_off), i.cut_s, rng)
            extractor.times["read_and_cut"] += time.time() - t
            extractor.add(window, int(i.item_id))
    return extractor.finish(f"part_{convs[0][:8]}_{len(convs)}", t0, CANDOR_OUT, candor_volume)


@app.function(image=image, volumes={CANDOR: candor_volume}, timeout=1800, memory=8192)
def merge_candor(n_items: int, out_file: str = f"{CANDOR}/eot_test/parakeet.npz") -> dict:
    import glob
    import numpy as np
    candor_volume.reload()
    parts = [dict(np.load(f)) for f in sorted(glob.glob(f"{CANDOR_OUT}/*.npz"))]
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(arrays["item_id"])
    arrays = {k: v[order] for k, v in arrays.items()}
    ok = bool((arrays["item_id"] == np.arange(n_items)).all()) if len(order) == n_items else False
    if ok:
        np.savez(out_file, **arrays)
        candor_volume.commit()
    return {"files": len(parts), "rows": len(order), "complete": ok, "written": out_file if ok else False}


@app.function(image=image, volumes={VOL: volume}, timeout=3600, memory=32768, cpu=4.0)
def merge(n_items: int, out_file: str = "data/parakeet/feats.npz") -> dict:
    """Place every task file's rows at index item_id, so the result is ordered and memory stays ~1x."""
    import glob
    import time
    import numpy as np
    t0 = time.time()
    volume.reload()
    files = sorted(glob.glob(f"{VOL}/{OUT_DIR}/*.npz"))
    out = {"item_id": np.full(n_items, -1, np.int64), "last": np.zeros((n_items, LAST_FRAMES, 512), np.float16),
           "pooled": np.zeros((n_items, 512), np.float16), "jitter": np.zeros(n_items, np.int16)}
    rows = 0
    for f in files:
        part = np.load(f)
        ids = part["item_id"]
        for k in out:
            out[k][ids] = part[k]
        rows += len(ids)
    missing = int((out["item_id"] < 0).sum())
    if missing or rows != n_items:
        return {"files": len(files), "rows": rows, "missing": missing, "written": False}
    np.savez(f"{VOL}/{out_file}", **out)
    volume.commit()
    return {"files": len(files), "rows": rows, "missing": 0, "written": out_file, "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(items: str, what: str = "measure", shard: str = "train-00018-of-00083.parquet", ami_group: int = 6):
    """what = measure (one shard and the median-sized AMI group), all (every task, then merge), merge, or candor
    (``items`` is then the local CANDOR test parquet, already uploaded to the candor volume as eot_test/items.parquet)."""
    import json
    import pandas as pd
    if what == "candor":
        table = pd.read_parquet(items, columns=["item_id", "conv"])
        convs = sorted(table.conv.unique())
        groups = [convs[i::8] for i in range(8)]
        results = list(featurize_candor.map(groups, return_exceptions=True))
        print(json.dumps([r if isinstance(r, dict) else str(r) for r in results], indent=1), flush=True)
        if all(isinstance(r, dict) for r in results):
            print(json.dumps(merge_candor.remote(len(table)), indent=1), flush=True)
        return
    table = pd.read_parquet(items, columns=["item_id", "source", "shard", "audio"])
    assert (table.item_id.to_numpy() == range(len(table))).all()
    counts = table[table.source == "ami"].groupby("audio").size().sort_index()
    paths = counts.index.tolist()
    groups = [paths[i:i + ami_group] for i in range(0, len(paths), ami_group)]
    if what == "measure":
        group = sorted(groups, key=lambda g: int(counts[g].sum()))[len(groups) // 2]
        std_call, ami_call = featurize_std.spawn(shard), featurize_ami.spawn(group)
        print(json.dumps({"ami_groups": len(groups), "std": std_call.get(), "ami": ami_call.get()}, indent=1), flush=True)
        return
    if what == "all":
        shards = sorted(table[table.source == "std"].shard.unique())
        results = list(featurize_std.map(shards, return_exceptions=True))
        results += list(featurize_ami.map(groups, return_exceptions=True))
        failed = [str(r) for r in results if not isinstance(r, dict)]
        done = [r for r in results if isinstance(r, dict)]
        print(json.dumps({"tasks": len(results), "failed": failed[:10], "windows": sum(r["windows"] for r in done),
                          "task_seconds": round(sum(r["total_s"] for r in done)),
                          "encode_seconds": round(sum(r["stages_s"]["encode"] for r in done)),
                          "bytes": sum(r["bytes"] for r in done),
                          "non_finite": [r["name"] for r in done if not r["finite"]]}, indent=1), flush=True)
        if failed:
            return
    print(json.dumps(merge.remote(len(table)), indent=1), flush=True)
