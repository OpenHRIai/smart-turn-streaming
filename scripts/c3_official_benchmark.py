"""Stage C3: pipecat's official Smart Turn benchmark on the local smart-turn-data v3.2 test set.

    .venv/bin/python scripts/c3_official_benchmark.py \
        --model data/models/smart-turn-v3.2-gpu.onnx:smart-turn-v3.2 \
        --model data/models/smart-turn-streaming-clipend.onnx:streaming-heads-calibrated \
        --model data/models/smart-turn-streaming-clipend-raw.onnx:streaming-heads-raw \
        --out-dir runs/c3/official

Uses the vendored, unmodified ``scripts/vendor/pipecat_benchmark.py`` (pipecat-ai/smart-turn
``benchmark.py`` at 4786657e) for everything it defines: the performance runs, the metric functions
(threshold 0.5, per language and per source dataset) and the markdown report. What this file adds is
the data path upstream keeps in ``train.py``: the parquet shards are read locally, each clip is cut to
its last 8 s or zero-padded at the start (``truncate_audio_to_last_n_seconds``), and features come from
``WhisperFeatureExtractor(chunk_length=8)`` with ``do_normalize=True``, exactly as the trainer does.
The features are extracted once and every model is scored on the same batches, so the three runs are
strictly comparable; each model still gets its own official-format report, plus ``comparison.md``.
"""

from __future__ import annotations

import argparse
import glob
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/vendor"))
import pipecat_benchmark as B  # noqa: E402
from audio_utils import truncate_audio_to_last_n_seconds  # noqa: E402


def _extract_features_cpu(fe, audio):
    """Upstream's perf helper hardcodes ``device="cuda"``; the same call on the CPU (the only change)."""
    out = fe(audio, sampling_rate=B.SAMPLING_RATE, return_tensors="np", padding="max_length",
             max_length=B.AUDIO_SECONDS * B.SAMPLING_RATE, truncation=True, do_normalize=True)["input_features"].astype(np.float32)
    return out.reshape(B.FEATURE_SHAPE) if out.shape != B.FEATURE_SHAPE else out


B._extract_features_np = _extract_features_cpu


class ParquetClips(torch.utils.data.Dataset):
    """smart-turn-data parquet shards -> the items upstream's collator expects."""

    def __init__(self, files: list[str], feature_extractor):
        import pyarrow.parquet as pq
        self.files = files
        self.counts = [pq.read_metadata(f).num_rows for f in files]
        self.offsets = np.cumsum([0] + self.counts)
        self.fe = feature_extractor
        self._cur_i, self._cur = -1, None

    def __len__(self):
        return int(self.offsets[-1])

    def _row(self, idx: int) -> dict:
        import pyarrow.parquet as pq
        i = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        if i != self._cur_i:
            self._cur = pq.read_table(self.files[i], columns=["audio", "language", "endpoint_bool", "dataset"])
            self._cur_i = i
        return self._cur.slice(idx - int(self.offsets[i]), 1).to_pylist()[0]

    @staticmethod
    def _decode(audio) -> np.ndarray:
        import soundfile as sf
        if isinstance(audio, dict) and audio.get("array") is not None:
            x, sr = np.asarray(audio["array"], np.float32), int(audio.get("sampling_rate", 16000))
        else:
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        return x.astype(np.float32)

    def __getitem__(self, idx):
        r = self._row(idx)
        x = truncate_audio_to_last_n_seconds(self._decode(r["audio"]), n_seconds=8)
        feats = self.fe(x, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=8 * 16000,
                        truncation=True, do_normalize=True).input_features.squeeze(0)
        return {"input_features": feats, "labels": torch.tensor(int(bool(r["endpoint_bool"]))),
                "language": str(r["language"]), "dataset": str(r["dataset"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="path[:label], repeatable")
    ap.add_argument("--test-dir", default="data/smart_turn_data/test/data")
    ap.add_argument("--out-dir", default="runs/c3/official")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--perf-runs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    models = [(m.split(":")[0], m.split(":")[1] if ":" in m else Path(m).stem) for m in a.model]

    fe = B.WhisperFeatureExtractor(chunk_length=B.AUDIO_SECONDS)
    files = sorted(glob.glob(str(ROOT / a.test_dir / "*.parquet")))
    ds = ParquetClips(files, fe)
    if a.limit:
        ds = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    B.log_progress(f"{len(ds):,} clips from {len(files)} shards; models: {[l for _, l in models]}")

    gpu_name = B.get_gpu_model_name()
    results = {}
    sessions = {}
    for path, label in models:
        r = {"onnx_path": path}
        sess = B.build_session(path, providers=["CPUExecutionProvider"])
        r["perf_cpu"] = B.run_perf(sess, runs=a.perf_runs)
        r["perf_gpu"] = {"note": "CUDAExecutionProvider not available; skipped."}
        zero = B._zero_audio(B.AUDIO_SECONDS, B.SAMPLING_RATE)
        r["perf_feature_extractor"] = B.run_fe_perf(fe, zero, runs=a.perf_runs)
        r["perf_e2e_cpu"] = B.run_e2e_perf(sess, fe, zero, runs=a.perf_runs)
        r["perf_e2e_gpu"] = {"note": "CUDAExecutionProvider not available; skipped."}
        results[label] = r
        sessions[label] = (sess, sess.get_inputs()[0].name)

    # one pass over the data, every model on the same batches (upstream's run_accuracy loop, shared)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=B.WhisperDataCollator(),
                                         num_workers=a.workers)
    probs = {l: [] for _, l in models}
    labels, langs, dsets = [], [], []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        x = batch["input_features"].numpy()
        for _, l in models:
            sess, inp = sessions[l]
            p, _ = B.process_predictions(sess.run(None, {inp: x})[0])
            probs[l].append(p)
        labels.append(batch["labels"].numpy()); langs.extend(batch["languages"]); dsets.extend(batch["datasets"])
        if (bi + 1) % 50 == 0:
            B.log_progress(f"{(bi + 1) * a.batch_size:,}/{len(ds):,} clips, {time.time() - t0:.0f} s")
    y = np.concatenate(labels).astype(np.int32)
    comp = ["| model | n | accuracy | precision | recall | F1 | FPR | FNR |", "|---|---|---|---|---|---|---|---|"]
    per_lang = {}
    for _, l in models:
        p = np.concatenate(probs[l]).astype(np.float32)
        acc = {"overall": B.compute_metrics_with_confusion(p, y),
               "per_language": B.compute_per_category_metrics(p, y, langs, "language"),
               "per_dataset": B.compute_per_category_metrics(p, y, dsets, "dataset"),
               "total_samples": int(len(y)), "unique_languages": sorted(set(langs)), "unique_datasets": sorted(set(dsets))}
        results[l]["accuracy"] = acc
        (out / f"{l}.md").write_text(B.format_markdown_report(results[l], gpu_name))
        np.save(out / f"{l}.probs.npy", p)
        o = acc["overall"]
        comp.append(f"| {l} | {o['sample_count']} | {o['accuracy']:.2f} % | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | "
                    f"{o['false_positive_rate']:.2f} % | {o['false_negative_rate']:.2f} % |")
        per_lang[l] = acc["per_language"]
    np.save(out / "labels.npy", y)
    (out / "languages.txt").write_text("\n".join(langs))
    # per-language accuracy side by side
    langs_sorted = sorted(set(langs), key=lambda k: -langs.count(k))
    head = "| language | n | " + " | ".join(f"{l}: acc / FPR" for _, l in models) + " |"
    comp += ["", head, "|---|---|" + "---|" * len(models)]
    for k in langs_sorted:
        cells = []
        for _, l in models:
            m = per_lang[l].get(k) or per_lang[l].get(B.format_language_name(k)) or {}
            cells.append(f"{m.get('accuracy', float('nan')):.1f} % / {m.get('false_positive_rate', float('nan')):.1f} %")
        comp.append(f"| {B.format_language_name(k)} | {langs.count(k)} | " + " | ".join(cells) + " |")
    text = "\n".join(comp)
    (out / "comparison.md").write_text(text + "\n")
    print(text)
    B.log_progress(f"reports in {out}")


if __name__ == "__main__":
    main()
