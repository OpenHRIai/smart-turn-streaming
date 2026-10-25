"""Stage A1: acoustic offsets and clip statistics for every row of a smart-turn-data parquet set.

    .venv/bin/python scripts/a1_offsets.py --parquet-dir data/smart_turn_data/test \
        --out data/a1/test_offsets.parquet [--limit N] [--workers 8]

Reads the parquet shards directly (no `datasets` cache), decodes each clip, resamples to 16 kHz if
needed, runs ``eot.offsets.OffsetDetector`` and writes one row per clip with the metadata columns
of the source and the ``ClipStats`` fields. Idempotent per shard: shards already present in the
output are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.offsets import SR, OffsetDetector  # noqa: E402

META = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset"]
_DET: OffsetDetector | None = None


def _init():
    global _DET
    import torch
    torch.set_num_threads(1)
    _DET = OffsetDetector()


def decode(audio: dict) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
        sr = SR
    return x.astype(np.float32), sr


def work(args: tuple) -> dict:
    shard, row, meta, audio = args
    try:
        x, sr = decode(audio)
        st = _DET.stats(x, sr).as_row()
        err = None
    except Exception as e:  # keep going; the row is flagged
        st, err = {}, f"{type(e).__name__}: {e}"
    return dict(shard=shard, row=row, **meta, **st, error=err)


def rows_of(path: Path, limit: int | None):
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=128, columns=META + ["audio"]):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            meta = {k: d[k][i] for k in META}
            yield (path.name, n, meta, d["audio"][i])
            n += 1
            if limit is not None and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="rows per shard")
    ap.add_argument("--shards", type=int, help="number of shards to process")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    shards = sorted(Path(a.parquet_dir).rglob("*.parquet"))
    if a.shards:
        shards = shards[: a.shards]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    tables = []
    if out.exists() and not a.force:
        prev = pq.read_table(out)
        tables.append(prev)
        done = set(prev.column("shard").to_pylist())
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for shard in shards:
            if shard.name in done:
                print(f"skip {shard.name} (done)")
                continue
            ts = time.time()
            rows = list(pool.imap(work, rows_of(shard, a.limit), chunksize=8))
            tables.append(pa.Table.from_pylist(rows))
            n_err = sum(r["error"] is not None for r in rows)
            n_nos = sum(r.get("t_off") is None and r["error"] is None for r in rows)
            print(f"{shard.name}: {len(rows)} rows, {n_err} errors, {n_nos} without speech, "
                  f"{time.time() - ts:.0f} s", flush=True)
            tmp = out.with_suffix(".tmp.parquet")
            pq.write_table(pa.concat_tables(tables, promote_options="default"), tmp)
            tmp.replace(out)   # atomic: a kill mid-write cannot corrupt the output
    print(f"wrote {out} ({sum(t.num_rows for t in tables)} rows) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
