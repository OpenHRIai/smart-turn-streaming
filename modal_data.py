"""Fetch the raw training data straight into the Modal volume `smart-turn-v2` (no laptop upload).

    modal run --detach modal_data.py                       # smart-turn-data (train+test) + all AMI headsets
    modal run --detach modal_data.py --what smart-turn     # only pipecat's parquet shards
    modal run --detach modal_data.py --what ami --splits dev,test
    modal volume ls smart-turn-v2 data

Volume layout mirrors the repo's `data/` so the local scripts run unchanged with the volume
mounted at the repo root:

    /vol/data/smart_turn_data/{train,test}/data/*.parquet   (HF snapshot, CC BY 4.0)
    /vol/data/ami/annotations/...                           (ami_public_manual_1.6.2.zip unpacked, CC BY 4.0)
    /vol/data/ami/audio/<meeting>/<meeting>.Headset-<n>.wav
    /vol/data/a1, /vol/data/a2                              (derived artifacts, `modal volume put` from local)
"""

from __future__ import annotations

import modal

app = modal.App("smart-turn-v2-data")
volume = modal.Volume.from_name("smart-turn-v2", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip", "curl")
    .pip_install("huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("eot")
)

AMI_ANN_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


@app.function(image=image, volumes={VOL: volume}, timeout=4 * 3600, cpu=4)
def fetch_smart_turn(split: str) -> str:
    from huggingface_hub import snapshot_download
    dest = f"{VOL}/data/smart_turn_data/{split}"
    snapshot_download(f"pipecat-ai/smart-turn-data-v3.2-{split}", repo_type="dataset", local_dir=dest, max_workers=16)
    volume.commit()
    import os
    n = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dest) for f in fs)
    return f"{split}: {n / 1e9:.1f} GB in {dest}"


@app.function(image=image, volumes={VOL: volume}, timeout=3600)
def fetch_ami_annotations() -> str:
    import subprocess
    from pathlib import Path
    d = Path(f"{VOL}/data/ami")
    d.mkdir(parents=True, exist_ok=True)
    z = d / "ami_public_manual_1.6.2.zip"
    if not z.exists():
        subprocess.run(["curl", "-sL", "-o", str(z), AMI_ANN_URL], check=True)
    ann = d / "annotations"
    ann.mkdir(exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(ann)], check=True)
    volume.commit()
    return f"annotations: {(ann / 'LICENCE.txt').read_text()[:80]!r}"


@app.function(image=image, volumes={VOL: volume}, timeout=4 * 3600, cpu=2)
def fetch_ami_meetings(meetings: list[str]) -> str:
    """Headset wavs for a chunk of meetings (one container per chunk, 4 connections each)."""
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    sys.path.insert(0, "/root")
    from eot import ami
    from scripts_a2_fetch_shim import fetch  # noqa: F401  (defined below at import time)

    ann = Path(f"{VOL}/data/ami/annotations")
    ms = ami.meetings(ann)
    jobs = [(ami.headset_url(m, s.channel), Path(f"{VOL}/data/ami/audio") / m / f"{m}.Headset-{s.channel}.wav")
            for m in meetings for s in ms[m]]
    with ThreadPoolExecutor(4) as ex:
        msgs = list(ex.map(lambda j: fetch(*j), jobs))
    volume.commit()
    fails = [m for m in msgs if m.startswith("FAIL")]
    return f"{meetings[0]}..{meetings[-1]}: {len(jobs)} files, {len(fails)} failed" + ("\n" + "\n".join(fails) if fails else "")


# The download helper is identical to scripts/a2_ami_fetch.py; inlined as a module so the
# container does not need the scripts/ directory.
import sys as _sys
import types as _types

_shim = _types.ModuleType("scripts_a2_fetch_shim")
_shim.__dict__["__doc__"] = "fetch(url, dest) with retries and size check"
exec(
    '''
import time, urllib.request
from pathlib import Path

def fetch(url, dest, retries=4):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                size = int(r.headers.get("Content-Length", "0"))
            if dest.exists() and dest.stat().st_size == size and size > 0:
                return f"skip {dest.name}"
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if size and tmp.stat().st_size != size:
                raise OSError(f"size mismatch {tmp.stat().st_size} != {size}")
            tmp.replace(dest)
            return f"got {dest.name}"
        except Exception as e:
            if attempt == retries - 1:
                return f"FAIL {dest.name}: {e}"
            time.sleep(5 * (attempt + 1))
    return f"FAIL {dest.name}"
''',
    _shim.__dict__,
)
_sys.modules["scripts_a2_fetch_shim"] = _shim


@app.local_entrypoint()
def main(what: str = "all", splits: str = "dev,test,train", chunks: int = 12):
    from eot import ami
    out = []
    if what in ("all", "smart-turn"):
        out += list(fetch_smart_turn.map(["test", "train"]))
    if what in ("all", "ami"):
        out.append(fetch_ami_annotations.remote())
        want = [m for m in sorted(ami.meetings()) if ami.split_of(m) in splits.split(",")]
        size = -(-len(want) // chunks)
        parts = [want[i:i + size] for i in range(0, len(want), size)]
        out += list(fetch_ami_meetings.map(parts))
    print("\n".join(out))
