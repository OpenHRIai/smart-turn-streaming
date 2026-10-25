"""Stage A2: download AMI headset audio for a set of meetings (resumable).

    .venv/bin/python scripts/a2_ami_fetch.py --split dev test [--meetings ES2002a ...] [--workers 4]

Files go to data/ami/audio/<meeting>/<meeting>.Headset-<n>.wav (the mirror's 16 kHz PCM16 mono,
~40 MB per 35 min). A file is skipped when its size matches the server's Content-Length.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import ami  # noqa: E402


def fetch(url: str, dest: Path, retries: int = 4) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get("Content-Length", "0"))
            if dest.exists() and dest.stat().st_size == size and size > 0:
                return f"skip {dest.name} ({size / 1e6:.0f} MB)"
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
            return f"got  {dest.name} ({size / 1e6:.0f} MB)"
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"FAIL {dest.name}: {e}"
            time.sleep(5 * (attempt + 1))
    return f"FAIL {dest.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="*", default=[], choices=["dev", "test", "train"])
    ap.add_argument("--meetings", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ms = ami.meetings()
    want = [m for m in ms if ami.split_of(m) in a.split] + a.meetings
    jobs = [(ami.headset_url(m, s.channel), ami.headset_path(m, s.channel)) for m in want for s in ms[m]]
    print(f"{len(want)} meetings, {len(jobs)} headset files", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda j: fetch(*j), jobs):
            print(msg, flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
