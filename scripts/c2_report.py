"""Stage C2: summarise the pipeline-task cost of the streaming analyzer from b3_replay CSVs.

    .venv/bin/python scripts/c2_report.py runs/c2/replay_worker_wait20.csv runs/c2/replay_worker_wait50.csv runs/c2/replay_inline.csv

Per CSV (one configuration): whole-process CPU share (getrusage over the replay, mean / p95), the
pipeline-task wall time of append_audio (p99 and max over all frames of a replay, then the median and
max of those across replays), the analyze_end_of_turn call at the VAD stop (max per replay -> median /
max), the age of the encoder vector at each rule offset and the share of decisions taken before their
refresh arrived, plus the turn-level outcome (false alarms, model commits, latency) so a change in the
compute path can be checked not to have changed the decisions.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np


def f(x):
    return float(x) if x not in ("", "None", None) else float("nan")


def summarise(path: str) -> str:
    rows = list(csv.DictReader(open(path)))
    cpu = np.array([f(r["cpu_share"]) for r in rows])
    p99 = np.array([f(r["append_ms_p99"]) for r in rows]); mx = np.array([f(r["append_ms_max"]) for r in rows])
    aeot = np.array([f(r["analyze_ms"]) for r in rows])
    dec = {}
    for r in rows:
        for q in json.loads(r["queries"]):
            if "pend" in q:
                dec.setdefault(q["off"], []).append((q["stale"], q["pend"], q["ms"]))
    pos = [r for r in rows if r["label"] == "1"]; neg = [r for r in rows if r["label"] == "0"]
    lat = np.array([f(r["latency_s"]) for r in pos]); lat_f = np.where(np.isnan(lat), 3.2, lat)
    fa = np.mean([not np.isnan(f(r["latency_s"])) and f(r["latency_s"]) < (f(r["gap_s"]) if not np.isnan(f(r["gap_s"])) else np.inf) for r in neg]) if neg else float("nan")
    by_model = np.mean([r["committed_at"] not in ("", "stop_secs") for r in pos]) if pos else float("nan")
    out = [f"### {path} ({rows[0]['rule']}, {len(pos)} turn ends / {len(neg)} pauses)", "",
           f"- whole-process CPU share of one core: mean {np.nanmean(cpu):.3f}, p95 {np.nanpercentile(cpu, 95):.3f}",
           f"- append_audio on the pipeline task, per-replay p99: median {np.nanmedian(p99):.2f} ms, worst replay {np.nanmax(p99):.2f} ms; "
           f"per-replay max: median {np.nanmedian(mx):.2f} ms, worst {np.nanmax(mx):.2f} ms",
           f"- analyze_end_of_turn at the VAD stop (awaited refresh included): median {np.nanmedian(aeot):.1f} ms, max {np.nanmax(aeot):.1f} ms",
           f"- outcome: strict false alarms {fa:.1%}, turn ends committed by the model {by_model:.2f}, commit latency p50 / p90 / mean "
           f"{np.percentile(lat_f, 50):.2f} / {np.percentile(lat_f, 90):.2f} / {lat_f.mean():.2f} s", ""]
    if dec:
        out += ["| rule offset | decisions | vector age at the decision: mean / p90 / max | decided before the refresh arrived | decision-path ms: p50 / max |", "|---|---|---|---|---|"]
        for off in sorted(dec):
            st = np.array([d[0] for d in dec[off]]); pend = np.array([d[1] for d in dec[off]]); ms = np.array([d[2] for d in dec[off]])
            out.append(f"| {off:g} s | {len(st)} | {st.mean() * 1e3:.0f} / {np.percentile(st, 90) * 1e3:.0f} / {st.max() * 1e3:.0f} ms | {(pend > 0).mean():.0%} | "
                       f"{np.percentile(ms, 50):.1f} / {ms.max():.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("\n\n".join(summarise(p) for p in sys.argv[1:]))
