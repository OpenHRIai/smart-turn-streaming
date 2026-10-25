"""Stage B2: does the streaming detector reproduce the offline (B1) scores, and what does it cost?

    .venv/bin/python scripts/b2_validate.py --run runs/b1/knob_ami07_auc --n 150 --out runs/b2/validate.md

For ``--n`` AMI test silences (balanced), replay the headset audio from 9 s before the VAD stop to
2.5 s after it through ``StreamingEOT`` in 20 ms frames, with the VAD verdict taken from the gated
regions of the build (``vad_<L>.json``). At tau = 0.2 / 0.5 / 1.2 s read the streaming p_end and
compare it with the offline prediction of the same run for the same item (``predictions.npz``).
Configurations (``--configs`` picks a subset by name prefix): ``timer`` (encoder every 320 ms),
``silence`` (the default silence-clock schedule), ``aligned`` (refresh at every query, no
staleness: isolates the incremental-mel error), ``threads`` (4 ORT threads), and the Stage C2
``worker`` configurations, where the encoder runs on a background thread and the replay is paced at
real time (``--pace``, implied for them) so the thread hand-off is what a deployment sees. Reports
|delta p| statistics, the AUC of streaming vs offline scores on those silences, the per-call costs,
the decision-path burst measured on the calling thread, and for the worker the share of
silence-clock decisions taken before their refresh arrived.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.streaming import SR, StreamingEOT  # noqa: E402
from eot.windows import room_tone  # noqa: E402

OFFSETS = (0.2, 0.5, 1.2)
PRE_S = 9.0
POST_S = 2.5
FRAME = 320   # 20 ms


def vad_flags(meeting: str, speaker: str, t0: float, n: int) -> np.ndarray:
    regs = json.load(open(f"data/a2/ami/{meeting}/vad_{speaker}.json"))
    flags = np.zeros(n, bool)
    for a, b in regs:
        i0, i1 = int(max(a - t0, 0) * SR), int(max(b - t0, 0) * SR)
        if i1 > 0 and i0 < n:
            flags[i0: min(i1, n)] = True
    return flags


def replay(eot: StreamingEOT, x: np.ndarray, flags: np.ndarray, t_stop: float, queries, exact: bool, pace: bool = False) -> dict:
    """Run the stream; return the hop output at each query tau. ``exact``: force a hop (and, when the
    detector refreshes every hop, a refresh) on the frame that reaches t_stop + tau, so the query is
    aligned to the offline window within one 20 ms frame; otherwise take the detector's silence-clock
    hop for that tau (the hop the deployed rule decides on; with the grid off, the first scheduled hop).
    ``pace``: push each frame at its real-time instant (the worker thread needs wall time)."""
    want = {q: None for q in queries}
    t_wall0 = time.perf_counter()
    for i in range(0, len(x) - FRAME + 1, FRAME):
        if pace:
            due = t_wall0 + i / SR
            while True:
                dt = due - time.perf_counter()
                if dt <= 0:
                    break
                time.sleep(dt)
        o = eot.push(x[i:i + FRAME], bool(flags[i:i + FRAME].mean() > 0.5))
        now = eot.n_samples / SR
        for q in queries:
            if want[q] is None and now >= t_stop + q:
                if exact:
                    if eot.refresh_n <= eot.hop_n:
                        eot.refresh()
                    want[q] = eot.hop()
                elif o is not None and (not eot.silence_grid or (o.get("grid") and o["tau"] + 1e-6 >= q)):
                    want[q] = o
    return want


def cut_at_partner(x: np.ndarray, t_stop: float, b_onset: float | None, rng) -> np.ndarray:
    """What a deployed microphone hears: the user's own audio up to the stop, then room tone
    instead of the other speaker (the offline windows were built the same way)."""
    if b_onset is None or np.isnan(b_onset):
        return x
    cut = int((t_stop + b_onset) * SR)
    if cut >= len(x):
        return x
    return np.concatenate([x[:cut], room_tone(x[:cut], SR, t_stop, len(x) - cut, rng)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/b1/knob_ami07_auc")
    ap.add_argument("--encoder", default="data/models/smart-turn-v3.2-gpu.pooled.onnx")
    ap.add_argument("--heads", default="data/models/eot_heads_v32.onnx")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/b2/validate.md")
    ap.add_argument("--configs", nargs="*", default=None, help="name prefixes of the configurations to run (default: all)")
    ap.add_argument("--pace", action="store_true", help="real-time pace for every configuration (implied for worker ones)")
    ap.add_argument("--lead", type=float, default=0.0, help="worker: seconds before a silence-clock point its refresh is requested")
    ap.add_argument("--max-defer", type=float, default=0.08, help="worker: how long a silence-clock hop may wait for its refresh")
    a = ap.parse_args()

    items = pd.read_parquet("data/b1/items.parquet")
    z = np.load(f"{a.run}/predictions.npz")
    test = items.iloc[z["test_ami"]].assign(p_off=z["p_test_ami"][:, 0])
    test = test[(test.kind == "tau") & test.value.isin(OFFSETS)]
    ev = test.groupby(["meeting", "speaker", "t_off"]).agg(label=("p_end", "first"), audio=("audio", "first"), gap=("gap_s", "first"),
                                                            b_onset=("b_onset_s", "first")).reset_index()
    feats = np.load("data/b1/feats.npz")
    base_off = pd.Series(feats["p_v32"][z["test_ami"]], index=items.iloc[z["test_ami"]].index)
    test = test.assign(pb_off=base_off.reindex(test.index).values)
    rng = np.random.default_rng(a.seed)
    pick = pd.concat([ev[ev.label == 1].sample(min(a.n // 2, (ev.label == 1).sum()), random_state=a.seed),
                      ev[ev.label == 0].sample(min(a.n // 2, (ev.label == 0).sum()), random_state=a.seed)])
    idx = test.set_index(["meeting", "speaker", "t_off", "value"])
    p_off, pb_off = idx.p_off, idx.pb_off

    configs = {"timer only (refresh 320 ms), scheduled hops": dict(refresh_s=0.32, threads=1, exact=False, grid=False),
               "silence-clock refresh (default), scheduled hops": dict(refresh_s=0.32, threads=1, exact=False, grid=True),
               "aligned: refresh at the query (no staleness)": dict(refresh_s=0.1, threads=1, exact=True, grid=False),
               "threads: silence-clock refresh, 4 ORT threads": dict(refresh_s=0.32, threads=4, exact=False, grid=True),
               f"worker thread, 1 ORT thread, lead {a.lead * 1e3:.0f} ms, defer up to {a.max_defer * 1e3:.0f} ms, real-time pace": dict(refresh_s=0.32, threads=1, exact=False, grid=True, worker=True, pace=True),
               f"worker thread, 2 ORT threads, lead {a.lead * 1e3:.0f} ms, defer up to {a.max_defer * 1e3:.0f} ms, real-time pace": dict(refresh_s=0.32, threads=2, exact=False, grid=True, worker=True, pace=True)}
    if a.configs:
        configs = {k: v for k, v in configs.items() if any(k.startswith(c) for c in a.configs)}
    report = [f"# B2 validation: streaming vs offline ({a.run}), {len(pick)} AMI test silences\n"]
    cache = {}
    for name, cfg in configs.items():
        rows, timings = [], None
        eot = None
        t_wall = time.time()
        for r in pick.itertuples(index=False):
            if r.audio not in cache:
                cache[r.audio] = sf.read(r.audio, dtype="float32")[0]
            x_all = cache[r.audio]
            t0 = max(r.t_off - PRE_S, 0.0)
            seg = x_all[int(t0 * SR): int((r.t_off + POST_S) * SR)]
            if r.label == 1:
                seg = cut_at_partner(seg, r.t_off - t0, r.b_onset, rng)
            flags = vad_flags(r.meeting, r.speaker, t0, len(seg))
            eot = StreamingEOT(a.encoder, a.heads, refresh_s=cfg["refresh_s"], threads=cfg["threads"], silence_grid=cfg["grid"],
                               worker=cfg.get("worker", False), lead_s=a.lead, max_defer_s=a.max_defer)
            got = replay(eot, seg, flags, r.t_off - t0, OFFSETS, cfg["exact"], pace=cfg.get("pace", False) or a.pace)
            eot.close()
            for q, o in got.items():
                if o is None:
                    continue
                key = (r.meeting, r.speaker, r.t_off, q)
                if key not in p_off.index:
                    continue          # incompletes whose pause ended before q have no offline item
                rows.append(dict(label=int(r.label), q=q, p_stream=o["p_end"], p_off=float(p_off[key]), stale=o["stale_s"],
                                 pending=o.get("pending", 0), tau_stream=o["tau"], pb_stream=o["p_base"], pb_off=float(pb_off[key])))
            if timings is None:
                timings = eot.timings
            else:
                for f in ("hop_ms", "refresh_ms", "mel_ms", "push_ms", "decision_ms", "wait_ms"):
                    getattr(timings, f).extend(getattr(eot.timings, f))
                timings.n_stale_decisions += eot.timings.n_stale_decisions
        d = pd.DataFrame(rows)
        from sklearn.metrics import roc_auc_score
        report.append(f"\n## {name}\n")
        report.append("| tau | n | heads: mean / p90 / max |dp| | heads AUC offline / streaming | frozen v3.2: mean / max |dp| | frozen AUC offline / streaming | tau fed (mean) | vector age at the query: mean / max | queries before their refresh arrived |")
        report.append("|---|---|---|---|---|---|---|---|---|")
        for q in OFFSETS:
            g = d[d.q == q]
            dp = (g.p_stream - g.p_off).abs(); db = (g.pb_stream - g.pb_off).abs()
            auc = lambda col: roc_auc_score(g.label, g[col]) if g.label.nunique() == 2 else float("nan")
            report.append(f"| {q} | {len(g)} | {dp.mean():.3f} / {dp.quantile(.9):.3f} / {dp.max():.3f} | {auc('p_off'):.3f} / {auc('p_stream'):.3f} | "
                          f"{db.mean():.3f} / {db.max():.3f} | {auc('pb_off'):.3f} / {auc('pb_stream'):.3f} | {g.tau_stream.mean():.2f} | "
                          f"{g.stale.mean() * 1e3:.0f} / {g.stale.max() * 1e3:.0f} ms | {(g.pending > 0).mean():.1%} |")
        tm = timings.summary()
        hop, ref, mel = tm["hop"], tm["refresh"], tm["mel_per_push"]
        n_audio_s = len(pick) * (PRE_S + POST_S)
        core = (hop["p50"] * hop["n"] + ref["p50"] * ref["n"] + mel["p50"] * mel["n"]) / (n_audio_s * 1000) * 100
        report.append(f"\nCost per call (ms): hop p50 {hop['p50']:.2f} / p99 {hop['p99']:.2f} / max {hop['max']:.2f}; "
                      f"encoder refresh p50 {ref['p50']:.1f} / p99 {ref['p99']:.1f}; mel per 20 ms frame p50 {mel['p50']:.3f} / p99 {mel['p99']:.3f}. "
                      f"Share of one core at this cadence: **{core:.1f} %**. Replay wall time {time.time() - t_wall:.0f} s for {len(pick)} silences of {PRE_S + POST_S:.1f} s.")
        push, dec = tm["push"], tm["decision"]
        report.append(f"Calling-thread wall time per 20 ms frame (push): p50 {push['p50']:.2f} / p99 {push['p99']:.2f} / max {push['max']:.2f} ms. "
                      f"Decision-path burst (frames that hop at a silence-clock point, n = {dec['n']}): p50 {dec['p50']:.2f} / p99 {dec['p99']:.2f} / max {dec['max']:.2f} ms"
                      + (f"; silence-clock decisions taken before their refresh arrived: {tm['n_stale_decisions']}." if cfg.get("worker") else "."))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
