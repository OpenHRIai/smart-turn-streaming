"""Stage B3: end-to-end trigger latency in a real Pipecat pipeline, streaming detector vs stock Smart Turn.

    .venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far10 --n-complete 80 --n-incomplete 80 \
        --out runs/b3/streaming_far10.csv --quiet
    .venv/bin/python scripts/b3_replay.py --analyzer stock --out runs/b3/stock.csv --quiet
    .venv/bin/python scripts/b3_replay.py --report runs/b3/*.csv

AMI test silences (``data/a2/ami/manifest_test.parquet``) are replayed from 9 s before the VAD
stop to 3.6 s after it as 20 ms frames at real-time pace through ``VADProcessor(SileroVADAnalyzer)``
-> ``UserTurnProcessor(VAD start, TurnAnalyzerUserTurnStopStrategy(analyzer))``, the same harness as
Phase 0 (``measure_pipecat_latency.py``, whose Trace / summary / report are reused). For complete
turns the audio after the next speaker's onset is replaced by the speaker's room tone (a deployed
microphone does not hear the agent). Headset level is normalised to -23 dBFS (pipecat's VAD gates
on loudness; AMI headsets sit around -37 dBFS).

Analyzers: ``stock`` = pipecat's ``LocalSmartTurnAnalyzerV3`` with the v3.2 fp32 model (one query at
the VAD stop, theta 0.5, then stop_secs); ``streaming`` = ``eot.pipecat_analyzer.StreamingEOTAnalyzer``
with a rule table from ``RULES``; ``phase0`` = the Phase 0 re-query analyzer on the multimodal
audio-only graph (for reference).

Stage C2: the streaming analyzer runs its encoder refresh on a worker thread by default (``--no-worker``
restores the B3 inline behaviour; ``--lead`` / ``--wait`` are the analyzer's anticipation and VAD-stop wait).
Each row records the pipeline-task wall time of ``append_audio`` (p99 / max over the replay), of the
``analyze_end_of_turn`` call at the VAD stop, and per decision the age of the encoder vector (``stale``)
and whether its refresh was still pending (``pend``) in the ``queries`` JSON.

Stage C1 (``--strategy eager --eager-rule far15_gen30``): the streaming analyzer with an eager table and
``eot.pipecat_eager_strategy.TurnAnalyzerEagerUserTurnStopStrategy`` instead of the stock stop strategy.
The trace then also records the speculative inference (eager), its withdrawal (cancel) and whether the
commit confirmed it; ``--report-two-stage`` scores perceived response time, generations spent on pauses
and audible false alarms at ``--response-latency`` seconds (inference trigger -> first bot audio).

    .venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far15 --strategy eager --eager-rule far15_gen30 \
        --n-complete 80 --n-incomplete 80 --out runs/c1/far15_gen30.csv --quiet
    .venv/bin/python scripts/b3_replay.py --report-two-stage runs/b3/stock.csv runs/b3/streaming_far15.csv runs/c1/far15_gen30.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure_pipecat_latency as H  # noqa: E402  (Trace, summarize_trace, report, normalize_level, build_pipeline_modules)
from eot.windows import room_tone  # noqa: E402

SR = 16000
FRAME_MS = 20


class Trace(H.Trace):
    """Phase 0 trace plus the speculation events of the eager strategy."""

    def __init__(self):
        super().__init__()
        self.eager: list[float] = []        # speculative inference triggered (time.time())
        self.cancels: list[float] = []      # speculation withdrawn
        self.confirms: list[tuple[float, bool]] = []   # (turn stop time, confirms_speculation)
        self.analyzer = None                            # the streaming analyzer (for its timing lists)


def make_stop_strategy(analyzer, a):
    if a.strategy == "eager":
        from eot.pipecat_eager_strategy import TurnAnalyzerEagerUserTurnStopStrategy
        return TurnAnalyzerEagerUserTurnStopStrategy(turn_analyzer=analyzer, wait_for_transcript=False,
                                                     speculation_timeout=a.speculation_timeout)
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer, wait_for_transcript=False)


def make_processors(mods, trace, audio, offset_sample, analyzer, vad_stop_secs: float, a=None):
    """As measure_pipecat_latency.make_processors, with the VAD stop_secs passed explicitly."""
    FrameProcessor = mods["FrameProcessor"]
    InputAudioRawFrame = mods["InputAudioRawFrame"]
    StartFrame = mods["StartFrame"]
    VADUserStoppedSpeakingFrame = mods["VADUserStoppedSpeakingFrame"]
    VADUserStartedSpeakingFrame = mods["VADUserStartedSpeakingFrame"]

    class WavSource(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            if isinstance(frame, StartFrame):
                self.create_task(self._stream())

        async def _stream(self):
            n = SR * FRAME_MS // 1000
            t0 = time.monotonic()
            for i in range(0, len(audio) - n + 1, n):
                due = t0 + (i / SR)
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if trace.t_offset is None and i + n > offset_sample:
                    trace.t_offset = time.time() - (i + n - offset_sample) / SR
                chunk = np.ascontiguousarray(audio[i: i + n]).tobytes()
                await self.push_frame(InputAudioRawFrame(audio=chunk, sample_rate=SR, num_channels=1))
            await asyncio.sleep(0.1)
            trace.done.set()

    class Tap(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                trace.vad_stops.append(frame.timestamp)
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                trace.vad_starts.append(frame.timestamp)
            await self.push_frame(frame, direction)

    vad = mods["VADProcessor"](vad_analyzer=mods["SileroVADAnalyzer"](params=mods["VADParams"](stop_secs=vad_stop_secs)))
    stop = make_stop_strategy(analyzer, a) if a is not None else \
        mods["TurnAnalyzerUserTurnStopStrategy"](turn_analyzer=analyzer, wait_for_transcript=False)
    strategies = mods["UserTurnStrategies"](start=[mods["VADUserTurnStartStrategy"]()], stop=[stop])
    turns = mods["UserTurnProcessor"](user_turn_strategies=strategies, user_turn_stop_timeout=30.0)

    # the speculation events, straight from the stop strategy (the controller re-emits them without the
    # speculation argument)
    @stop.event_handler("on_user_turn_inference_triggered")
    async def _inference(strategy, speculation):
        if speculation is not None:
            trace.eager.append(time.time())

    @stop.event_handler("on_user_turn_speculation_cancelled")
    async def _cancelled(strategy):
        trace.cancels.append(time.time())

    @stop.event_handler("on_user_turn_stopped")
    async def _stopped_params(strategy, params):
        trace.confirms.append((time.time(), bool(params.confirms_speculation)))

    @turns.event_handler("on_user_turn_started")
    async def _started(processor, strategy):
        trace.turn_starts.append(time.time())

    @turns.event_handler("on_user_turn_stopped")
    async def _stopped(processor, strategy):
        trace.turn_stops.append(time.time())

    return [WavSource(), vad, turns, Tap()]


def make_analyzer(kind: str, rule: str, trace, a):
    if kind == "streaming":
        from eot.pipecat_analyzer import StreamingEOTAnalyzer, StreamingEOTParams
        params = StreamingEOTParams.from_rule(rule, eager=(a.eager_rule or None), stop_secs=a.stop_secs, vad_stop_secs=a.vad_stop_secs)
        an = StreamingEOTAnalyzer(encoder_onnx=a.encoder, heads_onnx=a.heads, params=params, threads=a.threads,
                                  worker=not a.no_worker, lead_s=a.lead, wait_s=a.wait, max_defer_s=a.max_defer)
        an.decisions = trace.queries          # the analyzer appends its decision points to the trace
        trace.analyzer = an
        return an
    if kind == "stock":
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        class Tracing(LocalSmartTurnAnalyzerV3):
            def _predict_endpoint(self, audio_array):
                t0 = time.perf_counter()
                r = super()._predict_endpoint(audio_array)
                trace.queries.append({"t": time.time(), "offset_s": a.vad_stop_secs, "p": float(r["probability"]), "theta": 0.5,
                                      "decision": int(r["prediction"]), "ms": (time.perf_counter() - t0) * 1e3})
                return r

        an = Tracing(smart_turn_model_path=a.stock_model or None, cpu_count=a.threads, params=SmartTurnParams(stop_secs=a.stop_secs))
        an._predict_endpoint(np.zeros(8 * SR, np.float32)); trace.queries.clear()
        return an
    if kind == "phase0":
        return H.make_analyzer(a.phase0_model, rule, trace, a.stop_secs, a.vad_stop_secs)
    raise ValueError(kind)


async def replay(mods, audio, offset_sample, kind, rule, a):
    trace = Trace()
    analyzer = make_analyzer(kind, rule, trace, a)
    procs = make_processors(mods, trace, audio, offset_sample, analyzer, a.vad_stop_secs, a)
    pipeline = mods["Pipeline"](procs)
    worker = mods["PipelineWorker"](pipeline, params=mods["PipelineParams"](audio_in_sample_rate=SR),
                                    idle_timeout_secs=None, check_dangling_tasks=False, enable_turn_tracking=False)
    runner = mods["WorkerRunner"](handle_sigint=False)
    run = asyncio.create_task(runner.run(worker))
    await trace.done.wait()
    await worker.stop_when_done()
    await run
    return trace


def ami_jobs(a) -> list[tuple]:
    man = pd.read_parquet(a.ami_manifest)
    rng = np.random.default_rng(a.seed)
    com = man[man.label == "complete"].sample(min(a.n_complete, (man.label == "complete").sum()), random_state=a.seed)
    inc = man[man.label == "incomplete"].sample(min(a.n_incomplete, (man.label == "incomplete").sum()), random_state=a.seed)
    jobs, cache = [], {}
    for r in pd.concat([com, inc]).itertuples(index=False):
        if r.audio not in cache:
            cache[r.audio] = sf.read(r.audio, dtype="float32")[0]
        x = cache[r.audio]
        t = float(r.t)
        s0, s1 = max(0, int((t - a.pre) * SR)), min(len(x), int((t + a.post) * SR))
        seg = x[s0:s1].copy(); off = int(t * SR) - s0
        if r.label == "complete" and r.b_onset_s is not None and not np.isnan(r.b_onset_s):
            cut = off + int(r.b_onset_s * SR)
            if cut < len(seg):
                seg = np.concatenate([seg[:cut], room_tone(seg[:cut], SR, off / SR, len(seg) - cut, rng)])
        seg16 = np.clip(seg * 32768.0, -32768, 32767).astype(np.int16)
        if a.normalize_dbfs is not None:
            seg16 = H.normalize_level(seg16, off, a.normalize_dbfs)
        jobs.append((f"{r.meeting}/{r.speaker}/{t:.3f}", int(r.label == "complete"), float(r.gap_s) if r.gap_s is not None else None, seg16, off))
    return jobs


FIELDS = H.FIELDS + ["eager_s", "cancel_s", "confirmed", "cpu_share", "max_rss_mb", "append_ms_p99", "append_ms_max", "analyze_ms", "n_stale_decisions"]


def timing_fields(trace: Trace) -> dict:
    """Pipeline-task wall times of the streaming analyzer over the replay."""
    an = trace.analyzer
    if an is None or not getattr(an, "append_ms", None):
        return {"append_ms_p99": "", "append_ms_max": "", "analyze_ms": "", "n_stale_decisions": ""}
    t0 = trace.t_offset
    stale = sum(1 for q in trace.queries if t0 is not None and q["t"] >= t0 and q.get("pending", 0) > 0)
    return {"append_ms_p99": f"{np.percentile(an.append_ms, 99):.2f}", "append_ms_max": f"{max(an.append_ms):.2f}",
            "analyze_ms": f"{max(an.analyze_ms):.1f}" if an.analyze_ms else "", "n_stale_decisions": stale}


def speculation_fields(trace: Trace) -> dict:
    """Speculation events of the silence under test, relative to the acoustic offset (the strategy's
    rule table only fires on this silence; the pre-roll pauses can also speculate, so take the first
    event after the offset and the first stop that follows it)."""
    t0 = trace.t_offset
    if t0 is None:
        return {"eager_s": "", "cancel_s": "", "confirmed": ""}
    eag = [t for t in trace.eager if t >= t0 - 0.05]
    stops = [(t, c) for t, c in trace.confirms if t >= t0 - 0.05]
    out = {"eager_s": f"{eag[0] - t0:.3f}" if eag else "", "cancel_s": "", "confirmed": ""}
    if eag:
        can = [t for t in trace.cancels if t >= eag[0] and (not stops or t <= stops[0][0] + 1e-3)]
        out["cancel_s"] = f"{can[0] - t0:.3f}" if can else ""
    if stops:
        out["confirmed"] = int(stops[0][1])
    return out


def report_two_stage(paths: list[str], L: float, timeout: float) -> str:
    """Per rule: what the user experiences with a response latency L (inference trigger -> first bot
    audio). Completes: response time = max(commit, eager + L) if the commit confirmed a speculation,
    else commit + L (a complete with no stop is charged ``timeout``). Pauses: a generation is spent when
    an eager or a commit happened before the speaker resumed; a false alarm is a commit before the
    resumption (strict, as in B3) and *audible* if the response time came before it too."""
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    by_rule: dict[str, list[dict]] = {}
    for r in rows:
        by_rule.setdefault(r["rule"], []).append(r)

    def f(x):
        return float(x) if x not in ("", "None", None) else float("nan")

    def resp(r):
        lat = f(r["latency_s"]); lat = timeout if np.isnan(lat) else lat
        e = f(r.get("eager_s", "")); conf = r.get("confirmed", "") == "1"
        return max(lat, e + L) if (not np.isnan(e) and conf) else lat + L

    lines = [f"Response latency L = {L:g} s; completes with no stop charged {timeout:g} s.", "",
             "| rule | n complete / incomplete | generations on pauses | turn splits (strict FAR) | audible FAR | "
             "turn ends committed by the model (not the backstop) | eager on turn ends / held / median lead to the commit | "
             "commit latency p50 / p90 / mean | response time p50 / p90 / mean |", "|---|---|---|---|---|---|---|---|---|"]
    for rule, rs in by_rule.items():
        pos = [r for r in rs if r["label"] == "1"]
        neg = [r for r in rs if r["label"] == "0"]
        lat = np.array([f(r["latency_s"]) for r in pos]); lat_f = np.where(np.isnan(lat), timeout, lat)
        rt = np.array([resp(r) for r in pos])
        by_model = np.mean([r["committed_at"] not in ("", "stop_secs") for r in pos]) if pos else float("nan")
        eag = np.array([f(r.get("eager_s", "")) for r in pos])
        held = np.array([r.get("confirmed", "") == "1" for r in pos])
        lead = lat_f[held & ~np.isnan(eag)] - eag[held & ~np.isnan(eag)]
        gen, fa, aud = [], [], []
        for r in neg:
            gap = f(r["gap_s"]); gap = np.inf if np.isnan(gap) else gap
            l = f(r["latency_s"]); e = f(r.get("eager_s", ""))
            first = np.nanmin([l, e]) if not (np.isnan(l) and np.isnan(e)) else np.inf
            gen.append(first < gap)
            fa.append(not np.isnan(l) and l < gap)
            aud.append(not np.isnan(l) and l < gap and resp(r) < gap)
        lines.append(f"| {rule} | {len(pos)} / {len(neg)} | {np.mean(gen):.1%} | {np.mean(fa):.1%} | {np.mean(aud):.1%} | {by_model:.2f} | "
                     f"{np.mean(~np.isnan(eag)):.2f} / {held.mean():.2f} / {np.median(lead) if len(lead) else float('nan'):.2f} s | "
                     f"{np.percentile(lat_f, 50):.2f} / {np.percentile(lat_f, 90):.2f} / {lat_f.mean():.2f} s | "
                     f"{np.percentile(rt, 50):.2f} / {np.percentile(rt, 90):.2f} / {rt.mean():.2f} s |")
    return "\n".join(lines)


async def main_async(a):
    mods = H.build_pipeline_modules()
    if a.quiet:
        from loguru import logger
        logger.remove(); logger.add(sys.stderr, level="WARNING")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    jobs = ami_jobs(a)
    rule = a.rule if a.analyzer != "stock" else "one_shot"
    rule_label = f"{rule}+{a.eager_rule or 'noeager'}" if a.strategy == "eager" else rule
    done = set()
    if out.exists() and a.resume:
        done = {r["id"] for r in csv.DictReader(open(out))}
    mode = "a" if (out.exists() and a.resume) else "w"
    with out.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        t_start = time.time()
        import resource
        for n, (rid, label, gap, audio, off) in enumerate(jobs, 1):
            if rid in done:
                continue
            ru0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.time()
            trace = await replay(mods, audio, off, a.analyzer, rule, a)
            ru1, w1 = resource.getrusage(resource.RUSAGE_SELF), time.time()
            row = H.summarize_trace(trace, rid, f"{a.analyzer}:{rule_label}", label, gap, a.vad_stop_secs)
            row.update(speculation_fields(trace))
            row.update(timing_fields(trace))
            # whole-process CPU share over this replay (pipeline + VAD + analyzer) and peak RSS so far
            cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
            row["cpu_share"] = f"{cpu_s / max(w1 - w0, 1e-6):.3f}"
            row["max_rss_mb"] = f"{ru1.ru_maxrss / (1 << 20):.0f}" if sys.platform == "darwin" else f"{ru1.ru_maxrss / 1024:.0f}"
            w.writerow(row); f.flush()
            print(f"[{n}/{len(jobs)} {time.time() - t_start:.0f} s] {rid} label={label} gap={gap} latency={row['latency_s'] or 'none'} "
                  f"at={row['committed_at'] or '-'} eager={row['eager_s'] or '-'} cancel={row['cancel_s'] or '-'} "
                  f"confirmed={row['confirmed']} queries={row['n_queries']}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyzer", default="streaming", choices=["streaming", "stock", "phase0"])
    ap.add_argument("--rule", default="far10", help="streaming: far5 | far10 | far15 | one_shot; phase0: see local_smart_turn_multimodal.RULES")
    ap.add_argument("--encoder", default="data/models/smart-turn-v3.2-gpu.pooled.onnx")
    ap.add_argument("--heads", default="data/models/eot_heads_v32.onnx")
    ap.add_argument("--stock-model", default="data/models/smart-turn-v3.2-gpu.onnx", help="'' = pipecat's bundled int8 model")
    ap.add_argument("--phase0-model", default="data/models/smart-turn-multimodal-audio.onnx")
    ap.add_argument("--threads", type=int, default=1, help="ORT intra-op threads of the encoder (streaming) or the model (stock)")
    ap.add_argument("--no-worker", action="store_true", help="streaming: run the encoder refresh inline on the pipeline task (B3 behaviour)")
    ap.add_argument("--lead", type=float, default=0.0, help="streaming worker: request a silence-clock refresh this early (s)")
    ap.add_argument("--wait", type=float, default=0.05, help="streaming worker: await the refresh at the VAD stop up to this long (s)")
    ap.add_argument("--max-defer", type=float, default=0.08, help="streaming worker: how long a silence-clock decision may wait for its refresh (s)")
    ap.add_argument("--strategy", default="analyzer", choices=["analyzer", "eager"],
                    help="stop strategy: pipecat's TurnAnalyzerUserTurnStopStrategy or the eager one (Stage C1)")
    ap.add_argument("--eager-rule", default="", help="streaming + eager: an EAGER_RULES name, e.g. far15_gen30")
    ap.add_argument("--speculation-timeout", type=float, default=5.0)
    ap.add_argument("--response-latency", type=float, default=0.8, help="for --report-two-stage: inference trigger -> first bot audio, s")
    ap.add_argument("--report-two-stage", nargs="*")
    ap.add_argument("--ami-manifest", default="data/a2/ami/manifest_test.parquet")
    ap.add_argument("--n-complete", type=int, default=80); ap.add_argument("--n-incomplete", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre", type=float, default=9.0); ap.add_argument("--post", type=float, default=3.6)
    ap.add_argument("--stop-secs", type=float, default=3.0); ap.add_argument("--vad-stop-secs", type=float, default=0.2)
    ap.add_argument("--vad-start-secs", type=float, default=0.2)
    ap.add_argument("--normalize-dbfs", type=float, default=-23.0)
    ap.add_argument("--out", default="runs/b3/out.csv"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", nargs="*"); ap.add_argument("--timeout", type=float, default=3.2)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(H.report(a.report, a.timeout, a.vad_start_secs))
        return
    if a.report_two_stage:
        print(report_two_stage(a.report_two_stage, a.response_latency, a.timeout))
        return
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
