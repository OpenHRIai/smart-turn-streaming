"""End-to-end trigger latency of the smart-turn-multimodal analyzer inside a real pipecat pipeline.

Plays recorded speech into a pipeline in real time and measures the wall-clock time from the acoustic
offset (the speaker's last speech) to pipecat's `on_user_turn_stopped` event, for the one-shot rule
(query once at the VAD stop, θ 0.5, else `stop_secs`) and the silence-aware re-query rules of
docs/design_streaming_eot.md §9 as implemented by `local_smart_turn_multimodal.py`. Unlike
eval_silence_requery.py, the numbers here include everything between the microphone and the decision:
the VAD's `stop_secs`, pipecat's frame handling, the model runs and the analyzer's scheduling.

Pipeline (pipecat 1.12): WavSource -> VADProcessor(SileroVADAnalyzer) ->
UserTurnProcessor(start=VAD, stop=TurnAnalyzerUserTurnStopStrategy(analyzer, wait_for_transcript=False)).
Audio is pushed in 20 ms InputAudioRawFrames paced by the wall clock, as a transport would.

Inputs, one of:
  --manifest <OpenVideoFDB>/manifest/endpoint_naturalistic_test.jsonl --root <OpenVideoFDB>
        Seamless end-of-turn silences (see eval_silence_requery.py). For each sampled silence the speaker's
        source audio is replayed from `--pre` s before the VAD stop `t_raw` (the acoustic offset) to `--post` s
        after it, so the pipeline sees the real end of the turn and the real continuation (silence, the
        speaker resuming, backchannels). The 16 kHz decode is cached in --cache.
  --wav a.wav [b.wav ...] [--offset-s T]
        Any 16 kHz mono wavs; the acoustic offset is --offset-s or, by default, the last sample above
        -35 dBFS. Everything after it is treated as silence to be endpointed.

Per silence the CSV has: the rule, label, gap to the speaker's resumption, wall-clock latency from the
acoustic offset to the turn stop (NaN when no stop happened in the window), the offset at which the
rule committed (0.2 / 0.6 / 1.2 s, or "stop_secs"), pipecat's own VAD-stop time relative to the acoustic
offset, every model query (time, offset, p, θ, decision), and the compute time of each query.
`--report <csv> [<csv> ...]` summarises CSVs (recall, false alarms, p50/p90/mean latency) as markdown.

usage:
  python measure_pipecat_latency.py --manifest ... --root ... --rule far15_min_latency --n-complete 40 --n-incomplete 40 --out runs/pipecat_latency/far15.csv
  python measure_pipecat_latency.py --report runs/pipecat_latency/*.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

SR = 16000
FRAME_MS = 20


# --- audio -----------------------------------------------------------------------------------

def decode_16k(src: Path, cache: Path | None) -> np.ndarray:
    """Whole file as int16 16 kHz mono via ffmpeg (as eval_silence_requery.py does)."""
    if cache is not None:
        npy = cache / (src.stem + ".npy")
        if npy.exists():
            return np.load(npy, mmap_mode="r")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(npy, audio)
    return audio


def load_wav_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, need {SR} Hz (resample with ffmpeg first)")
    return audio


def normalize_level(audio: np.ndarray, offset_sample: int, target_dbfs: float, ref_secs: float = 3.0) -> np.ndarray:
    """Scale so that the RMS of the `ref_secs` before the acoustic offset is `target_dbfs` (peak-limited to -1 dBFS).

    The Seamless recordings sit anywhere between -25 and -56 dBFS RMS; pipecat's VAD gates on loudness
    (`min_volume`), so without this the quiet speakers lose their soft endings or are never detected,
    which a live microphone feed would not do."""
    ref = audio[max(0, offset_sample - int(ref_secs * SR)): offset_sample].astype(np.float32) / 32768.0
    rms = float(np.sqrt((ref ** 2).mean() + 1e-12))
    gain = 10 ** (target_dbfs / 20) / rms
    peak = float(np.abs(audio).max()) / 32768.0
    gain = min(gain, 10 ** (-1 / 20) / max(peak, 1e-6))
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def last_speech_sample(audio: np.ndarray, dbfs: float = -35.0, win_ms: float = 20) -> int:
    """Index just after the last 20 ms frame whose RMS is above `dbfs`."""
    n = int(SR * win_ms / 1000)
    x = audio[: len(audio) // n * n].astype(np.float32).reshape(-1, n) / 32768.0
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-12)
    above = np.nonzero(20 * np.log10(rms) > dbfs)[0]
    return int((above[-1] + 1) * n) if len(above) else len(audio)


# --- pipeline --------------------------------------------------------------------------------

def build_pipeline_modules():
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import (
        EndFrame, Frame, InputAudioRawFrame, StartFrame, VADUserStoppedSpeakingFrame, VADUserStartedSpeakingFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.workers.runner import WorkerRunner
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.turns.user_start import VADUserTurnStartStrategy
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    return locals()


class Trace:
    """Wall-clock events of one replay (time.time() everywhere, like pipecat's VAD frame timestamps)."""

    def __init__(self):
        self.t_offset: float | None = None  # when the frame holding the acoustic-offset sample was pushed
        self.vad_starts: list[float] = []
        self.vad_stops: list[float] = []  # VADUserStoppedSpeakingFrame.timestamp
        self.turn_starts: list[float] = []
        self.turn_stops: list[float] = []
        self.queries: list[dict] = []  # {t, offset_s, p, theta, decision, ms}
        self.done = asyncio.Event()


def make_processors(mods, trace: Trace, audio: np.ndarray, offset_sample: int, analyzer):
    FrameProcessor = mods["FrameProcessor"]
    FrameDirection = mods["FrameDirection"]
    InputAudioRawFrame = mods["InputAudioRawFrame"]
    StartFrame = mods["StartFrame"]
    VADUserStoppedSpeakingFrame = mods["VADUserStoppedSpeakingFrame"]
    VADUserStartedSpeakingFrame = mods["VADUserStartedSpeakingFrame"]

    class WavSource(FrameProcessor):
        """Pushes `audio` as 20 ms InputAudioRawFrames at real-time pace, like an input transport."""

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
                    # the frame containing the last speech sample is on its way now
                    trace.t_offset = time.time() - (i + n - offset_sample) / SR
                chunk = np.ascontiguousarray(audio[i: i + n]).tobytes()
                await self.push_frame(InputAudioRawFrame(audio=chunk, sample_rate=SR, num_channels=1))
            await asyncio.sleep(0.1)  # let the last frames settle before the harness ends the pipeline
            trace.done.set()

    class Tap(FrameProcessor):
        """Records VAD frames downstream of the turn processor."""

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                trace.vad_stops.append(frame.timestamp)
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                trace.vad_starts.append(frame.timestamp)
            await self.push_frame(frame, direction)

    vad = mods["VADProcessor"](vad_analyzer=mods["SileroVADAnalyzer"](params=mods["VADParams"](stop_secs=analyzer.params.vad_stop_secs)))
    strategies = mods["UserTurnStrategies"](
        start=[mods["VADUserTurnStartStrategy"]()],
        stop=[mods["TurnAnalyzerUserTurnStopStrategy"](turn_analyzer=analyzer, wait_for_transcript=False)],
    )
    turns = mods["UserTurnProcessor"](user_turn_strategies=strategies, user_turn_stop_timeout=30.0)

    @turns.event_handler("on_user_turn_started")
    async def _started(processor, strategy):
        trace.turn_starts.append(time.time())

    @turns.event_handler("on_user_turn_stopped")
    async def _stopped(processor, strategy):
        trace.turn_stops.append(time.time())

    return [WavSource(), vad, turns, Tap()]


def make_analyzer(model_path: str, rule: str, trace: Trace, stop_secs: float, vad_stop_secs: float):
    from local_smart_turn_multimodal import LocalSmartTurnMultimodalAnalyzer, SmartTurnMultimodalParams

    class TracingAnalyzer(LocalSmartTurnMultimodalAnalyzer):
        def _predict_endpoint(self, audio_array):
            t0 = time.perf_counter()
            r = super()._predict_endpoint(audio_array)
            trace.queries.append({"t": time.time(), "offset_s": r["offset_s"], "p": r["probability"], "theta": r["threshold"],
                                  "decision": r["prediction"], "ms": (time.perf_counter() - t0) * 1e3})
            return r

    params = SmartTurnMultimodalParams.from_rule(rule, stop_secs=stop_secs, vad_stop_secs=vad_stop_secs)
    analyzer = TracingAnalyzer(model_path=model_path, params=params)
    analyzer._predict_endpoint(np.zeros(8 * SR, np.float32))  # warm up (first ONNX run is slow)
    trace.queries.clear()
    return analyzer


async def replay(mods, audio: np.ndarray, offset_sample: int, model_path: str, rule: str, stop_secs: float, vad_stop_secs: float) -> Trace:
    trace = Trace()
    analyzer = make_analyzer(model_path, rule, trace, stop_secs, vad_stop_secs)
    procs = make_processors(mods, trace, audio, offset_sample, analyzer)
    pipeline = mods["Pipeline"](procs)
    worker = mods["PipelineWorker"](pipeline, params=mods["PipelineParams"](audio_in_sample_rate=SR),
                                    idle_timeout_secs=None, check_dangling_tasks=False, enable_turn_tracking=False)
    runner = mods["WorkerRunner"](handle_sigint=False)
    run = asyncio.create_task(runner.run(worker))
    await trace.done.wait()
    await worker.stop_when_done()
    await run
    return trace


# --- rows ------------------------------------------------------------------------------------

FIELDS = ["id", "rule", "label", "gap_s", "latency_s", "committed_at", "vad_stop_rel_s", "pipecat_offset_rel_s",
          "latency_from_pipecat_offset_s", "n_queries", "queries", "query_ms_mean", "n_turn_stops", "premature_stops",
          "turn_started"]


def summarize_trace(trace: Trace, row_id: str, rule: str, label, gap_s, vad_stop_secs: float) -> dict:
    t0 = trace.t_offset
    stops_after = [t for t in trace.turn_stops if t0 is not None and t >= t0 - 0.05]
    stops_before = [t for t in trace.turn_stops if t0 is not None and t < t0 - 0.05]
    latency = (stops_after[0] - t0) if stops_after else float("nan")
    queries_after = [q for q in trace.queries if t0 is not None and q["t"] >= t0]
    committed = ""
    if stops_after:
        firing = [q for q in queries_after if q["decision"] == 1 and q["t"] <= stops_after[0] + 1e-3]
        committed = f"{firing[-1]['offset_s']:g}" if firing else "stop_secs"
    vad_stops_after = [t for t in trace.vad_stops if t0 is not None and t >= t0 - 0.05]
    vad_stop_rel = (vad_stops_after[0] - t0) if vad_stops_after else float("nan")
    # pipecat's own acoustic offset: its last VAD stop at or before the turn stop, minus the VAD's stop_secs
    pipecat_offset_rel = float("nan"); lat_pc = float("nan")
    if stops_after:
        vs = [t for t in trace.vad_stops if t <= stops_after[0] + 1e-3]
        if vs:
            pipecat_offset_rel = vs[-1] - vad_stop_secs - t0
            lat_pc = stops_after[0] - (vs[-1] - vad_stop_secs)
    return {
        "id": row_id, "rule": rule, "label": label, "gap_s": gap_s,
        "latency_s": f"{latency:.3f}" if not math.isnan(latency) else "",
        "committed_at": committed,
        "vad_stop_rel_s": f"{vad_stop_rel:.3f}" if not math.isnan(vad_stop_rel) else "",
        "pipecat_offset_rel_s": f"{pipecat_offset_rel:.3f}" if not math.isnan(pipecat_offset_rel) else "",
        "latency_from_pipecat_offset_s": f"{lat_pc:.3f}" if not math.isnan(lat_pc) else "",
        "n_queries": len(queries_after),
        "queries": json.dumps([{"t": round(q["t"] - t0, 3), "off": q["offset_s"], "p": round(q["p"], 4), "th": q["theta"], "d": q["decision"],
                                "ms": round(q["ms"], 1), **({"stale": round(q["stale_s"], 3), "pend": q["pending"]} if "pending" in q else {})}
                               for q in queries_after]) if t0 is not None else "[]",
        "query_ms_mean": f"{np.mean([q['ms'] for q in trace.queries]):.1f}" if trace.queries else "",
        "n_turn_stops": len(trace.turn_stops), "premature_stops": len(stops_before),
        "turn_started": int(bool(trace.turn_starts)),
    }


def report(paths: list[str], timeout: float, vad_start_secs: float = 0.2) -> str:
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    by_rule: dict[str, list[dict]] = {}
    for r in rows:
        by_rule.setdefault(r["rule"], []).append(r)

    def f(x):
        return float(x) if x not in ("", "None", None) else float("nan")

    lines = ["| rule | n complete / incomplete | recall (turn stopped in window) | false alarms, strict / effective | "
             "latency from acoustic offset, completes: p50 / p90 / mean | from pipecat's VAD offset: p50 / p90 | "
             "committed at 0.2 / 0.6 / 1.2 s / stop_secs / none | model ms per query |", "|---|---|---|---|---|---|---|---|"]
    for rule, rs in by_rule.items():
        pos = [r for r in rs if r["label"] == "1"]
        neg = [r for r in rs if r["label"] == "0"]
        lat = np.array([f(r["latency_s"]) for r in pos])
        recall = np.mean(~np.isnan(lat)) if len(pos) else float("nan")
        lat_f = np.where(np.isnan(lat), timeout, lat)
        lat_pc = np.array([f(r["latency_from_pipecat_offset_s"]) for r in pos])
        lat_pc = np.where(np.isnan(lat_pc), timeout, lat_pc)
        strict, eff = [], []
        for r in neg:
            gap = f(r["gap_s"]); gap = np.inf if math.isnan(gap) else gap
            l = f(r["latency_s"])
            strict.append(not math.isnan(l) and l < gap)  # stopped before the speaker resumed
            eff.append(not math.isnan(l) and l < gap + vad_start_secs)  # ... or before the VAD could have noticed
        where = {k: sum(1 for r in pos if r["committed_at"] == k) for k in ("0.2", "0.6", "1.2", "stop_secs")}
        where["none"] = sum(1 for r in pos if not r["committed_at"])
        ms = np.array([f(r["query_ms_mean"]) for r in rs if r["query_ms_mean"]])
        lines.append(
            f"| {rule} | {len(pos)} / {len(neg)} | {recall:.2f} | {np.mean(strict):.1%} / {np.mean(eff):.1%} | "
            f"{np.percentile(lat_f, 50):.2f} / {np.percentile(lat_f, 90):.2f} / {lat_f.mean():.2f} s | "
            f"{np.percentile(lat_pc, 50):.2f} / {np.percentile(lat_pc, 90):.2f} s | "
            f"{where['0.2']} / {where['0.6']} / {where['1.2']} / {where['stop_secs']} / {where['none']} | {np.nanmean(ms):.0f} |")
    lines += ["", "Latency of the committed turns by committing offset (completes and incompletes), measured from pipecat's own "
              "VAD offset, so the overhead over the nominal offset is the pipeline's: VAD frame granularity, frame handling, "
              "the model run and the analyzer's one-frame poll.", "",
              "| rule | committed at | n | p50 | p90 | median overhead over the nominal offset |", "|---|---|---|---|---|---|"]
    for rule, rs in by_rule.items():
        for k in ("0.2", "0.6", "1.2", "stop_secs"):
            sel = np.array([f(r["latency_from_pipecat_offset_s"]) for r in rs if r["committed_at"] == k and r["latency_from_pipecat_offset_s"]])
            if len(sel):
                nominal = float(k) if k != "stop_secs" else timeout
                lines.append(f"| {rule} | {k} | {len(sel)} | {np.percentile(sel, 50):.3f} s | {np.percentile(sel, 90):.3f} s | "
                             f"{np.median(sel - nominal) * 1e3:+.0f} ms |")
    dis = [r for r in rows if r["pipecat_offset_rel_s"] and abs(f(r["pipecat_offset_rel_s"])) > 0.3]
    lines += ["", f"Rows where pipecat's VAD offset differs from the dataset's `t_raw` by more than 0.3 s: {len(dis)} of "
              f"{sum(1 for r in rows if r['pipecat_offset_rel_s'])} committed turns "
              f"(median |Δ| over all: {np.median([abs(f(r['pipecat_offset_rel_s'])) for r in rows if r['pipecat_offset_rel_s']]) * 1e3:.0f} ms)."]
    return "\n".join(lines)


# --- main ------------------------------------------------------------------------------------

async def main_async(args):
    mods = build_pipeline_modules()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    jobs = []  # (id, label, gap_s, audio, offset_sample)
    if args.manifest:
        root = Path(args.root)
        rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
        rng = random.Random(args.seed)
        pos = [r for r in rows if int(r["label"]) == 1]
        neg = [r for r in rows if int(r["label"]) == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        picked = pos[: args.n_complete] + neg[: args.n_incomplete]
        for r in picked:
            src = root / "data" / "seamless" / "raw" / r["label_set"] / r["split"] / "audio" / f"{r['file_id']}.wav"
            audio = decode_16k(src, cache)
            t = float(r["t_raw"])
            a, b = max(0, int((t - args.pre) * SR)), min(len(audio), int((t + args.post) * SR))
            seg, off = np.array(audio[a:b]), int(t * SR) - a
            if args.normalize_dbfs is not None:
                seg = normalize_level(seg, off, args.normalize_dbfs)
            jobs.append((r["id"], int(r["label"]), r.get("gap_s"), seg, off))
    else:
        for w in args.wav:
            audio = load_wav_16k(Path(w))
            off = int(args.offset_s * SR) if args.offset_s is not None else last_speech_sample(audio)
            if args.post and len(audio) < off + int(args.post * SR):
                audio = np.concatenate([audio, np.zeros(off + int(args.post * SR) - len(audio), np.int16)])
            if args.normalize_dbfs is not None:
                audio = normalize_level(audio, off, args.normalize_dbfs)
            jobs.append((Path(w).name, 1, None, audio, off))

    done_ids = set()
    if out.exists() and args.resume:
        done_ids = {r["id"] for r in csv.DictReader(open(out)) if r["rule"] == args.rule}
    mode = "a" if (out.exists() and args.resume) else "w"
    with out.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        t_start = time.time()
        for n, (rid, label, gap, audio, off) in enumerate(jobs, 1):
            if rid in done_ids:
                continue
            trace = await replay(mods, audio, off, args.model, args.rule, args.stop_secs, args.vad_stop_secs)
            row = summarize_trace(trace, rid, args.rule, label, gap, args.vad_stop_secs)
            w.writerow(row); f.flush()
            print(f"[{n}/{len(jobs)} {time.time() - t_start:.0f} s] {rid} label={label} gap={gap} latency={row['latency_s'] or 'none'} "
                  f"at={row['committed_at'] or '-'} vad_stop={row['vad_stop_rel_s']} queries={row['queries']}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest"); ap.add_argument("--root", default=".")
    ap.add_argument("--wav", nargs="*", default=[]); ap.add_argument("--offset-s", type=float, default=None)
    ap.add_argument("--model", default="smart-turn-multimodal-audio.onnx")
    ap.add_argument("--rule", default="far15_min_latency", help="one_shot | far15_min_latency | far10_min_latency (local_smart_turn_multimodal.RULES)")
    ap.add_argument("--stop-secs", type=float, default=3.0, help="BaseSmartTurn stop_secs backstop")
    ap.add_argument("--vad-stop-secs", type=float, default=0.2, help="Silero VAD stop_secs (= first query offset)")
    ap.add_argument("--vad-start-secs", type=float, default=0.2, help="Silero VAD start_secs; used by --report for effective false alarms")
    ap.add_argument("--n-complete", type=int, default=40); ap.add_argument("--n-incomplete", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre", type=float, default=9.0, help="s of source audio replayed before the acoustic offset")
    ap.add_argument("--post", type=float, default=3.6, help="s replayed after it (must exceed vad stop_secs + stop_secs for the backstop to fire)")
    ap.add_argument("--cache", default=None, help="directory for decoded 16 kHz source audio (.npy)")
    ap.add_argument("--normalize-dbfs", type=float, default=-23.0,
                    help="scale each replayed window so the 3 s before the offset have this RMS (dBFS); 'nan' disables")
    ap.add_argument("--out", default="runs/pipecat_latency/out.csv")
    ap.add_argument("--resume", action="store_true", help="skip ids already in --out for this rule")
    ap.add_argument("--report", nargs="*", help="summarise these CSVs and exit")
    ap.add_argument("--timeout", type=float, default=3.2, help="latency charged to a miss in --report (VAD stop_secs + stop_secs)")
    ap.add_argument("--quiet", action="store_true", help="silence pipecat's debug logging")
    args = ap.parse_args()
    if args.report is not None:
        print(report(args.report, args.timeout, args.vad_start_secs))
        return
    if not args.manifest and not args.wav:
        ap.error("give --manifest/--root or --wav")
    if args.quiet:
        logger.remove(); logger.add(sys.stderr, level="WARNING")
    if args.normalize_dbfs is not None and math.isnan(args.normalize_dbfs):
        args.normalize_dbfs = None
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
