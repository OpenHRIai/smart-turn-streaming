"""Pipecat turn analyzer around the streaming detector (Stage B3).

`StreamingEOTAnalyzer` is a `BaseTurnAnalyzer` (pipecat >= 1.12; written against 1.14). Unlike
`BaseSmartTurn` it is not a batch model asked once at the VAD stop: every audio frame goes through
`StreamingEOT`, which keeps the Whisper features incrementally, refreshes the frozen encoder on the
silence clock and runs the trained heads every 100 ms. The decision rule is the re-query table of
Stage B1/B3: at silence offsets ``offsets`` (0.2 / 0.5 / 1.2 s after the acoustic offset) commit
COMPLETE when ``p_end >= theta_k``; ``stop_secs`` stays as the backstop. Stage C1 adds a second,
looser table ``eager_thresholds`` on the same offsets: the first offset with ``p_end >= theta_eager_k``
raises an *eager* end of turn (``pop_eager()``), which `eot.pipecat_eager_strategy` turns into a
speculative LLM inference whose response Pipecat holds until the commit (or discards when the user
resumes). Eager tables fitted on AMI dev under a budget of generations spent on pauses are in
``EAGER_RULES`` (`runs/c1/eager_tables.csv`).

How it meets pipecat's machinery. `TurnAnalyzerUserTurnStopStrategy` feeds every frame to
`append_audio(buffer, is_speech)`, where ``is_speech`` is the VAD's *state* (it turns false only
``vad_stop_secs`` after the last speech), and honours a COMPLETE returned from there; at the VAD
stop it also calls `analyze_end_of_turn()`. This analyzer decides inside `append_audio` on the
silence-clock hops (which land on the rule's offsets) and answers `analyze_end_of_turn()` with the
verdict of the 0.2 s point. Tables fitted on AMI dev are in `RULES` (`runs/b2/theta_tables.csv`).

Compute (Stage C2): the 0.7 ms heads hop and the 0.02 ms mel update run on the pipeline task; the
22 ms encoder refresh runs on one worker thread (``worker=True``, the default). At the 0.5 / 1.2 s
silence-clock points the decision waits for its refresh for up to ``max_defer_s`` (80 ms) on the frame
clock, so it is the inline decision taken 20-60 ms later (``lead_s`` > 0 instead requests the refresh
ahead and decides at the point); the 0.2 s decision at the VAD stop is taken in ``analyze_end_of_turn``, which awaits the refresh for
at most ``wait_s`` without blocking the event loop (50 ms by default: the encoder takes 22 ms on an M4 and
46 ms when the core was idle, so a one-frame wait of 20 ms misses it; C2 measured both) and otherwise
decides on the cached vector, recording the staleness in the decision trace (``stale_s``, ``pending``). ``worker=False`` restores the B3 inline
behaviour (everything on the pipeline task), which is what B3 measured.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS
from pipecat.metrics.metrics import MetricsData

from eot.streaming import SR, StreamingEOT


class RuleTable(NamedTuple):
    offsets: tuple[float, ...]
    thresholds: tuple[float, ...]


#: Thresholds fitted on AMI dev for the B1 heads (runs/b1/knob_ami07_auc), min-latency objective.
RULES: dict[str, RuleTable] = {
    "far5": RuleTable((0.2, 0.5, 1.2), (0.94, 0.94, 0.94)),
    "far10": RuleTable((0.2, 0.5, 1.2), (0.92, 0.98, 0.86)),
    "far15": RuleTable((0.2, 0.5, 1.2), (0.98, 0.88, 0.80)),
    "one_shot": RuleTable((0.2,), (0.5,)),
}

#: Eager tables (same offsets as the commit rule), fitted on AMI dev at a budget of generations spent
#: on pauses (20 / 30 / 40 %), minimising the perceived response time at 0.8 s of response latency.
EAGER_RULES: dict[str, tuple[float, ...]] = {
    "far10_gen20": (0.92, 0.84, 0.68),
    "far10_gen30": (0.92, 0.54, 0.20),
    "far10_gen40": (0.74, 0.24, 0.02),
    "far15_gen20": (0.92, 0.84, 0.68),
    "far15_gen30": (0.84, 0.56, 0.42),
    "far15_gen40": (0.72, 0.24, 0.04),
}


class StreamingEOTParams(BaseTurnParams):
    offsets: tuple[float, ...] = RULES["far10"].offsets
    thresholds: tuple[float, ...] = RULES["far10"].thresholds
    #: eager table on the same offsets (None = no eager signal); each entry <= the commit threshold
    eager_thresholds: tuple[float, ...] | None = None
    stop_secs: float = 3.0
    vad_stop_secs: float = VAD_STOP_SECS

    @classmethod
    def from_rule(cls, rule: str | RuleTable, eager: str | tuple[float, ...] | None = None, **kw) -> "StreamingEOTParams":
        t = RULES[rule] if isinstance(rule, str) else rule
        e = EAGER_RULES[eager] if isinstance(eager, str) else eager
        if e is not None and len(e) != len(t.offsets):
            raise ValueError("eager table must have one threshold per offset")
        return cls(offsets=t.offsets, thresholds=t.thresholds, eager_thresholds=tuple(e) if e is not None else None, **kw)


class StreamingEOTAnalyzer(BaseTurnAnalyzer):
    def __init__(self, *, encoder_onnx: str, heads_onnx: str, params: StreamingEOTParams | None = None,
                 threads: int = 1, sample_rate: int | None = None, worker: bool = True, lead_s: float = 0.0,
                 wait_s: float = 0.05, max_defer_s: float = 0.08):
        super().__init__(sample_rate=sample_rate)
        self._params = params or StreamingEOTParams()
        self._eot = StreamingEOT(encoder_onnx, heads_onnx, threads=threads, vad_stop_secs=self._params.vad_stop_secs,
                                 worker=worker, lead_s=lead_s, wait_s=wait_s, max_defer_s=max_defer_s)
        self.append_ms: list[float] = []     # wall time of every append_audio call on the pipeline task
        self.analyze_ms: list[float] = []    # wall time of every analyze_end_of_turn call (incl. the awaited refresh)
        self._speech_triggered = False
        self._committed: tuple[EndOfTurnState, MetricsData | None] | None = None
        self._decided = set()          # rule offsets already decided in this silence
        self._eager_fired = False      # an eager end of turn was raised in this silence
        self._eager_pending = False    # ... and not yet collected by pop_eager()
        self.decisions: list[dict] = []  # trace: every decision point {t, offset_s, p, theta, decision, ms}
        # warm up the sessions
        self._eot.refresh(); self._eot.hop(); self._eot.timings = type(self._eot.timings)()

    @property
    def speech_triggered(self) -> bool:
        return self._speech_triggered

    @property
    def params(self) -> StreamingEOTParams:
        return self._params

    @property
    def eot(self) -> StreamingEOT:
        return self._eot

    def clear(self):
        self._speech_triggered = False
        self._decided = set()
        self._eager_fired = self._eager_pending = False
        self._eot.reset_turn()

    async def cleanup(self):
        self._eot.close()

    def pop_eager(self) -> bool:
        """True once per eager end of turn: the evidence passed the eager table but not (yet) the commit
        table. The stop strategy polls this after ``append_audio``; a speech resumption clears it."""
        pending, self._eager_pending = self._eager_pending, False
        return pending

    def _check_eager(self, k: int, out: dict) -> int:
        """Eager test at rule offset k (after the commit test failed there). Returns 1 if raised now."""
        if self._params.eager_thresholds is None or self._eager_fired:
            return 0
        if out["p_end"] >= self._params.eager_thresholds[k]:
            self._eager_fired = self._eager_pending = True
            logger.debug(f"Eager end of turn at {self._params.offsets[k]:g} s: p={out['p_end']:.3f} >= {self._params.eager_thresholds[k]:g}")
            return 1
        return 0

    # --- audio in ---------------------------------------------------------------------------

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        if self.sample_rate and self.sample_rate != SR:
            raise ValueError(f"StreamingEOT needs {SR} Hz audio, pipeline runs at {self.sample_rate} Hz")
        x = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        if is_speech:
            self._speech_triggered = True
            self._decided = set()
            self._eager_fired = self._eager_pending = False
        t0 = time.perf_counter()
        try:
            return self._append(x, is_speech, t0)
        finally:
            self.append_ms.append((time.perf_counter() - t0) * 1e3)

    def _append(self, x: np.ndarray, is_speech: bool, t0: float) -> EndOfTurnState:
        out = self._eot.push(x, is_speech)
        if not self._speech_triggered or is_speech or out is None:
            return EndOfTurnState.INCOMPLETE
        tau = out["tau"]
        # backstop
        if tau >= self._params.stop_secs:
            logger.debug(f"End of Turn complete due to stop_secs ({tau:.2f} s of silence)")
            self._commit(out, None, (time.perf_counter() - t0) * 1e3)
            return EndOfTurnState.COMPLETE
        # rule offsets: decided on the silence-clock hops only (every offset is a grid point); a free-running
        # hop that lands between the offset and its grid hop would see the previous encoder vector
        if not out.get("grid"):
            return EndOfTurnState.INCOMPLETE
        for k, off in enumerate(self._params.offsets):
            if off in self._decided or tau + 1e-6 < off:
                continue
            self._decided.add(off)
            theta = self._params.thresholds[k]
            decision = int(out["p_end"] >= theta)
            eager = 0 if decision else self._check_eager(k, out)
            self.decisions.append(dict(t=time.time(), offset_s=off, p=out["p_end"], p_eager=out["p_eager"], theta=theta,
                                       decision=decision, eager=eager, ms=(time.perf_counter() - t0) * 1e3, stale_s=out["stale_s"],
                                       pending=out["pending"]))
            if decision:
                logger.debug(f"End of Turn complete at {off:g} s: p={out['p_end']:.3f} >= {theta:g}")
                self._commit(out, off, (time.perf_counter() - t0) * 1e3)
                return EndOfTurnState.COMPLETE
        return EndOfTurnState.INCOMPLETE

    def _commit(self, out: dict, off: float | None, ms: float) -> None:
        self._committed = (EndOfTurnState.COMPLETE, None)
        self.clear()

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """At the VAD stop (or right after a COMPLETE from append_audio): hand over the verdict."""
        if self._committed is not None:
            r, self._committed = self._committed, None
            return r
        # The strategy asks at the VAD stop. The 0.2 s silence-clock hop has normally just decided
        # (and returned INCOMPLETE); if the rule's first offset has not been decided yet (frame timing),
        # decide it now on a fresh hop.
        off = self._params.offsets[0]
        if off not in self._decided:
            t0 = time.perf_counter()
            self._eot.mark_silence_start()     # the VAD stop: tau = vad_stop_secs from here on
            if self._eot.worker:
                # the refresh runs on the worker; wait one frame for it without blocking the loop
                self._eot.request_refresh()
                await self._eot.await_refresh()
            else:
                self._eot.refresh()
            out = self._eot.hop(consistent=True)
            self._decided.add(off)
            theta = self._params.thresholds[0]
            decision = int(out["p_end"] >= theta)
            eager = 0 if decision else self._check_eager(0, out)
            ms = (time.perf_counter() - t0) * 1e3
            self.analyze_ms.append(ms)
            self.decisions.append(dict(t=time.time(), offset_s=off, p=out["p_end"], p_eager=out["p_eager"], theta=theta,
                                       decision=decision, eager=eager, ms=ms, stale_s=out["stale_s"], pending=out["pending"]))
            if decision:
                self.clear()
                return EndOfTurnState.COMPLETE, None
        return EndOfTurnState.INCOMPLETE, None
