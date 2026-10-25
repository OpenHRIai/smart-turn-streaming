"""Streaming end-of-turn inference (Stage B2): incremental Whisper log-mel, a slow encoder path and
a fast heads path, all on the CPU in one object.

    eot = StreamingEOT("data/models/smart-turn-v3.2-gpu.pooled.onnx", "data/models/eot_heads_v32.onnx")
    for frame, is_speech in audio_frames():          # e.g. 20 ms PCM frames + the VAD's verdict
        out = eot.push(frame, is_speech)             # None, or {"p_end", "p_eager", "tau", ...} at a hop

Timing: the heads run every ``hop_s`` (100 ms) on the cached pooled vector, the last 1 s of mel and
the silence length; the encoder is re-run on the last 8 s every ``refresh_s`` (320 ms) while the user
speaks and at the trained tau grid (``SILENCE_REFRESH_TAUS``) during silence. Every call records its
wall-clock cost in ``eot.timings``.

Worker thread (Stage C2, ``worker=True``): the 22 ms encoder runs on one background thread. The frame
path snapshots the 8 s feature window (0.6 ms), hands it over and keeps running the 0.7 ms heads hop on
the latest cached pooled vector. At a silence-clock point the refresh is requested and the decision is
postponed, frame by frame, until the vector arrives or ``max_defer_s`` (80 ms) has passed, so the decision
is the inline one taken up to the encoder's time (22-46 ms on an M4) later; an async caller awaits it for
``wait_s`` from ``await_refresh()`` instead. Optionally (``lead_s`` > 0) the refresh of a point is requested
that long ahead and the decision is taken at the point on the slightly older vector. A decision whose refresh has not arrived is taken on the cached vector and the hop output
records it (``pending`` > 0, ``stale_s``). Nothing on the frame path blocks. The heads were trained on a
pooled vector, a mel summary and a tau from the *same* window, so every request also snapshots mel20 and
tau_feat and a silence-clock hop uses the snapshot that belongs to the vector it decides on
(``hop(consistent=True)``); a mismatch of 60 ms between the two inputs cost 0.035 AUC at 1.2 s of silence
(C2 fidelity runs), the consistent pair reproduces the inline decision exactly.

Incremental mel: Whisper's log-mel for an 8 s window is a function of the *normalised* waveform
(zero mean, unit variance over the window) and clamps at the window maximum minus 8. Both are
window-global, but both are cheap to apply after the fact: scaling the waveform by 1/sigma adds
-2 log10(sigma) to every log-mel column, so the STFT columns are computed once on the raw audio
as it arrives and the window-level normalisation is applied when a window is assembled. The mean
offset is ignored (a DC term, invisible to speech bands); the reflect padding Whisper uses at the
window edges is replaced by the real audio before the window and by waiting 200 samples for the
last column. ``tests/test_streaming.py`` measures the remaining difference against the HF
extractor.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
WIN_FRAMES = 800                  # 8 s
MEL20_FRAMES = 100                # last 1 s
MEL20_GROUP = 5                   # 50 ms steps


def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    mels = 3.0 * f / 200.0
    hi = f >= 1000.0
    mels[hi] = 15.0 + np.log(f[hi] / 1000.0) * (27.0 / np.log(6.4))
    return mels


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    hi = m >= 15.0
    f[hi] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (m[hi] - 15.0))
    return f


def _mel_filters(n_freq: int = N_FFT // 2 + 1, n_mels: int = N_MELS, sr: int = SR) -> np.ndarray:
    """Whisper's (201, 80) Slaney-normalised triangular mel filterbank, as transformers'
    ``mel_filter_bank(norm="slaney", mel_scale="slaney")`` builds it; numpy only, so the inference
    path does not import transformers (which costs ~190 MB of resident memory)."""
    fft_freqs = np.linspace(0.0, sr / 2, n_freq)
    mel_pts = np.linspace(_hz_to_mel_slaney(np.array([0.0]))[0], _hz_to_mel_slaney(np.array([sr / 2]))[0], n_mels + 2)
    filter_freqs = _mel_to_hz_slaney(mel_pts)
    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]                     # (n_freq, n_mels + 2)
    down = -slopes[:, :-2] / filter_diff[None, :-1]
    up = slopes[:, 2:] / filter_diff[None, 1:]
    fb = np.maximum(0.0, np.minimum(down, up))                                # (n_freq, n_mels)
    enorm = 2.0 / (filter_freqs[2: n_mels + 2] - filter_freqs[:n_mels])
    return (fb * enorm[None, :]).astype(np.float32)


class IncrementalMel:
    """Raw log10-mel columns of the audio stream, one per 10 ms, computed as samples arrive."""

    def __init__(self, sr: int = SR, keep_frames: int = WIN_FRAMES + 8):
        if sr != SR:
            raise ValueError("16 kHz only")
        self.filters = _mel_filters()                                  # (201, 80)
        self.window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # periodic Hann, as HF
        self.buf = np.zeros(0, dtype=np.float32)                       # samples not yet consumed by a frame
        self.n_total = 0                                               # samples received
        self.n_frames = 0                                              # frames emitted; frame i is centred at i*HOP
        self.cols: deque[np.ndarray] = deque(maxlen=keep_frames)       # raw log-mel columns (80,)
        self.ring = np.zeros(WIN_FRAMES * HOP, dtype=np.float32)       # last 8 s of raw samples (for sigma)
        self.ring_pos = 0
        self.ring_filled = 0
        self._pad_left = N_FFT // 2                                    # first frame centred at 0 needs 200 samples before
        self.buf = np.zeros(self._pad_left, dtype=np.float32)          # zeros before the stream start
        W = np.fft.rfft(self.window)
        self.W0, self.W1 = complex(W[0]), complex(W[1])                 # DFT of the window at bins 0 and 1
        self.f0, self.f1 = self.filters[0], self.filters[1]              # mel weights of bins 0 and 1

    def push(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        # sample ring (for the window RMS)
        if n >= len(self.ring):
            self.ring[:] = x[-len(self.ring):]
            self.ring_pos = 0
        else:
            end = self.ring_pos + n
            if end <= len(self.ring):
                self.ring[self.ring_pos:end] = x
            else:
                k = len(self.ring) - self.ring_pos
                self.ring[self.ring_pos:] = x[:k]
                self.ring[: n - k] = x[k:]
            self.ring_pos = end % len(self.ring)
        self.ring_filled = min(self.ring_filled + n, len(self.ring))
        self.n_total += n
        self.buf = np.concatenate([self.buf, x])
        # emit every frame whose 400 samples are available: frame i needs samples [i*HOP-200, i*HOP+200).
        # Whisper subtracts the window mean from the waveform before the STFT. Subtracting a constant
        # c from a Hann-windowed frame changes only DFT bins 0 and 1 (by -c*W0, -c*W1), so each
        # column keeps those two complex bins and the mel power of the other bins; the mean of the
        # window in force at feature time is applied exactly in window_features().
        new = 0
        while len(self.buf) >= N_FFT:
            X = np.fft.rfft(self.buf[:N_FFT] * self.window)             # (201,) complex
            spec = np.abs(X) ** 2
            spec[0] = spec[1] = 0.0
            self.cols.append(((spec @ self.filters).astype(np.float32), complex(X[0]), complex(X[1])))
            self.buf = self.buf[HOP:]
            self.n_frames += 1
            new += 1
        return new

    def stats(self) -> tuple[float, float]:
        """Mean and std of the last 8 s of raw samples (zeros where the stream is shorter)."""
        r = self.ring if self.ring_filled == len(self.ring) else np.concatenate([self.ring[: self.ring_filled], np.zeros(len(self.ring) - self.ring_filled, np.float32)])
        r = r.astype(np.float64)
        return float(r.mean()), float(np.sqrt(r.var() + 1e-7))

    def sigma(self) -> float:
        return self.stats()[1]

    def _pad_column(self, mean: float, sigma: float) -> np.ndarray:
        """Mel power of a frame of the constant value Whisper's normalisation gives to zero padding."""
        c = -mean / sigma
        spec = np.abs(np.fft.rfft(np.full(N_FFT, c, np.float32) * self.window)) ** 2
        return (spec @ self.filters).astype(np.float32)

    def window_features(self) -> np.ndarray:
        """(80, 800) features of the last 8 s, zero-padded at the start while the stream is short:
        log10(max(mel / sigma^2, 1e-10)), clamped at the window max - 8, then (x + 4) / 4, as Whisper."""
        mean, sigma = self.stats()
        k = min(len(self.cols), WIN_FRAMES)
        last = list(self.cols)[-k:]
        if k:
            rest = np.stack([c[0] for c in last], axis=1)                                   # (80, k)
            x0 = np.array([c[1] for c in last]) - mean * self.W0
            x1 = np.array([c[2] for c in last]) - mean * self.W1
            cols = rest + np.outer(self.f0, np.abs(x0) ** 2) + np.outer(self.f1, np.abs(x1) ** 2)
        else:
            cols = np.zeros((N_MELS, 0), np.float32)
        cols = cols / (sigma * sigma)
        if k < WIN_FRAMES:
            pad = np.repeat(self._pad_column(mean, sigma)[:, None], WIN_FRAMES - k, axis=1)
            cols = np.concatenate([pad, cols], axis=1)
        ls = np.log10(np.maximum(cols, 1e-10))
        ls = np.maximum(ls, ls.max() - 8.0)
        return ((ls + 4.0) / 4.0).astype(np.float32)

    def mel20(self) -> np.ndarray:
        """(20, 80): the last 1 s averaged to 50 ms steps, normalised with the 8 s window's statistics."""
        feats = self.window_features()[:, -MEL20_FRAMES:]
        return feats.reshape(N_MELS, MEL20_FRAMES // MEL20_GROUP, MEL20_GROUP).mean(-1).T.astype(np.float32)


@dataclass
class Timings:
    hop_ms: list = field(default_factory=list)
    refresh_ms: list = field(default_factory=list)         # encoder call (on the worker when worker=True)
    mel_ms: list = field(default_factory=list)
    push_ms: list = field(default_factory=list)            # wall time of push() on the calling thread
    decision_ms: list = field(default_factory=list)        # push() wall time on frames that hop at a silence-clock point
    wait_ms: list = field(default_factory=list)            # time spent in wait_refresh / await_refresh
    n_stale_decisions: int = 0                             # silence-clock hops taken before their refresh arrived

    def summary(self) -> dict:
        def q(v):
            v = np.asarray(v) if len(v) else np.zeros(1)
            return dict(p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)), max=float(v.max()), n=len(v))
        return dict(hop=q(self.hop_ms), refresh=q(self.refresh_ms), mel_per_push=q(self.mel_ms), push=q(self.push_ms),
                    decision=q(self.decision_ms), wait=q(self.wait_ms), n_stale_decisions=self.n_stale_decisions)


class StreamingEOT:
    """Two-cadence streaming detector, CPU only. With ``worker=False`` all compute runs on the calling
    thread (measure there); with ``worker=True`` the encoder runs on one background thread."""

    SILENCE_REFRESH_TAUS = (0.0, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5)   # the tau grid the heads were trained on

    def __init__(self, encoder_onnx: str, heads_onnx: str, hop_s: float = 0.1, refresh_s: float = 0.32,
                 threads: int = 1, sr: int = SR, silence_grid: bool = True, vad_stop_secs: float = 0.0,
                 worker: bool = False, lead_s: float = 0.0, wait_s: float = 0.02, max_defer_s: float = 0.08):
        """``vad_stop_secs``: when the ``is_speech`` flag comes from a VAD that reports the stop only after
        that much silence (Pipecat's Silero: 0.2 s), the silence counter starts there instead of at 0, so
        tau measures the time since the acoustic offset as the model was trained.
        ``threads``: ORT intra-op threads of the encoder session (the heads always use one).
        ``worker``: run the encoder on a background thread; ``lead_s``: how long before a silence-clock
        point its refresh is requested; ``wait_s``: how long an async caller waits for a refresh;
        ``max_defer_s``: how long a silence-clock hop may be postponed, frame by frame, for a refresh that
        has not arrived. Default: ``lead_s`` 0 and ``max_defer_s`` 80 ms, i.e. the refresh is requested at
        the point and the decision waits for it (the encoder takes 22-46 ms on an M4), so it is the inline
        decision exactly, up to 80 ms later; ``lead_s`` > 0 with a one-frame ``max_defer_s`` decides at the
        point on a vector computed ``lead_s`` earlier (measured in C2; lower fidelity at 1.2 s)."""
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(encoder_onnx, sess_options=so, providers=["CPUExecutionProvider"])
        self.enc_in = self.enc.get_inputs()[0].name
        self.enc_pooled = self.enc.get_outputs()[-1].name
        self.enc_prob = self.enc.get_outputs()[0].name
        so1 = ort.SessionOptions()                 # the heads hop always runs single-threaded on the caller
        so1.intra_op_num_threads = 1
        so1.inter_op_num_threads = 1
        so1.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.heads = ort.InferenceSession(heads_onnx, sess_options=so1, providers=["CPUExecutionProvider"])
        self.mel = IncrementalMel(sr)
        self.hop_n = int(hop_s * sr)
        self.refresh_n = int(refresh_s * sr)
        self.lead_n = int(lead_s * sr)
        self.max_defer_n = int(max_defer_s * sr)
        self.wait_s = wait_s
        self.sr = sr
        self.timings = Timings()
        self.pooled = np.zeros((1, 384), np.float32)
        self.p_base = 0.0
        self.since_hop = 0
        self.silence_samples = 0
        self.speaking = False
        self.n_samples = 0
        self.last = None
        self.silence_grid = silence_grid
        self._grid_next = 0                     # index of the next silence-clock point
        self._grid_req = -1                     # index of the silence-clock point whose refresh was requested ahead
        self._defer = 0                         # samples a silence-clock hop may still wait for its refresh
        self.vad_stop_n = int(vad_stop_secs * sr)
        # refresh bookkeeping (sample counts; the inline path uses the same fields)
        self._vec_n = 0                         # n_samples at the end of the window the cached vector was computed on
        self._last_req_n = -self.refresh_n      # n_samples at the last refresh request (refresh on the first push)
        self._ex: ThreadPoolExecutor | None = ThreadPoolExecutor(1, thread_name_prefix="eot-encoder") if worker else None
        self._pending: list[tuple] = []         # (n_samples at request, future, mel20, tau_feat, tau) not yet applied, oldest first
        self._vec_snap: tuple | None = None     # (mel20, tau_feat, tau) of the window the cached vector was computed on

    @property
    def worker(self) -> bool:
        return self._ex is not None

    @property
    def pending(self) -> int:
        """Refresh requests not yet applied to the cached vector."""
        return len(self._pending)

    @property
    def stale_s(self) -> float:
        """Age of the cached encoder vector: audio received since the end of its window."""
        return (self.n_samples - self._vec_n) / self.sr

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=False, cancel_futures=True)
            self._ex = None
            self._pending = []

    # --- encoder (slow path) ------------------------------------------------------------------

    def _encode(self, feats: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the encoder on one (1, 80, 800) window. On the worker thread when worker=True."""
        t = time.perf_counter()
        prob, pooled = self.enc.run([self.enc_prob, self.enc_pooled], {self.enc_in: feats})
        self.timings.refresh_ms.append((time.perf_counter() - t) * 1000)
        return pooled.astype(np.float32), float(prob.reshape(-1)[0])

    def _apply(self, n: int, pooled: np.ndarray, p_base: float, snap: tuple | None = None) -> None:
        if n >= self._vec_n:
            self.pooled, self.p_base, self._vec_n, self._vec_snap = pooled, p_base, n, snap

    def _snapshot(self) -> tuple[np.ndarray, tuple]:
        """The current 8 s window and the heads' other inputs taken from the same instant."""
        feats = self.mel.window_features()
        m20 = feats[:, -MEL20_FRAMES:].reshape(N_MELS, MEL20_FRAMES // MEL20_GROUP, MEL20_GROUP).mean(-1).T.astype(np.float32)[None]
        tau = self.silence_samples / self.sr
        tau_feat = np.array([[tau / 2.5, np.log1p(tau), 1.0 if self.speaking else 0.0]], np.float32)
        return feats[None], (m20, tau_feat, tau)

    def refresh(self) -> None:
        """Run the encoder now, on the calling thread, on the current window (the inline path; also
        the warm-up and the offline validation's aligned queries)."""
        self._last_req_n = self.n_samples
        feats, snap = self._snapshot()
        self._apply(self.n_samples, *self._encode(feats), snap)

    def request_refresh(self) -> Future | None:
        """Snapshot the current window (0.6 ms) and refresh: inline without a worker, otherwise on the
        worker thread, returning the future. A request still queued (not started) is superseded."""
        if self._ex is None:
            self.refresh()
            return None
        feats, snap = self._snapshot()
        n = self.n_samples
        self._last_req_n = n
        self._pending = [r for r in self._pending if not r[1].cancel()]
        fut = self._ex.submit(self._encode, feats)
        self._pending.append((n, fut, *snap))
        return fut

    def collect(self) -> int:
        """Apply every finished refresh to the cached vector. Returns how many were applied."""
        if not self._pending:
            return 0
        done = [r for r in self._pending if r[1].done()]
        if done:
            self._pending = [r for r in self._pending if not r[1].done()]
            for n, f, *snap in done:
                if not f.cancelled():
                    self._apply(n, *f.result(), tuple(snap))
        return len(done)

    def wait_refresh(self, timeout_s: float | None = None) -> bool:
        """Block the calling thread up to ``timeout_s`` for the newest pending refresh, then apply what
        has finished. True when no refresh is pending afterwards (the vector is as fresh as requested)."""
        t = time.perf_counter()
        if self._pending:
            try:
                self._pending[-1][1].result(timeout=self.wait_s if timeout_s is None else timeout_s)
            except TimeoutError:
                pass
        self.collect()
        self.timings.wait_ms.append((time.perf_counter() - t) * 1000)
        return not self._pending

    async def await_refresh(self, timeout_s: float | None = None) -> bool:
        """As ``wait_refresh`` for an asyncio caller: the event loop keeps running while waiting."""
        t = time.perf_counter()
        if self._pending:
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(self._pending[-1][1])),
                                       self.wait_s if timeout_s is None else timeout_s)
            except (asyncio.TimeoutError, TimeoutError):
                pass
        self.collect()
        self.timings.wait_ms.append((time.perf_counter() - t) * 1000)
        return not self._pending

    # --- heads (fast path) --------------------------------------------------------------------

    def hop(self, consistent: bool = False) -> dict:
        """Run the heads. ``consistent``: feed the mel20 / tau snapshot taken with the cached vector's
        window (a silence-clock decision), so the three inputs come from one instant as in training;
        otherwise the current mel20 and tau (the free-running hops). ``tau`` in the output is always the
        current silence length; ``tau_vec`` is the snapshot's; ``grid`` marks a silence-clock decision (the
        rule offsets are decided on those only: a free-running hop can land just before a grid point while
        that point's refresh is still in flight, and would decide on the previous vector)."""
        t = time.perf_counter()
        self.collect()
        tau = self.silence_samples / self.sr
        if consistent and self._vec_snap is not None:
            mel20, tau_feat, tau_vec = self._vec_snap
        else:
            tau_feat = np.array([[tau / 2.5, np.log1p(tau), 1.0 if self.speaking else 0.0]], np.float32)
            mel20, tau_vec = self.mel.mel20()[None], tau
        p = self.heads.run(None, {"pooled": self.pooled, "mel20": mel20, "tau_feat": tau_feat})[0][0]
        self.timings.hop_ms.append((time.perf_counter() - t) * 1000)
        self.since_hop = 0
        self.last = dict(t=self.n_samples / self.sr, tau=tau, tau_vec=tau_vec, speaking=self.speaking, p_end=float(p[0]), p_eager=float(p[1]),
                         p_base=self.p_base, stale_s=self.stale_s, pending=len(self._pending), grid=consistent)
        return self.last

    def push(self, x: np.ndarray, is_speech: bool) -> dict | None:
        """Feed audio (float32 in [-1, 1]) with the VAD's verdict for it. Returns the hop output when a hop fell in it."""
        t_push = time.perf_counter()
        x = np.asarray(x, dtype=np.float32)
        t = time.perf_counter()
        self.mel.push(x)
        self.timings.mel_ms.append((time.perf_counter() - t) * 1000)
        n = len(x)
        self.n_samples += n
        if is_speech:
            self.silence_samples = 0
            self.speaking = True
            self._grid_next = 0
            self._grid_req = -1
            self._defer = 0
        else:
            if self.speaking:                    # speech -> silence transition
                self.silence_samples = self.vad_stop_n
            self.silence_samples += n
            self.speaking = False
        self.since_hop += n
        out, decision_frame = None, False
        try:
            # silence clock: refresh (and decide) exactly at the trained tau grid, so the decision at
            # each grid point sees a fresh encoder vector; the free-running timer covers speech
            if self.silence_grid and not self.speaking and self._grid_next < len(self.SILENCE_REFRESH_TAUS):
                k = self._grid_next
                tau_n = int(self.SILENCE_REFRESH_TAUS[k] * self.sr)
                if self._ex is not None and self._grid_req < k and tau_n - self.lead_n <= self.silence_samples < tau_n:
                    self.request_refresh()            # ahead of the point, so the vector is there when it comes
                    self._grid_req = k
                if self.silence_samples >= tau_n:
                    self._grid_next = k + 1
                    while self._grid_next < len(self.SILENCE_REFRESH_TAUS) and \
                            int(self.SILENCE_REFRESH_TAUS[self._grid_next] * self.sr) <= self.silence_samples:
                        self._grid_next += 1          # points already passed (silence started at vad_stop_secs)
                    decision_frame = True
                    if self._ex is None:
                        self.refresh()
                        out = self.hop(consistent=True)
                        return out
                    if self._grid_req < k:            # not anticipated (the transition): request now ...
                        if self.n_samples - self._last_req_n > self.hop_n // 5:    # ... unless one was just requested
                            self.request_refresh()
                        self._grid_req = k
                    self.collect()
                    if self._pending:                 # ... and give it up to max_defer_s before deciding
                        self._defer = max(self.max_defer_n, 1)
                        decision_frame = False
                        return None
                    out = self.hop(consistent=True)
                    return out
            if self._defer:
                self.collect()
                self._defer -= n
                if self._pending and self._defer > 0:
                    return None                       # still waiting, within the allowance
                self._defer = 0
                decision_frame = True
                if self._pending:
                    self.timings.n_stale_decisions += 1
                out = self.hop(consistent=True)
                return out
            if self.n_samples - self._last_req_n >= self.refresh_n:
                self.request_refresh()
            if self.since_hop >= self.hop_n:
                out = self.hop()
            return out
        finally:
            ms = (time.perf_counter() - t_push) * 1000
            self.timings.push_ms.append(ms)
            if decision_frame:
                self.timings.decision_ms.append(ms)

    def mark_silence_start(self) -> None:
        """Apply the speech -> silence transition without audio (a VAD reported the stop before the
        next frame arrived): tau starts at ``vad_stop_secs`` and the silence clock restarts."""
        if self.speaking:
            self.speaking = False
            self.silence_samples = self.vad_stop_n
            self._grid_next = 0
            self._grid_req = -1
            self._defer = 0

    def reset_turn(self) -> None:
        """Call when a turn is committed; the audio history is kept (the next turn's context)."""
        self.silence_samples = 0
        self._grid_next = 0
        self._grid_req = -1
        self._defer = 0
