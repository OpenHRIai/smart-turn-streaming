"""Acoustic offset and per-clip statistics for smart-turn-data clips (Stage A1).

A clip in ``pipecat-ai/smart-turn-data-v3.2`` is one utterance whose speech ends somewhere before
the end of the file. Everything downstream (tau-padded and pre-offset windows, ``eot.windows``)
needs one number per clip: ``t_off``, the time the voice stops. This module computes it with
Silero VAD (ONNX) and records enough side information to audit it (an energy-based offset, the
level of the tail after the offset, the noise floor).

Conventions: mono float32 at 16 kHz; all times in seconds from the start of the clip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SR = 16000
FRAME = 320  # 20 ms at 16 kHz


@dataclass
class ClipStats:
    duration_s: float
    t_on: float | None          # first speech onset (VAD)
    t_off: float | None         # last speech offset (VAD) = the acoustic offset
    n_segments: int
    speech_s: float             # total VAD speech
    leading_s: float | None     # silence before t_on
    trailing_s: float | None    # duration - t_off: room tone already in the clip
    speech_dbfs: float | None   # RMS over VAD speech
    tail_dbfs: float | None     # RMS of the audio after t_off (None if < 50 ms of tail)
    floor_dbfs: float           # 10th percentile of 20 ms frame RMS
    peak_dbfs: float
    t_off_energy: float | None  # energy-based offset, for QA against the VAD one

    def as_row(self) -> dict:
        return asdict(self)


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9)) if len(x) else -120.0


def frame_rms_dbfs(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([dbfs(x)])
    f = x[: n * frame].reshape(n, frame).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(f * f, axis=1)) + 1e-9)


def energy_offset(x: np.ndarray, sr: int = SR, floor_db: float | None = None,
                  margin_db: float = 12.0, min_db: float = -50.0) -> float | None:
    """End of the last 20 ms frame louder than max(floor + margin, min_db)."""
    db = frame_rms_dbfs(x)
    if floor_db is None:
        floor_db = float(np.percentile(db, 10))
    loud = np.where(db > max(floor_db + margin_db, min_db))[0]
    if len(loud) == 0:
        return None
    return float((loud[-1] + 1) * FRAME / sr)


class OffsetDetector:
    """Silero VAD (ONNX, CPU) wrapped for one-clip statistics. One instance per process."""

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 100, min_speech_ms: int = 100,
                 speech_pad_ms: int = 30):
        from silero_vad import get_speech_timestamps, load_silero_vad
        self._model = load_silero_vad(onnx=True)
        self._gst = get_speech_timestamps
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.speech_pad_ms = speech_pad_ms

    def segments(self, x: np.ndarray, sr: int = SR) -> list[tuple[float, float]]:
        import torch
        if sr != SR:
            raise ValueError(f"expected {SR} Hz, got {sr}")
        ts = self._gst(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), self._model,
                       threshold=self.threshold, sampling_rate=SR,
                       min_speech_duration_ms=self.min_speech_ms,
                       min_silence_duration_ms=self.min_silence_ms,
                       speech_pad_ms=self.speech_pad_ms, return_seconds=True)
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        return [(float(s["start"]), float(s["end"])) for s in ts]

    def stats(self, x: np.ndarray, sr: int = SR) -> ClipStats:
        x = np.asarray(x, dtype=np.float32)
        dur = len(x) / sr
        segs = self.segments(x, sr)
        db = frame_rms_dbfs(x)
        floor = float(np.percentile(db, 10))
        peak = float(20 * np.log10(np.max(np.abs(x)) + 1e-9)) if len(x) else -120.0
        if not segs:
            return ClipStats(dur, None, None, 0, 0.0, None, None, None, None, floor, peak,
                             energy_offset(x, sr, floor))
        t_on, t_off = segs[0][0], min(segs[-1][1], dur)
        speech = np.concatenate([x[int(a * sr): int(b * sr)] for a, b in segs]) if segs else x[:0]
        tail = x[int(round(t_off * sr)):]
        return ClipStats(
            duration_s=round(dur, 4), t_on=round(t_on, 3), t_off=round(t_off, 3), n_segments=len(segs),
            speech_s=round(float(sum(b - a for a, b in segs)), 3),
            leading_s=round(t_on, 3), trailing_s=round(dur - t_off, 3),
            speech_dbfs=round(dbfs(speech), 2) if len(speech) else None,
            tail_dbfs=round(dbfs(tail), 2) if len(tail) >= int(0.05 * sr) else None,
            floor_dbfs=round(floor, 2), peak_dbfs=round(peak, 2),
            t_off_energy=energy_offset(x, sr, floor),
        )
