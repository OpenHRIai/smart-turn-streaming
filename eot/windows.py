"""Training windows cut on the fly from a clip and its acoustic offset (Stage A1).

Nothing here is stored as audio. Given a clip ``x`` (16 kHz mono) and its offset ``t_off`` from
``eot.offsets``, three kinds of 8 s windows are produced:

* ``tau``: ends ``tau`` seconds *after* the offset. Where the clip is too short the tail is
  extended with the clip's own room tone (the audio after ``t_off``, tiled with crossfades), so
  the model sees silence that sounds like that recording, not digital zeros. Label ``p_end`` is
  the clip's label (complete stays complete at every tau, incomplete stays incomplete).
* ``pre``: ends ``delta`` seconds *before* the offset, while the person is still talking.
  ``p_end`` is 0; ``p_eager`` is 1 when the clip is complete and ``delta <= EAGER_HORIZON_S``.
* ``mid``: ends at a random point well inside the speech. Both labels 0.

Leading padding (window longer than the available audio) is zeros at the start, as the upstream
trainer does (``truncate_audio_to_last_n_seconds``).
"""

from __future__ import annotations

import numpy as np

SR = 16000
WIN_S = 8.0
TAUS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5)
DELTAS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
EAGER_HORIZON_S = 0.4
MIN_SPEECH_BEFORE_S = 0.5   # a pre-offset window must still hold this much speech
MIN_TAIL_S = 0.05           # tail shorter than this: synthesise room tone at the tail's level
TILE_S = 0.5                # room-tone tile length
FADE_MS = 10


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def room_tone(x: np.ndarray, sr: int, t_off: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of room tone made from the clip's tail after ``t_off``."""
    tail = x[int(round(t_off * sr)):]
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(tail) < int(MIN_TAIL_S * sr):
        # No room tone in the clip (speech runs to the file end, typical of TTS clips). Use the
        # level of the clip's quietest 20 ms frames, capped at -45 dBFS so a clip that is speech
        # throughout does not get a loud tail.
        nf = len(x) // 320
        if nf >= 5:
            fr = x[: nf * 320].reshape(nf, 320).astype(np.float64)
            level = float(np.percentile(np.sqrt(np.mean(fr * fr, axis=1)), 5))
        else:
            level = _rms(x)
        level = float(np.clip(level, 10 ** (-70 / 20), 10 ** (-45 / 20)))
        return (rng.standard_normal(n) * level).astype(np.float32)
    fade = int(FADE_MS * sr / 1000)
    tile = min(len(tail), int(TILE_S * sr))
    out = np.zeros(n + tile, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32) if fade > 0 else None
    pos = 0
    while pos < n:
        start = int(rng.integers(0, len(tail) - tile + 1))
        seg = tail[start: start + tile].copy()
        if rng.random() < 0.5:
            seg = seg[::-1]   # reversed tiles break any periodicity in the tile boundary
        if pos > 0 and ramp is not None and tile > fade:
            seg[:fade] *= ramp
            out[pos - fade: pos] *= ramp[::-1]
            out[pos - fade: pos - fade + tile] += seg
            pos += tile - fade
        else:
            out[pos: pos + tile] = seg
            pos += tile
    return out[:n]


def cut_window(x: np.ndarray, sr: int, end_s: float, t_off: float, rng: np.random.Generator,
               win_s: float = WIN_S) -> np.ndarray:
    """Exactly ``win_s`` seconds ending at ``end_s``; zeros before the clip, room tone after it."""
    x = np.asarray(x, dtype=np.float32)
    n = int(round(win_s * sr))
    end = int(round(end_s * sr))
    start = end - n
    L = len(x)
    out = np.zeros(n, dtype=np.float32)
    s0, s1 = max(start, 0), min(end, L)
    if s1 > s0:
        out[s0 - start: s1 - start] = x[s0:s1]
    if end > L:
        pad_from = max(L, start)
        out[pad_from - start:] = room_tone(x, sr, t_off, end - pad_from, rng)
    return out


def variant(x: np.ndarray, sr: int, t_off: float, kind: str, value: float,
            rng: np.random.Generator, win_s: float = WIN_S) -> np.ndarray:
    if kind == "tau":
        end_s = t_off + value
    elif kind == "pre":
        end_s = t_off - value
    elif kind == "mid":
        end_s = value
    else:
        raise ValueError(kind)
    return cut_window(x, sr, end_s, t_off, rng, win_s)


def enumerate_items(t_on: float, t_off: float, label: int, *, taus=TAUS, deltas=DELTAS,
                    n_mid: int = 1, rng: np.random.Generator | None = None) -> list[dict]:
    """The training items one clip yields. ``p_eager`` is None where it is undefined (silence)."""
    items = [dict(kind="tau", value=float(t), p_end=int(label), p_eager=None) for t in taus]
    for d in deltas:
        if t_off - d >= t_on + MIN_SPEECH_BEFORE_S:
            items.append(dict(kind="pre", value=float(d), p_end=0,
                              p_eager=int(label == 1 and d <= EAGER_HORIZON_S)))
    lo, hi = t_on + 1.0, t_off - max(deltas) - 0.2
    if n_mid and hi > lo:
        r = rng if rng is not None else np.random.default_rng(0)
        for _ in range(n_mid):
            items.append(dict(kind="mid", value=float(r.uniform(lo, hi)), p_end=0, p_eager=0))
    return items
