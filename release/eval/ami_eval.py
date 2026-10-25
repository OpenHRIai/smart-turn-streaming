#!/usr/bin/env python3
"""Reproduce the AMI end-of-turn evaluation of the Smart Turn v2 streaming heads (release R1).

    python ami_eval.py --audio-dir ./ami_audio            # downloads the 34 test (and dev) headset files, ~3 GB
    python ami_eval.py --audio-dir ./ami_audio --splits test --no-fit    # test only, shipped thresholds, ~5 min

What it computes, from the public AMI Meeting Corpus audio and the labelled-silence manifests shipped
next to this script (``ami_dev.csv``, ``ami_test.csv``; CC BY 4.0, see ../ATTRIBUTION.md):

1. For every labelled silence (a VAD stop of speaker A at time ``t``) and every silence offset tau in
   ``--taus`` the 8 s window of A's headset audio ending at ``t + tau``. For complete turns the audio after
   the next speaker's onset (``t + b_onset_s``) is replaced by A's own room tone, because a deployed
   microphone never hears the agent's reply; for incomplete turns a window exists only while the pause is
   still going on (``tau < gap_s``).
2. Whisper log-mel features (80 x 800, as ``WhisperFeatureExtractor(chunk_length=8)`` with
   ``do_normalize=True``), the frozen Smart Turn v3.2 encoder with its pooled 384-d vector exposed, and
   the trained heads (pooled vector + last 1 s of mel + tau) -> P(complete). The frozen model's own
   probability is kept as the baseline.
3. AUC (complete vs incomplete) per tau for the heads and the frozen model, and the deployed decision
   rule: query at 0.2 / 0.5 / 1.2 s of silence while the pause lasts, commit COMPLETE at the first offset
   with P >= theta(offset); a complete with no commit is charged the 3 s timeout. Thresholds are the
   shipped tables (``../rules/theta_tables.csv``) and, unless ``--no-fit``, re-fitted on the dev split under
   false-alarm budgets of 5 / 10 / 15 % (exhaustive grid, min mean latency), then scored on test.

Dependencies: numpy, onnxruntime, soundfile. CPU only; about 16k encoder windows for dev + test at the
three rule offsets (~7 min single-threaded on an Apple M4). Scores are cached in ``--cache`` so the
tables can be re-made without re-running the models.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SR = 16000
WIN_S = 8.0
N_FFT, HOP, N_MELS, WIN_FRAMES = 400, 160, 80, 800
MEL20_FRAMES, MEL20_GROUP = 100, 5
OFFSETS = (0.2, 0.5, 1.2)
TAU_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5)
TIMEOUT_S = 3.0
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
HERE = Path(__file__).resolve().parent


# --- Whisper log-mel in numpy ---------------------------------------------------------------------

def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    m = 3.0 * f / 200.0
    hi = f >= 1000.0
    m[hi] = 15.0 + np.log(f[hi] / 1000.0) * (27.0 / np.log(6.4))
    return m


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    hi = m >= 15.0
    f[hi] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (m[hi] - 15.0))
    return f


def mel_filters(n_freq: int = N_FFT // 2 + 1, n_mels: int = N_MELS, sr: int = SR) -> np.ndarray:
    """(201, 80) Slaney-normalised triangular filterbank, identical to transformers' ``mel_filter_bank``."""
    fft_freqs = np.linspace(0.0, sr / 2, n_freq)
    mel_pts = np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sr / 2]))[0], n_mels + 2)
    f = _mel_to_hz(mel_pts)
    d = np.diff(f)
    slopes = f[None, :] - fft_freqs[:, None]
    fb = np.maximum(0.0, np.minimum(-slopes[:, :-2] / d[None, :-1], slopes[:, 2:] / d[None, 1:]))
    enorm = 2.0 / (f[2: n_mels + 2] - f[:n_mels])
    return (fb * enorm[None, :]).astype(np.float32)


_FILTERS = mel_filters()
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def whisper_logmel(x: np.ndarray) -> np.ndarray:
    """(80, 800) features of exactly 8 s of 16 kHz audio, as WhisperFeatureExtractor(chunk_length=8)
    with padding='max_length', truncation=True, do_normalize=True gives them."""
    x = np.asarray(x, dtype=np.float64)
    assert len(x) == int(WIN_S * SR), len(x)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)                       # do_normalize
    pad = N_FFT // 2
    xp = np.concatenate([x[1: pad + 1][::-1], x, x[-pad - 1: -1][::-1]])   # reflect padding, centre=True
    n_frames = 1 + (len(xp) - N_FFT) // HOP                               # 801; the last one is dropped
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * _WINDOW
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2                       # (801, 201)
    mel = spec[:-1] @ _FILTERS                                            # (800, 80)
    ls = np.log10(np.maximum(mel, 1e-10))
    ls = np.maximum(ls, ls.max() - 8.0)
    return ((ls + 4.0) / 4.0).T.astype(np.float32)


def mel20(feats: np.ndarray) -> np.ndarray:
    """(20, 80): the last second of the features averaged to 50 ms steps (the heads' fresh-mel input)."""
    return feats[:, -MEL20_FRAMES:].reshape(N_MELS, MEL20_FRAMES // MEL20_GROUP, MEL20_GROUP).mean(-1).T.astype(np.float32)


# --- windows ----------------------------------------------------------------------------------------

def room_tone(x: np.ndarray, sr: int, t_off: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of room tone tiled from the audio after ``t_off`` (0.5 s tiles, random order, some
    reversed, 10 ms crossfades), as the training windows were built."""
    tail = x[int(round(t_off * sr)):]
    if n <= 0:
        return np.zeros(0, np.float32)
    if len(tail) < int(0.05 * sr):
        nf = len(x) // 320
        if nf >= 5:
            fr = x[: nf * 320].reshape(nf, 320).astype(np.float64)
            level = float(np.percentile(np.sqrt(np.mean(fr * fr, axis=1)), 5))
        else:
            level = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
        level = float(np.clip(level, 10 ** (-70 / 20), 10 ** (-45 / 20)))
        return (rng.standard_normal(n) * level).astype(np.float32)
    fade = int(10 * sr / 1000)
    tile = min(len(tail), int(0.5 * sr))
    out = np.zeros(n + tile, np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, len(tail) - tile + 1))
        seg = tail[start: start + tile].copy()
        if rng.random() < 0.5:
            seg = seg[::-1]
        if pos > 0 and tile > fade:
            seg[:fade] *= ramp
            out[pos - fade: pos] *= ramp[::-1]
            out[pos - fade: pos - fade + tile] += seg
            pos += tile - fade
        else:
            out[pos: pos + tile] = seg
            pos += tile
    return out[:n]


def cut_window(x: np.ndarray, end_s: float, t_off: float, cut_s: float | None, rng: np.random.Generator) -> np.ndarray:
    """8 s ending at ``end_s``: zeros before the recording, A's room tone after ``cut_s`` (the next
    speaker's onset) or after the end of the recording."""
    if cut_s is not None:
        x = x[: int(round(cut_s * SR))]
    n = int(WIN_S * SR)
    end = int(round(end_s * SR))
    start = end - n
    out = np.zeros(n, np.float32)
    s0, s1 = max(start, 0), min(end, len(x))
    if s1 > s0:
        out[s0 - start: s1 - start] = x[s0:s1]
    if end > len(x):
        pad_from = max(len(x), start)
        out[pad_from - start:] = room_tone(x, SR, t_off, end - pad_from, rng)
    return out


# --- data -------------------------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(meeting=r["meeting"], speaker=r["speaker"], channel=int(r["channel"]), t=float(r["t"]),
                             label=int(r["label"] == "complete"),
                             gap_s=float(r["gap_s"]) if r["gap_s"] else math.nan,
                             b_onset_s=float(r["b_onset_s"]) if r["b_onset_s"] else math.nan))
    return rows


def headset_path(audio_dir: Path, meeting: str, channel: int) -> Path:
    return audio_dir / meeting / f"{meeting}.Headset-{channel}.wav"


def fetch_audio(audio_dir: Path, rows: list[dict], retries: int = 4) -> None:
    need = sorted({(r["meeting"], r["channel"]) for r in rows})
    missing = [(m, c) for m, c in need if not headset_path(audio_dir, m, c).exists()]
    if not missing:
        return
    print(f"downloading {len(missing)} headset files (~40 MB each) from {MIRROR} into {audio_dir}", flush=True)
    for m, c in missing:
        url = f"{MIRROR}/{m}/audio/{m}.Headset-{c}.wav"
        dest = headset_path(audio_dir, m, c)
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                tmp = dest.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                    size = int(r.headers.get("Content-Length", "0"))
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                if size and tmp.stat().st_size != size:
                    raise OSError(f"size mismatch {tmp.stat().st_size} != {size}")
                tmp.replace(dest)
                print(f"  got {dest.name}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                print(f"  retry {dest.name}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))


# --- models -----------------------------------------------------------------------------------------

class Models:
    def __init__(self, encoder: Path, heads: Path, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(str(encoder), sess_options=so, providers=["CPUExecutionProvider"])
        self.heads = ort.InferenceSession(str(heads), sess_options=so, providers=["CPUExecutionProvider"])
        self.enc_in = self.enc.get_inputs()[0].name
        outs = [o.name for o in self.enc.get_outputs()]
        self.enc_prob, self.enc_pooled = outs[0], outs[-1]

    def score(self, feats: np.ndarray, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """feats (B, 80, 800), taus (B,) -> (p_end of the heads (B,), p of the frozen model (B,))."""
        prob, pooled = self.enc.run([self.enc_prob, self.enc_pooled], {self.enc_in: feats.astype(np.float32)})
        m20 = np.stack([mel20(f) for f in feats])
        tau_feat = np.stack([taus / 2.5, np.log1p(taus), np.zeros_like(taus)], axis=1).astype(np.float32)
        p = self.heads.run(None, {"pooled": pooled.astype(np.float32), "mel20": m20, "tau_feat": tau_feat})[0]
        return p[:, 0].astype(np.float64), prob.reshape(-1).astype(np.float64)


def score_split(rows: list[dict], taus: tuple, models: Models, audio_dir: Path, seed: int, batch: int = 16, tag: str = "") -> dict:
    """Per silence and tau: P(complete) of the heads and of the frozen model (nan where no window)."""
    rng = np.random.default_rng(seed)
    P = np.full((len(rows), len(taus)), np.nan)
    PB = np.full((len(rows), len(taus)), np.nan)
    import soundfile as sf
    by_file: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        by_file.setdefault((r["meeting"], r["channel"]), []).append(i)
    jobs = []        # (row index, tau index, window)
    t0, done, total = time.time(), 0, sum(len(v) for v in by_file.values())

    def flush():
        if not jobs:
            return
        feats = np.stack([whisper_logmel(w) for _, _, w in jobs])
        p, pb = models.score(feats, np.array([taus[k] for _, k, _ in jobs], np.float64))
        for (i, k, _), a, b in zip(jobs, p, pb):
            P[i, k], PB[i, k] = a, b
        jobs.clear()

    for (meeting, channel), idx in by_file.items():
        x, sr = sf.read(headset_path(audio_dir, meeting, channel), dtype="float32")
        if sr != SR:
            raise ValueError(f"{meeting} headset {channel}: {sr} Hz, expected 16 kHz")
        if x.ndim > 1:
            x = x[:, 0]
        for i in idx:
            r = rows[i]
            for k, tau in enumerate(taus):
                if not r["label"] and not (tau < r["gap_s"]):
                    continue                                   # the pause ended before this offset
                cut = r["t"] + r["b_onset_s"] if (r["label"] and not math.isnan(r["b_onset_s"]) and tau > r["b_onset_s"]) else None
                jobs.append((i, k, cut_window(x, r["t"] + tau, r["t"], cut, rng)))
                if len(jobs) >= batch:
                    flush()
        done += len(idx)
        print(f"  {tag}{done}/{total} silences, {time.time() - t0:.0f} s", end="\r", file=sys.stderr, flush=True)
    flush()
    print(file=sys.stderr)
    return dict(p=P, pb=PB, label=np.array([r["label"] for r in rows]), gap=np.array([r["gap_s"] for r in rows]), taus=np.array(taus))


# --- metrics ----------------------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC by the rank statistic (ties counted half)."""
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def simulate(p, avail, label, gap, theta, offsets=OFFSETS) -> dict:
    th = np.asarray(theta)[None, :]
    hit = avail & (np.nan_to_num(p, nan=-1.0) >= th)
    first = np.where(hit.any(1), hit.argmax(1), -1)
    comp, inc = label == 1, label == 0
    lat = np.full(comp.sum(), TIMEOUT_S)
    ok = first[comp] >= 0
    lat[ok] = np.asarray(offsets)[first[comp][ok]]
    fa = first[inc] >= 0
    bins = {"<0.5": gap[inc] < 0.5, "0.5-1.5": (gap[inc] >= 0.5) & (gap[inc] < 1.5), ">=1.5": gap[inc] >= 1.5}
    return dict(theta=tuple(float(t) for t in theta), recall=float(ok.mean()), far=float(fa.mean()),
                p50=float(np.percentile(lat, 50)), p90=float(np.percentile(lat, 90)), mean=float(lat.mean()),
                far_by_bin={k: (float(fa[m].mean()) if m.any() else math.nan) for k, m in bins.items()},
                n_complete=int(comp.sum()), n_incomplete=int(inc.sum()))


def fit(p, avail, label, gap, target_far: float, offsets=OFFSETS, grid=np.round(np.arange(0.0, 1.0001, 0.02), 2)) -> dict:
    """Exhaustive grid over theta per offset with FAR <= target, minimising the mean latency (misses charged 3 s)."""
    inc, comp = label == 0, label == 1
    K = len(offsets)
    hits = [np.stack([avail[:, k] & (np.nan_to_num(p[:, k], nan=-1.0) >= t) for t in grid]) for k in range(K)]
    best = None
    for combo in itertools.product(range(len(grid)), repeat=K):
        h = np.stack([hits[k][combo[k]] for k in range(K)], axis=1)
        first = np.where(h.any(1), h.argmax(1), -1)
        if (first[inc] >= 0).mean() > target_far:
            continue
        rec = (first[comp] >= 0).mean()
        lat = np.where(first[comp] >= 0, np.asarray(offsets)[np.maximum(first[comp], 0)], TIMEOUT_S).mean()
        key = (lat, -rec)
        if best is None or key < best[0]:
            best = (key, tuple(grid[i] for i in combo))
    theta = best[1] if best else tuple([1.01] * K)
    return simulate(p, avail, label, gap, theta, offsets)


def rule_inputs(sc: dict, offsets=OFFSETS):
    cols = [int(np.where(np.isclose(sc["taus"], o))[0][0]) for o in offsets]
    p, pb = sc["p"][:, cols], sc["pb"][:, cols]
    label, gap = sc["label"], sc["gap"]
    avail = (label[:, None] == 1) | (np.nan_to_num(gap, nan=np.inf)[:, None] > np.asarray(offsets)[None, :])
    avail &= ~np.isnan(p)
    # the rule is scored on silences that have a decision point before the resumption
    keep = avail.any(1)
    return p[keep], pb[keep], avail[keep], label[keep], gap[keep]


def rule_row(name: str, r: dict) -> str:
    th = " / ".join(f"{t:.2f}" if t <= 1 else "-" for t in r["theta"])
    fb = " / ".join(f"{r['far_by_bin'][k]:.2f}" for k in ("<0.5", "0.5-1.5", ">=1.5"))
    return f"| {name} | {th} | {r['recall']:.2f} | {r['far']:.3f} | {r['p50']:.2f} s | {r['p90']:.2f} s | {r['mean']:.2f} s | {fb} |"


def read_theta_tables(path: Path) -> list[tuple[str, tuple]]:
    out = []
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == "heads":
                out.append((f"shipped table, {float(r['far_budget']):.0%} budget", tuple(float(t) for t in json.loads(r["theta"].replace("(", "[").replace(")", "]")))))
    return out


# --- main -------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", default="./ami_audio", help="AMI headset audio, <meeting>/<meeting>.Headset-<n>.wav (downloaded when missing)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--manifest-dir", default=str(HERE), help="directory with ami_dev.csv / ami_test.csv")
    ap.add_argument("--encoder", default=str(HERE.parent / "smart-turn-v3.2-encoder-pooled.onnx"))
    ap.add_argument("--heads", default=str(HERE.parent / "eot-heads-v32.onnx"))
    ap.add_argument("--theta-tables", default=str(HERE.parent / "rules" / "theta_tables.csv"))
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    ap.add_argument("--taus", nargs="*", default=None, help="silence offsets to score (default: the rule offsets 0.2 0.5 1.2; 'all' = the trained grid)")
    ap.add_argument("--budgets", nargs="*", type=float, default=[0.05, 0.10, 0.15])
    ap.add_argument("--no-fit", action="store_true", help="skip the threshold fit on dev (score the shipped tables only)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="./ami_eval_scores.npz")
    ap.add_argument("--out", default="./ami_eval.md")
    a = ap.parse_args()

    taus = TAU_GRID if (a.taus and a.taus[0] == "all") else tuple(float(t) for t in (a.taus or OFFSETS))
    for o in OFFSETS:
        if not any(np.isclose(o, taus)):
            sys.exit(f"--taus must include the rule offsets {OFFSETS}")
    audio_dir = Path(a.audio_dir)
    rows = {sp: read_manifest(Path(a.manifest_dir) / f"ami_{sp}.csv") for sp in a.splits}
    if not a.no_download:
        fetch_audio(audio_dir, [r for rs in rows.values() for r in rs])

    cache = dict(np.load(a.cache, allow_pickle=True)) if os.path.exists(a.cache) else {}
    scores = {}
    models = None
    for sp in a.splits:
        key = f"{sp}:{','.join(f'{t:g}' for t in taus)}:{a.seed}"
        if key in cache:
            scores[sp] = cache[key].item()
            print(f"{sp}: scores from {a.cache}", file=sys.stderr)
            continue
        if models is None:
            models = Models(Path(a.encoder), Path(a.heads), a.threads)
        print(f"{sp}: scoring {len(rows[sp])} silences at tau = {taus}", file=sys.stderr, flush=True)
        scores[sp] = score_split(rows[sp], taus, models, audio_dir, a.seed, tag=f"{sp} ")
        cache[key] = np.array(scores[sp], dtype=object)
        np.savez(a.cache, **cache)

    lines = [f"# AMI end-of-turn evaluation ({', '.join(a.splits)}; thresholds {'shipped' if a.no_fit else 'shipped and re-fitted on dev'})", ""]
    for sp in a.splits:
        sc = scores[sp]
        lines += [f"## AMI {sp}: AUC of P(complete) per silence offset", "",
                  "| tau (s) | n complete / incomplete | AUC heads | AUC frozen Smart Turn v3.2 | mean P on incompletes: heads / frozen |", "|---|---|---|---|---|"]
        for k, tau in enumerate(sc["taus"]):
            ok = ~np.isnan(sc["p"][:, k])
            y, p, pb = sc["label"][ok], sc["p"][ok, k], sc["pb"][ok, k]
            lines.append(f"| {tau:g} | {int((y == 1).sum())} / {int((y == 0).sum())} | {auc(y, p):.3f} | {auc(y, pb):.3f} | "
                         f"{p[y == 0].mean():.3f} / {pb[y == 0].mean():.3f} |")
        lines.append("")
    test_sp = "test" if "test" in scores else a.splits[-1]
    P, PB, A, L, G = rule_inputs(scores[test_sp])
    hdr = ["| rule | theta at 0.2 / 0.5 / 1.2 s | recall | FAR | p50 | p90 | mean latency | FAR by pause length <0.5 / 0.5-1.5 / >=1.5 s |",
           "|---|---|---|---|---|---|---|---|"]
    lines += [f"## Decision rule on AMI {test_sp} ({int((L == 1).sum())} complete / {int((L == 0).sum())} incomplete silences with a decision point; "
              f"misses charged {TIMEOUT_S:g} s)", "", "**Heads (this release)**", ""] + hdr
    lines.append(rule_row("one-shot at the VAD stop, theta 0.5 (stock rule)", simulate(P, A, L, G, (0.5, 1.01, 1.01))))
    for name, th in read_theta_tables(Path(a.theta_tables)):
        lines.append(rule_row(name, simulate(P, A, L, G, th)))
    fitted = {}
    if not a.no_fit and "dev" in scores:
        Pd, PBd, Ad, Ld, Gd = rule_inputs(scores["dev"])
        for b in a.budgets:
            t_fit = time.time()
            fitted[b] = (fit(Pd, Ad, Ld, Gd, b), fit(PBd, Ad, Ld, Gd, b))
            print(f"fit at {b:.0%}: {time.time() - t_fit:.0f} s", file=sys.stderr)
            lines.append(rule_row(f"re-fitted on dev, {b:.0%} budget", simulate(P, A, L, G, fitted[b][0]["theta"])))
    lines += ["", "**Frozen Smart Turn v3.2 (baseline, same protocol)**", ""] + hdr
    lines.append(rule_row("one-shot at the VAD stop, theta 0.5 (stock rule)", simulate(PB, A, L, G, (0.5, 1.01, 1.01))))
    for b, (_, rb) in fitted.items():
        lines.append(rule_row(f"re-fitted on dev, {b:.0%} budget", simulate(PB, A, L, G, rb["theta"])))
    lines += ["", "**No model**", ""] + hdr
    lines.append(rule_row("silence only, commit at 1.2 s", simulate(np.ones_like(P), A, L, G, (1.01, 1.01, 0.0))))
    lines += ["", f"Scores cached in `{a.cache}`; audio from {MIRROR} (AMI Meeting Corpus, CC BY 4.0)."]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
