"""Decision-rule evaluation for silence-aware end-of-turn models (the Phase 0 protocol, generic).

Given, per labelled silence, the model's P(complete) at a few silence offsets, simulate the
deployed rule: query at each offset while the silence is still going on, commit COMPLETE at the
first offset with p >= theta(offset); a complete with no commit ends on the ``timeout``. Fit the
theta table on one set (dev) under a false-alarm constraint and score it on another (test).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

OFFSETS = (0.2, 0.5, 1.2)      # the taus we query at (items exist at these taus)
TIMEOUT_S = 3.0
GRID = np.round(np.arange(0.0, 1.0001, 0.02), 2)


@dataclass
class RuleResult:
    theta: tuple
    recall: float
    far: float
    p50: float
    p90: float
    mean: float
    far_by_bin: dict          # pause-length bins of incompletes
    commits_at: dict          # offset -> (completes, false alarms)


def _lat(commit_idx: np.ndarray, offsets) -> np.ndarray:
    """commit_idx: index of the committing offset per complete, -1 = miss."""
    lat = np.full(len(commit_idx), TIMEOUT_S)
    ok = commit_idx >= 0
    lat[ok] = np.asarray(offsets)[commit_idx[ok]]
    return lat


def simulate(p: np.ndarray, avail: np.ndarray, label: np.ndarray, gap: np.ndarray, theta, offsets=OFFSETS) -> RuleResult:
    """p: (N, K) scores (nan where no query); avail: (N, K) bool (silence still going on);
    label: (N,) 1 complete; gap: (N,) incomplete pause length (nan for completes)."""
    th = np.asarray(theta)[None, :]
    hit = avail & (np.nan_to_num(p, nan=-1.0) >= th)          # (N, K)
    first = np.where(hit.any(1), hit.argmax(1), -1)
    comp, inc = label == 1, label == 0
    recall = float((first[comp] >= 0).mean()) if comp.any() else float("nan")
    fa = first[inc] >= 0
    far = float(fa.mean()) if inc.any() else float("nan")
    lat = _lat(first[comp], offsets)
    bins = {"<0.5": gap[inc] < 0.5, "0.5-1.5": (gap[inc] >= 0.5) & (gap[inc] < 1.5), ">=1.5": gap[inc] >= 1.5}
    far_by_bin = {k: (float(fa[m].mean()) if m.any() else float("nan")) for k, m in bins.items()}
    commits = {offsets[k]: (int((first[comp] == k).sum()), int((first[inc] == k).sum())) for k in range(len(offsets))}
    return RuleResult(tuple(float(t) for t in theta), recall, far, float(np.percentile(lat, 50)),
                      float(np.percentile(lat, 90)), float(lat.mean()), far_by_bin, commits)


def fit(p, avail, label, gap, target_far: float, objective: str = "min_latency", offsets=OFFSETS, grid=GRID) -> RuleResult:
    """Exhaustive grid over theta per offset subject to FAR <= target; objective 'min_latency'
    (mean latency incl. timeouts) or 'max_recall'."""
    best = None
    inc = label == 0
    comp = label == 1
    K = len(offsets)
    # precompute hit matrices per offset and theta
    hits = [np.stack([avail[:, k] & (np.nan_to_num(p[:, k], nan=-1.0) >= t) for t in grid]) for k in range(K)]  # K x (G, N)
    for combo in itertools.product(range(len(grid)), repeat=K):
        h = np.stack([hits[k][combo[k]] for k in range(K)], axis=1)       # (N, K)
        first = np.where(h.any(1), h.argmax(1), -1)
        far = (first[inc] >= 0).mean()
        if far > target_far:
            continue
        rec = (first[comp] >= 0).mean()
        lat = _lat(first[comp], offsets).mean()
        key = (-rec, lat) if objective == "max_recall" else (lat, -rec)
        if best is None or key < best[0]:
            best = (key, tuple(grid[i] for i in combo))
    if best is None:
        return simulate(p, avail, label, gap, tuple([1.01] * K), offsets)
    return simulate(p, avail, label, gap, best[1], offsets)


def one_shot(p, avail, label, gap, theta: float = 0.5, offsets=OFFSETS) -> RuleResult:
    """The stock rule: one query at the first offset, then the timeout."""
    th = [theta] + [1.01] * (len(offsets) - 1)
    return simulate(p, avail, label, gap, th, offsets)


def table(rows: list[tuple[str, RuleResult]]) -> str:
    out = ["| rule | theta | recall | FAR | p50 | p90 | mean | FAR <0.5 / 0.5-1.5 / >=1.5 s |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        th = " / ".join(f"{t:.2f}" if t <= 1 else "-" for t in r.theta)
        fb = " / ".join(f"{r.far_by_bin[k]:.2f}" for k in ("<0.5", "0.5-1.5", ">=1.5"))
        out.append(f"| {name} | {th} | {r.recall:.2f} | {r.far:.3f} | {r.p50:.2f} s | {r.p90:.2f} s | {r.mean:.2f} s | {fb} |")
    return "\n".join(out)
