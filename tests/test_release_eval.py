"""The self-contained release evaluation script (release/eval/ami_eval.py) must build the same features
and windows as the training / offline pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eot.streaming import SR
from eot.windows import cut_window as cut_window_ref

spec = importlib.util.spec_from_file_location("ami_eval", Path(__file__).resolve().parents[1] / "release" / "eval" / "ami_eval.py")
ami_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ami_eval)


def _speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.7 * t)
    x = np.zeros_like(t)
    for h in range(1, 12):
        x += np.sin(2 * np.pi * h * np.cumsum(f0) / SR) / h
    env = (np.sin(2 * np.pi * 3 * t) > -0.2).astype(float)
    return (0.2 * x * env + rng.standard_normal(len(t)) * 1e-3).astype(np.float32)


def test_logmel_matches_hf_extractor():
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8)
    x = _speechlike(8.0)
    ref = fe(x, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=8 * SR, truncation=True,
             do_normalize=True).input_features[0]
    ours = ami_eval.whisper_logmel(x)
    assert ours.shape == (80, 800)
    assert np.abs(ours - ref).max() < 1e-3, np.abs(ours - ref).max()
    # a window with zeros at the start (recording shorter than 8 s), as cut_window pads
    x2 = np.concatenate([np.zeros(3 * SR, np.float32), _speechlike(5.0, 1)])
    ref2 = fe(x2, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=8 * SR, truncation=True,
              do_normalize=True).input_features[0]
    assert np.abs(ami_eval.whisper_logmel(x2) - ref2).max() < 1e-3


def test_cut_window_matches_training_windows():
    x = _speechlike(12.0)
    for end_s, cut_s in ((10.0, None), (11.5, 10.6), (13.0, None), (6.0, None)):
        a = ami_eval.cut_window(x, end_s, 9.3, cut_s, np.random.default_rng(3))
        xr = x[: int(round(cut_s * SR))] if cut_s is not None else x
        b = cut_window_ref(xr, SR, end_s, 9.3, np.random.default_rng(3))
        assert a.shape == (8 * SR,) and np.array_equal(a, b), (end_s, cut_s)


def test_auc_and_rule_match_reference():
    from sklearn.metrics import roc_auc_score
    from eot import rules_eval as RE
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.normal(0.4, 0.3, 500), 0, 1).round(2)      # ties included
    assert abs(ami_eval.auc(y, s) - roc_auc_score(y, s)) < 1e-12
    N = 400
    label = rng.integers(0, 2, N); gap = np.where(label == 0, rng.exponential(0.8, N), np.nan)
    p = rng.random((N, 3))
    avail = (label[:, None] == 1) | (np.nan_to_num(gap, nan=np.inf)[:, None] > np.array(RE.OFFSETS)[None, :])
    p[~avail] = np.nan
    ours = ami_eval.simulate(p, avail, label, gap, (0.9, 0.7, 0.5))
    ref = RE.simulate(p, avail, label, gap, (0.9, 0.7, 0.5))
    assert (ours["recall"], ours["far"], ours["mean"], ours["p90"]) == (ref.recall, ref.far, ref.mean, ref.p90)
    assert ami_eval.fit(p, avail, label, gap, 0.2, grid=np.round(np.arange(0.0, 1.0001, 0.1), 1))["theta"] == \
        RE.fit(p, avail, label, gap, 0.2, grid=np.round(np.arange(0.0, 1.0001, 0.1), 1)).theta
