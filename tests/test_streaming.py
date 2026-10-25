"""Incremental mel against the HF WhisperFeatureExtractor, and the streaming object's plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from eot.streaming import SR, IncrementalMel


def _speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.7 * t)
    x = np.zeros_like(t)
    for h in range(1, 12):
        x += np.sin(2 * np.pi * h * np.cumsum(f0) / SR) / h
    env = (np.sin(2 * np.pi * 3 * t) > -0.2).astype(float)
    x = 0.2 * x * env + rng.standard_normal(len(t)) * 1e-3
    return x.astype(np.float32)


def test_incremental_mel_matches_hf_extractor():
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8)
    x = _speechlike(10.0)
    im = IncrementalMel()
    for i in range(0, len(x), 320):           # 20 ms frames
        im.push(x[i:i + 320])
    ours = im.window_features()               # last 8 s of columns, the last centred at (n_frames-1)*160
    # reference: the 8 s window whose HF frame grid coincides with ours: frames j=0..799 centred at s + j*160
    s0 = (im.n_frames - 1 - 799) * 160
    ref = fe(x[s0: s0 + 8 * SR], sampling_rate=SR, return_tensors="np", padding="max_length",
             max_length=8 * SR, truncation=True, do_normalize=True).input_features[0]
    d = np.abs(ours - ref)
    # interior columns, bins 2+: float error and the 160-sample offset of the sigma window
    assert d[2:, 2:-2].max() < 0.02, d[2:, 2:-2].max()
    # bins 0-1 carry the DC term: the synthetic signal has a strong, drifting mean, so the window
    # mean (HF) and the ring mean (ours, 160 samples apart) differ; real audio is near zero-mean
    # (see test_incremental_mel_on_real_audio)
    assert d[:2, 2:-2].max() < 0.6, d[:2, 2:-2].max()
    # edge columns: HF reflect-pads, we use real audio -> bounded difference
    assert d.max() < 1.0, d.max()
    m = im.mel20()
    assert m.shape == (20, 80)
    assert np.allclose(m, ours[:, -100:].reshape(80, 20, 5).mean(-1).T, atol=1e-6)


def test_short_stream_is_zero_padded_like_hf():
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8)
    x = _speechlike(3.0, seed=1)
    im = IncrementalMel()
    im.push(x)
    ours = im.window_features()
    # the trainer's convention: zeros at the START to 8 s, then HF (normalisation over the whole window)
    end = im.n_frames * 160
    win = np.concatenate([np.zeros(8 * SR - end, np.float32), x[:end]])
    ref = fe(win, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=8 * SR,
             truncation=True, do_normalize=True).input_features[0]
    d = np.abs(ours - ref)
    pad_cols = (8 * SR - end) // 160
    # the one frame straddling the padding boundary is not reproduced (we pad whole columns)
    assert d[2:, pad_cols + 2: -2].max() < 0.02, d[2:, pad_cols + 2: -2].max()
    assert d[:, : pad_cols - 1].max() < 0.05, d[:, : pad_cols - 1].max()      # padded columns (DC term of the synthetic signal)
    assert d.max() < 1.5, d.max()


def test_incremental_mel_on_real_audio():
    """Strict check on real recordings (near zero-mean): every interior bin within 0.01."""
    import os
    import soundfile as sf
    path = "data/ami/audio/ES2004a/ES2004a.Headset-0.wav"
    if not os.path.exists(path):
        pytest.skip("AMI audio not present")
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8)
    x, _ = sf.read(path, dtype="float32", start=SR * 120, frames=SR * 12)
    im = IncrementalMel()
    for i in range(0, len(x) // 320 * 320, 320):
        im.push(x[i:i + 320])
    s0 = (im.n_frames - 1 - 799) * 160
    ref = fe(x[s0: s0 + 8 * SR], sampling_rate=SR, return_tensors="np", padding="max_length",
             max_length=8 * SR, truncation=True, do_normalize=True).input_features[0]
    d = np.abs(im.window_features() - ref)
    assert d[:, 2:-2].max() < 0.01, d[:, 2:-2].max()


def test_streaming_object_runs(tmp_path):
    pytest.importorskip("onnxruntime")
    import os
    enc, heads = "data/models/smart-turn-v3.2-gpu.pooled.onnx", "data/models/eot_heads_v32.onnx"
    if not (os.path.exists(enc) and os.path.exists(heads)):
        pytest.skip("models not present")
    from eot.streaming import StreamingEOT
    s = StreamingEOT(enc, heads)
    x = _speechlike(4.0)
    outs = []
    for i in range(0, len(x), 320):
        o = s.push(x[i:i + 320], is_speech=i < len(x) - SR)
        if o:
            outs.append(o)
    # 100 ms hops over 4 s, plus the silence-clock hops at tau = 0, 0.2, 0.5, 0.8 during the last second
    assert 39 <= len(outs) <= 46, len(outs)
    assert 0.0 <= outs[-1]["p_end"] <= 1.0 and outs[-1]["tau"] > 0.9 and not outs[-1]["speaking"]
    assert outs[5]["speaking"] and outs[5]["tau"] == 0.0
    grid = [o for o in outs if not o["speaking"] and o["stale_s"] == 0.0]
    assert [round(o["tau"], 1) for o in grid][:4] == [0.0, 0.2, 0.5, 0.8], [o["tau"] for o in grid]   # tau is counted in 20 ms frames
    t = s.timings.summary()
    assert t["hop"]["n"] == len(outs) and t["refresh"]["n"] >= 12


# --- Stage C2: the encoder refresh on a worker thread -------------------------------------------

MODELS = ("data/models/smart-turn-v3.2-gpu.pooled.onnx", "data/models/eot_heads_v32.onnx")


def _need_models():
    import os
    pytest.importorskip("onnxruntime")
    if not all(os.path.exists(m) for m in MODELS):
        pytest.skip("models not present")


def test_worker_vector_matches_inline_when_waited():
    """With lead 0 the worker is asked at each silence-clock point and given one frame; waiting for it
    between frames must leave the inline detector's vector exactly (same window, same session)."""
    _need_models()
    from eot.streaming import StreamingEOT
    x = _speechlike(4.0)
    inline = StreamingEOT(*MODELS)
    work = StreamingEOT(*MODELS, worker=True, lead_s=0.0)
    try:
        grid_vecs = []
        for i in range(0, len(x), 320):
            sp = i < len(x) - SR
            inline.push(x[i:i + 320], sp)
            o = work.push(x[i:i + 320], sp)
            assert work.wait_refresh(5.0)                   # every refresh applied before the next frame
            if o is not None and not o["speaking"] and o["pending"] == 0 and o["stale_s"] <= 0.02 + 1e-9:
                grid_vecs.append((o["tau"], work.pooled.copy()))
        # the deferred silence-clock hops land one frame after the point (the transition at 0.02 s, then
        # 0.2 / 0.5 / 0.8 s), on a vector whose window ended at the point itself
        assert [round(t, 2) for t, _ in grid_vecs][:4] == [0.04, 0.22, 0.52, 0.82], [t for t, _ in grid_vecs]
        # same window -> same pooled vector as an inline refresh of that window gives
        chk = StreamingEOT(*MODELS, silence_grid=False, refresh_s=100.0)
        n_grid = 0
        for i in range(0, len(x), 320):
            chk.push(x[i:i + 320], i < len(x) - SR)
            if not chk.speaking and n_grid < len(grid_vecs) and abs(chk.silence_samples / SR - (grid_vecs[n_grid][0] - 0.02)) < 1e-9:
                chk.refresh()                           # the window the worker was handed
                assert np.allclose(chk.pooled, grid_vecs[n_grid][1], atol=1e-5), grid_vecs[n_grid][0]
                n_grid += 1
        assert n_grid == len(grid_vecs)
        t = work.timings.summary()
        assert t["n_stale_decisions"] == 0 and t["refresh"]["n"] >= 12 and t["wait"]["n"] > 0
    finally:
        work.close()


def test_worker_reproduces_inline_decisions_when_the_vector_arrives():
    """Lead 0 with a long defer and a wait between frames: each silence-clock decision of the worker path
    uses the vector, mel20 and tau of the point itself, so p_end equals the inline detector's at that point."""
    _need_models()
    from eot.streaming import StreamingEOT
    x = _speechlike(4.0)
    inline = StreamingEOT(*MODELS)
    work = StreamingEOT(*MODELS, worker=True, lead_s=0.0, max_defer_s=0.1)
    try:
        ref, got = {}, {}
        for i in range(0, len(x), 320):
            sp = i < len(x) - SR
            o = inline.push(x[i:i + 320], sp)
            if o is not None and not o["speaking"] and o["stale_s"] == 0.0:
                ref[round(o["tau"], 2)] = o["p_end"]
            o = work.push(x[i:i + 320], sp)
            work.wait_refresh(5.0)
            if o is not None and not o["speaking"] and o["pending"] == 0 and o["tau_vec"] != o["tau"]:
                got[round(o["tau_vec"], 2)] = o["p_end"]
        assert set(got) >= {0.2, 0.5, 0.8} and all(abs(got[k] - ref[k]) < 1e-6 for k in got), (ref, got)
    finally:
        work.close()


def test_worker_never_blocks_the_frame_path(monkeypatch):
    """A slow encoder (50 ms) with a one-frame allowance must not show up on the frame path: pushes stay
    under 5 ms, the decision at a silence-clock point is taken on the cached vector and recorded as stale,
    later hops pick the new vector up."""
    _need_models()
    import time as _time
    from eot.streaming import StreamingEOT
    s = StreamingEOT(*MODELS, worker=True, max_defer_s=0.02)
    real = s._encode

    def slow(feats):
        _time.sleep(0.05)
        return real(feats)
    monkeypatch.setattr(s, "_encode", slow)
    x = _speechlike(4.0)
    outs = []
    try:
        for i in range(0, len(x), 320):
            o = s.push(x[i:i + 320], i < len(x) - SR)
            _time.sleep(0.02)                               # real-time pace
            if o:
                outs.append(o)
        tm = s.timings.summary()
        assert tm["push"]["p99"] < 5.0 and tm["push"]["max"] < 20.0, tm["push"]
        assert tm["refresh"]["n"] >= 8
        stale = [o for o in outs if not o["speaking"] and o["pending"] > 0]
        fresh = [o for o in outs if not o["speaking"] and o["pending"] == 0]
        assert stale and fresh, (len(stale), len(fresh))      # some decisions waited, later ones caught up
        assert max(o["stale_s"] for o in outs) < 0.5          # the vector never gets older than the timer
        assert tm["n_stale_decisions"] >= 1
    finally:
        s.close()


def test_worker_defaults_decide_on_the_points_vector_at_real_time_pace():
    """At real-time pace with the defaults (request at the point, wait up to 80 ms), the 0.2 / 0.5 / 0.8 s
    decisions use the vector and snapshot of the point itself and land within 80 ms of it."""
    _need_models()
    import time as _time
    from eot.streaming import StreamingEOT
    s = StreamingEOT(*MODELS, worker=True)
    x = _speechlike(4.0)
    outs = []
    try:
        for i in range(0, len(x), 320):
            o = s.push(x[i:i + 320], i < len(x) - SR)
            _time.sleep(0.02)
            if o:
                outs.append(o)
    finally:
        s.close()
    pts = {round(o["tau_vec"], 2): o for o in outs if not o["speaking"] and o["grid"]}
    for tau in (0.2, 0.5, 0.8):
        o = pts[tau]
        assert o["pending"] == 0 and o["stale_s"] <= 0.08 + 1e-9 and o["tau"] - o["tau_vec"] <= 0.08 + 1e-9, (tau, o)
    tm = s.timings.summary()
    assert tm["decision"]["max"] < 5.0, tm["decision"]


def test_await_refresh_in_asyncio():
    _need_models()
    import asyncio
    from eot.streaming import StreamingEOT

    async def run():
        s = StreamingEOT(*MODELS, worker=True)
        try:
            s.push(_speechlike(2.0), True)
            fut = s.request_refresh()
            assert fut is not None and s.pending >= 1      # plus the timer's request from the push
            ok = await s.await_refresh(2.0)
            assert ok and s.pending == 0 and s.stale_s == 0.0
            assert not np.allclose(s.pooled, 0.0)
            return s.timings.summary()["wait"]["n"]
        finally:
            s.close()
    assert asyncio.run(run()) == 1


def test_analyzer_worker_path_decides_at_the_vad_stop():
    """StreamingEOTAnalyzer with the worker: analyze_end_of_turn awaits the refresh (here with a wait long
    enough for any machine) and records a fresh 0.2 s decision; append_audio never takes more than 5 ms."""
    _need_models()
    pytest.importorskip("pipecat")
    import asyncio
    import time as _time
    from eot.pipecat_analyzer import StreamingEOTAnalyzer, StreamingEOTParams

    async def run():
        an = StreamingEOTAnalyzer(encoder_onnx=MODELS[0], heads_onnx=MODELS[1], params=StreamingEOTParams.from_rule("far15"), wait_s=0.5)
        an.set_sample_rate(SR)
        x = (_speechlike(4.0) * 32767).astype(np.int16)
        n_speech = len(x) - SR
        for i in range(0, n_speech, 320):
            assert an.append_audio(x[i:i + 320].tobytes(), True) is not None
            await asyncio.sleep(0.02)
        # the VAD reports the stop 0.2 s into the silence (pipecat forwards those frames as speech)
        for i in range(n_speech, n_speech + 3200, 320):
            an.append_audio(x[i:i + 320].tobytes(), True)
            await asyncio.sleep(0.02)
        state, _ = await an.analyze_end_of_turn()
        d02 = [d for d in an.decisions if d["offset_s"] == 0.2]
        assert len(d02) == 1 and d02[0]["pending"] == 0 and d02[0]["stale_s"] == 0.0, d02
        for i in range(n_speech + 3200, len(x), 320):
            st = an.append_audio(x[i:i + 320].tobytes(), False)
            await asyncio.sleep(0.02)
            if st.name == "COMPLETE":
                break
        offs = sorted({d["offset_s"] for d in an.decisions})
        assert offs[:2] == [0.2, 0.5], offs
        assert all(d["pending"] == 0 for d in an.decisions), an.decisions
        assert np.percentile(an.append_ms, 99) < 5.0 and max(an.append_ms) < 15.0, (np.percentile(an.append_ms, 99), max(an.append_ms))
        assert an.analyze_ms and an.analyze_ms[0] < 200.0
        await an.cleanup()
    asyncio.run(run())
