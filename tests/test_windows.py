import numpy as np
import pytest

from eot import windows as W
from eot.offsets import energy_offset, frame_rms_dbfs

SR = 16000


def _clip(speech_s=2.0, tail_s=0.3, noise_db=-55.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(speech_s * SR)) / SR
    speech = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    tail = rng.standard_normal(int(tail_s * SR)) * 10 ** (noise_db / 20)
    x = np.concatenate([speech, tail]).astype(np.float32)
    return x, speech_s


def test_cut_window_length_and_tau_padding():
    x, t_off = _clip()
    rng = np.random.default_rng(1)
    w = W.variant(x, SR, t_off, "tau", 2.5, rng)
    assert w.shape == (int(W.WIN_S * SR),)
    # the window ends 2.5 s after the offset; the last 2 s are synthesised room tone
    tail = w[-int(2.0 * SR):]
    db = frame_rms_dbfs(tail)
    assert np.all(db < -45), db.max()
    assert np.all(db > -70), db.min()
    # speech is present just before the offset
    spk = w[-int(2.5 * SR) - 1600: -int(2.5 * SR)]
    assert 20 * np.log10(np.sqrt(np.mean(spk ** 2))) > -20
    # leading part (before the clip) is zeros
    assert np.all(w[: int((W.WIN_S - 2.0 - 2.5) * SR) - 10] == 0)


def test_tau_zero_matches_clip_up_to_offset():
    x, t_off = _clip()
    w = W.variant(x, SR, t_off, "tau", 0.0, np.random.default_rng(0))
    n = int(t_off * SR)
    assert np.allclose(w[-n:], x[:n])


def test_pre_offset_cut_ends_in_speech():
    x, t_off = _clip()
    w = W.variant(x, SR, t_off, "pre", 0.4, np.random.default_rng(0))
    assert np.allclose(w[-int(1.6 * SR):], x[: int(1.6 * SR)])
    assert 20 * np.log10(np.sqrt(np.mean(w[-800:] ** 2))) > -20


def test_short_tail_synthesises_room_tone():
    x, t_off = _clip(tail_s=0.0)
    w = W.variant(x, SR, t_off, "tau", 1.2, np.random.default_rng(0))
    tail = w[-int(1.0 * SR):]
    # a clip that is speech throughout gets the capped fallback level (-45 dBFS)
    assert 0 < np.sqrt(np.mean(tail ** 2)) <= 10 ** (-45 / 20) * 1.05


def test_enumerate_items_labels():
    items = W.enumerate_items(t_on=0.2, t_off=3.0, label=1, rng=np.random.default_rng(0))
    taus = [i for i in items if i["kind"] == "tau"]
    pres = [i for i in items if i["kind"] == "pre"]
    mids = [i for i in items if i["kind"] == "mid"]
    assert len(taus) == len(W.TAUS) and all(i["p_end"] == 1 and i["p_eager"] is None for i in taus)
    assert len(pres) == len(W.DELTAS)
    assert all(i["p_end"] == 0 for i in pres)
    assert [i["p_eager"] for i in pres] == [1, 1, 1, 1, 0, 0]
    assert len(mids) == 1 and 1.2 <= mids[0]["value"] <= 2.0
    inc = W.enumerate_items(t_on=0.2, t_off=3.0, label=0)
    assert all(i["p_eager"] in (None, 0) for i in inc)
    short = W.enumerate_items(t_on=0.0, t_off=0.9, label=1)
    assert [i["value"] for i in short if i["kind"] == "pre"] == [0.1, 0.2, 0.3, 0.4]
    assert not [i for i in short if i["kind"] == "mid"]


def test_energy_offset_on_synthetic_clip():
    x, t_off = _clip(tail_s=0.5)
    assert abs(energy_offset(x, SR) - t_off) <= 0.04
