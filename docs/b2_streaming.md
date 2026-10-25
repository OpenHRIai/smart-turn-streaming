# Stage B2: streaming inference on the CPU

Status: 2026-10-24. Plan: `docs/plan_online_eot.md` §4 (the CPU budget) and §6 B2. Code:
`eot/streaming.py` (`IncrementalMel`, `StreamingEOT`), `scripts/b2_validate.py`, tests in
`tests/test_streaming.py`. Models: `data/models/smart-turn-v3.2-gpu.pooled.onnx` (frozen encoder
with the pooled vector exposed, 32 MB) and `data/models/eot_heads_v32.onnx` (the B1 heads from
`runs/b1/knob_ami07_auc`, 0.7 MB). Report: `runs/b2/validate.md`.

## What runs where

One object, one thread, two cadences, no GPU:

| path | when | work | cost (M4, 1 thread) |
|---|---|---|---|
| mel | every audio frame (20 ms) | STFT of the new 10 ms frames, mel power, kept in a ring of 800 columns | 0.019 ms per frame |
| **fast: heads** | every 100 ms, and at each silence-clock point | window normalisation, last 1 s → 20×80, GRU + MLP → `p_end`, `p_eager` | 0.72 ms p50, 1.6 ms p99 |
| **slow: encoder** | every 320 ms while the user speaks; at τ = 0, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5 s of silence | 8 s window → Whisper-tiny encoder → pooled 384-d vector (cached) | 22 ms p50, 38 ms p99 |

Share of one core over 11.5 s replays with real speech and silence: **9.1 %** single-threaded
(7.7 % with a plain 320 ms timer), **7.6 %** with four ORT threads (the encoder call drops to
17 ms, the hop rises to 1 ms from thread hand-offs). Decision-path burst, i.e. the work done on the
frame where a decision is made: 1.7 ms p99 for a hop, 25–40 ms on the seven silence-clock frames
that also refresh the encoder. Memory: the two ONNX sessions and a 1.3 MB mel ring.

## Incremental Whisper features

Whisper's log-mel for an 8 s window depends on three window-global quantities: the waveform mean
and standard deviation (`do_normalize`), and the window maximum (clamp at max − 8). All three are
applied after the fact. Each 10 ms column stores the mel power of DFT bins 2–200 plus the complex
bins 0 and 1; subtracting the window mean from a Hann-windowed frame changes only those two bins
(by −mean·W₀, −mean·W₁), dividing by σ scales every bin by 1/σ², and the log, the floor at 1e-10,
the clamp and the (x + 4)/4 are taken when the window is assembled (0.6 ms for 80×800). Reflect
padding at the window edges is replaced by the real audio before the window and by waiting 200
samples for the last column.

Against the HuggingFace extractor on the same window: real recordings (AMI headsets, pipecat
clips) agree within 0.003 on every interior column (0.0001 on a clean clip); synthetic signals
with a strong drifting DC component differ by up to 0.4 in mel bins 0–1 only, because the ring
mean and the window mean are 160 samples apart. The frozen model's probability computed from the
streaming features agrees with the cached offline one to AUC 0.689 vs 0.688 on 150 silences.

## Fidelity: streaming vs the offline B1 scores (150 AMI test silences, 75 complete / 75 incomplete)

Replay of the headset audio from 9 s before the VAD stop to 2.5 s after, 20 ms frames, VAD verdict
from the build's gated regions; for complete turns the audio after the next speaker's onset is
replaced by the speaker's room tone, as the offline windows were (a deployed microphone does not
hear the agent's reply). Offline = `runs/b1/knob_ami07_auc/predictions.npz` for the same items.

| encoder refresh schedule | AUC offline / streaming at τ = 0.2 s | at 0.5 s | at 1.2 s | mean / max \|Δp\| at 0.2 s | core |
|---|---|---|---|---|---|
| timer only, every 320 ms | 0.774 / 0.731 | 0.767 / 0.753 | 0.877 / 0.812 | 0.105 / 0.66 | 7.7 % |
| **silence clock (default)** | **0.774 / 0.765** | **0.767 / 0.758** | **0.877 / 0.865** | 0.061 / 0.63 | 9.1 % |
| refresh at every query (no staleness) | 0.774 / 0.765 | 0.767 / 0.760 | 0.877 / 0.875 | 0.061 / 0.63 | 26.1 % |

Reading:

1. **Staleness is not free.** A free-running 320 ms timer costs 0.04 AUC at the VAD stop and
   0.06 at 1.2 s: the cached vector at a query can date from before the speaker stopped, and the
   heads were trained on vectors computed at the window end. The design assumed training with
   random staleness; B1 did not do that.
2. **The silence clock removes it at no extra budget.** Refreshing at the VAD stop and at the τ
   grid the heads were trained on puts every decision that matters on a fresh vector; the
   streaming AUCs are within 0.01 of the offline ones, and the per-call cost is unchanged because
   the refreshes merely move from the timer to the silence clock (9.1 % vs 7.7 % of a core).
3. **The residual mismatch is the room tone and the hop grid**, not the pipeline: the offline
   windows' room tone was tiled with a different random draw, and a scheduled hop lands up to
   100 ms after the nominal τ. Aligning the query exactly leaves the same numbers.
4. **The eager head during speech still sees a vector up to 320 ms old.** Its streaming AUC is not
   measured here (B3, in Pipecat, with the eager cancel path). Halving the speech-time timer to
   160 ms costs about 7 % of a core more if it turns out to matter.

## Answer to the CPU question

Yes. The complete online detector, Whisper-tiny encoder included, runs on one CPU thread at
about 9 % of an Apple M4 core and would run at roughly 25–30 % of a cloud vCPU (pipecat's own
figure for the same encoder on Pipecat Cloud is ~3× the M4 time), with a 0.7 ms decision every
100 ms, a 22 ms burst seven times per silence, 33 MB of model files, and streaming scores that
match the offline evaluation within 0.01 AUC. Everything in `design_streaming_eot.md` §2's CPU
target (≤ 30 % of a core, no burst over 40 ms, files ≤ 40 MB) is met.

## Next

B3: the Pipecat analyzer around `StreamingEOT` (`append_audio` drives it, `analyze_end_of_turn`
returns the current verdict), the θ(τ) table fitted on AMI dev, the end-to-end replay with
`measure_pipecat_latency.py` against the stock Smart Turn analyzer, and the eager cancel path.

## Reproduce

```
.venv/bin/python -m pytest -q tests/test_streaming.py
.venv/bin/python scripts/b2_validate.py --run runs/b1/knob_ami07_auc --n 150 --out runs/b2/validate.md   # ~15 min
```
