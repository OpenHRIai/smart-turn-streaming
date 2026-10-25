---
license: bsd-2-clause
library_name: onnx
pipeline_tag: audio-classification
tags:
  - turn-detection
  - end-of-turn
  - voice-agents
  - pipecat
  - streaming
  - onnx
datasets:
  - pipecat-ai/smart-turn-data-v3.2-train
language:
  - en
  - multilingual
base_model: pipecat-ai/smart-turn-v3
---

# Smart Turn streaming end-of-turn heads (R1, audio-only)

A silence-aware, streaming end-of-turn detector for voice agents, built on Pipecat's frozen
[Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) encoder. Instead of one query
at the VAD stop followed by a 3 s timeout, it scores the conversation every 100 ms on the CPU and
decides at 0.2 / 0.5 / 1.2 s of silence with thresholds fitted to a false-alarm budget. On real
multi-party conversation (AMI Meeting Corpus) it ends the turn on 5 % of mid-turn pauses where the
stock rule fires on 42 %, and brings the 90th-percentile response time from 3.2 s to 1.3 s. A Pipecat
process with it takes about 17 % of one Apple M4 core against 9 % with the stock analyzer (the detector
alone is 9 % in a tight loop); numpy and onnxruntime only.

Code (Pipecat 1.14 turn analyzer, eager-then-commit stop strategy, training and evaluation):
[github.com/OpenHRIai/smart-turn-streaming](https://github.com/OpenHRIai/smart-turn-streaming).

![Wins and costs against stock Smart Turn v3.2](https://raw.githubusercontent.com/OpenHRIai/smart-turn-streaming/main/docs/figures/wins_and_costs.png)

![One turn end under the three rules](https://raw.githubusercontent.com/OpenHRIai/smart-turn-streaming/main/docs/figures/timeline_turn_end.png)

![One mid-turn pause under the three rules](https://raw.githubusercontent.com/OpenHRIai/smart-turn-streaming/main/docs/figures/timeline_pause.png)

## Files

| file | size | what |
|---|---|---|
| `smart-turn-v3.2-encoder-pooled.onnx` | 31 MB | Pipecat's Smart Turn v3.2 (Whisper-tiny encoder + its classifier), weights unchanged, with the pooled 384-d vector exposed as a second output. Input `input_features` (1, 80, 800) Whisper log-mel of the last 8 s; outputs `logits` (the stock model's P(complete)) and `sum_1` (the pooled vector). |
| `eot-heads-v32.onnx` | 0.7 MB | The trained heads: LayerNorm + Linear on the pooled vector, a GRU over the last second of log-mel (20 × 80 at 50 ms), the silence length, a fusion MLP. Inputs `pooled` (1, 384), `mel20` (1, 20, 80), `tau_feat` (1, 3) = [τ / 2.5, log1p(τ), speaking]; output `probs` (1, 2) = [P(turn complete), P(turn ends within 0.4 s)]. 172k parameters. |
| `smart-turn-streaming-clipend.onnx` | 33 MB | Drop-in with the stock interface: the pooled encoder, the heads at τ = 0 and a Platt calibration fitted on the training split in one graph, `input_features` (B, 80, 800) → `probability` (B, 1). For pipecat's `benchmark.py` and for callers that ask once at the clip end; the streaming rules use the two files above. |
| `rules/theta_tables.csv` | | Commit thresholds θ(0.2 s), θ(0.5 s), θ(1.2 s) fitted on AMI dev at false-alarm budgets of 5 / 10 / 15 %, with their AMI test numbers. |
| `rules/eager_tables.csv` | | Eager thresholds for Pipecat's speculation path, fitted on AMI dev at budgets of LLM generations spent on pauses (20–50 %). |
| `eval/ami_eval.py`, `eval/ami_{dev,test}.csv`, `eval/ami_eval_reference.md` | | Reproduces the AMI tables below from the public corpus audio (numpy, onnxruntime, soundfile); the reference file is its output on 2026-10-25. |
| `LICENSE`, `AMI_LICENCE.txt`, `ATTRIBUTION.md` | | BSD 2-Clause for this release; CC BY 4.0 for the AMI-derived manifests; sources and citations. |

## How it works

Two cadences in one object on one CPU thread, plus one worker thread for the encoder:

| path | when | work | cost (Apple M4, 1 ORT thread) |
|---|---|---|---|
| mel | every 20 ms audio frame | incremental Whisper log-mel, one 10 ms column at a time (exact to 0.003 against the reference extractor on real audio) | 0.02 ms |
| heads | every 100 ms, and at each silence-clock point | last 1 s of mel → GRU, cached pooled vector, τ → P(complete) | 0.7 ms |
| encoder | every 320 ms while the user speaks; at τ = 0, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5 s of silence (the grid the heads were trained on), where the decision waits for it | 8 s window → Whisper-tiny → pooled vector, on a worker thread | 22 ms (46 ms when the core is otherwise idle) |

The decision rule queries P(complete) at 0.2, 0.5 and 1.2 s of silence while the pause lasts and
commits at the first offset where it exceeds θ(offset); Pipecat's `stop_secs` (3 s) stays as the
backstop. The budget is the product knob: `far10` for an agent that must not interrupt, `far15` when
a fast reply matters more. A second, looser table (`eager_tables.csv`) on the same decision points
drives Pipecat's speculation path (`UserTurnSpeculation` → `SpeculationGate`): the LLM starts
generating at the eager point, the response is held until the commit and discarded if the user
resumes.

Why silence awareness has to be trained: 79 % of the clips in `smart-turn-data-v3.2` end within
0.2 s of the speech, so the published models never saw a pause, and their false-alarm rate on
incomplete utterances rises from 5 % at the offset to 19 % after 2.5 s of silence. The heads were
trained on windows ending τ ∈ {0 … 2.5} s after the acoustic offset (clips padded with their own
room tone) plus 43k real conversational pauses from AMI. The GRU over the last second of mel is the
component that carries the silence awareness (AUC at 1.2 s of silence 0.74 with it, 0.66 without).

## Results

### AMI Meeting Corpus, test split (511 turn ends, 4,304 mid-turn pauses, 16 meetings, 4-party)

Offline, windows ending at τ seconds of silence (`eval/ami_eval.py` reproduces this table):

| τ (s) | AUC heads | AUC frozen Smart Turn v3.2 |
|---|---|---|
| 0.2 | 0.775 | 0.671 |
| 0.5 | 0.776 | 0.638 |
| 1.2 | 0.742 | 0.575 |
| 2.5 | 0.735 | 0.619 |

Over three training seeds the AUC at 0.2 s is 0.78 ± 0.01 and at 1.2 s 0.73 ± 0.01.

Decision rule, thresholds fitted on AMI dev, scored on AMI test; a turn end with no commit is
charged the 3 s timeout:

| rule | θ at 0.2 / 0.5 / 1.2 s | recall | false alarms on pauses | mean latency | FAR on pauses ≥ 1.5 s |
|---|---|---|---|---|---|
| stock rule (one query at the VAD stop, θ 0.5), frozen v3.2 | 0.50 / – / – | 0.51 | 30.4 % | 1.56 s | 0.45 |
| stock rule, heads | 0.50 / – / – | 0.64 | 25.5 % | 1.21 s | 0.46 |
| re-query, 10 % budget, frozen v3.2 | 1.00 / 1.00 / 0.28 | 0.70 | 10.6 % | 1.74 s | 0.61 |
| **re-query, 10 % budget, heads (`far10`)** | 0.92 / 0.98 / 0.86 | **0.80** | **9.4 %** | **1.48 s** | 0.50 |
| re-query, 15 % budget, frozen v3.2 | 1.00 / 1.00 / 0.02 | 0.87 | 13.7 % | 1.44 s | 0.78 |
| **re-query, 15 % budget, heads (`far15`)** | 0.98 / 0.88 / 0.80 | **0.86** | **13.3 %** | **1.21 s** | 0.61 |
| no model: end every turn at 1.2 s of silence | – | 1.00 | 18.0 % | 1.20 s | 1.00 |

### Pipecat's official benchmark (`benchmark.py`, full `smart-turn-data-v3.2` test, 31,527 clips, threshold 0.5, CPU)

| model | accuracy | precision | recall | F1 | FPR | inference p50 |
|---|---|---|---|---|---|---|
| Smart Turn v3.2 (fp32) | **93.71 %** | 0.931 | 0.944 | 0.937 | 3.51 % | 10.2 ms |
| `smart-turn-streaming-clipend.onnx` (heads, calibrated) | 93.23 % | 0.919 | 0.947 | 0.933 | 4.15 % | 10.3 ms |

The regression guard, not the result: the benchmark has no pauses, so the re-query, the eager stage and the
cancel path never run. At the clip end the calibrated heads trail v3.2 by 0.5 points at the fixed 0.5
threshold (same ranking, AUC 0.975 vs 0.978; the residual is the operating point), with the same per-language
pattern within noise (19 of 23 languages within 0.7 points; French, Polish and Turkish ahead; Arabic −1.6,
Marathi −1.4, Norwegian −1.1 behind). Per-language tables and the method: `docs/c3_benchmark.md` in the code
repository.

### Regression guard: `smart-turn-data-v3.2` test (6,000 clips, balanced)

| τ (s) | AUC heads / frozen v3.2 | recall at 5 % FAR, heads / frozen |
|---|---|---|
| 0.0 | 0.975 / 0.978 | 0.87 / 0.89 |
| 0.2 | 0.976 / 0.976 | 0.87 / 0.86 |
| 1.2 | 0.972 / 0.964 | 0.84 / 0.77 |
| 2.5 | 0.965 / 0.955 | 0.81 / 0.72 |

Per-language and per-source AUC at τ = 0.2 s is unchanged within noise. The heads are calibrated
differently from the frozen model (class-weighted loss), so use the fitted tables, not θ = 0.5.

### End to end in Pipecat 1.14 (real-time replay of 160 AMI test silences through Silero VAD and `UserTurnProcessor`)

| analyzer and rule | false alarms on pauses | turn ends committed by the model (not the 3 s backstop) | commit latency p50 / p90 | response time p50 / p90 / mean at 0.8 s of LLM latency | LLM generations spent on pauses |
|---|---|---|---|---|---|
| stock `LocalSmartTurnAnalyzerV3`, v3.2 fp32 | 42.5 % | 0.70 | 0.23 / 3.20 s | 1.03 / 4.00 / 1.89 s | 42.5 % |
| streaming, `far10` | 2.5 % | 0.78 | 1.21 / 3.00 s | 2.01 / 3.80 / 2.36 s | 2.5 % |
| streaming, `far15` | 5.0 % | 0.93 | 1.19 / 1.27 s | 1.99 / 2.07 / 1.92 s | 5.0 % |
| **streaming, `far15` + eager 30 %** | **5.0 %** | **0.93** | 1.19 / 1.26 s | **1.31 / 2.04 / 1.55 s** | 18.8 % |

With the eager stage the median response time matches the stock analyzer within 0.3 s, the p90 is
halved, an eighth of the user turns are split, and only 2.5 % of pauses are audibly interrupted
(stock: 7.5 %).

### Cost (Apple M4, single ORT thread, numpy + onnxruntime)

| | |
|---|---|
| detector share of one core, tight loop | 9.1 % (encoder refreshes included) |
| whole Pipecat process (Silero VAD, pipeline, analyzer) | 16–19 % of one core at real-time pace (stock analyzer: 9 %) |
| heads hop | 0.7 ms |
| encoder refresh | 22 ms (46 ms at real-time pace on an otherwise idle machine) |
| pipeline-task wall time per audio frame, encoder on the worker thread | p99 4–5 ms (inline: 45 ms); the 0.5 s and 1.2 s decisions 0.2 ms |
| resident memory added by the analyzer | +114 MB (stock: +76 MB) |
| model files | 32 MB |

A cloud vCPU is about 3× slower than an M4 core for this encoder (Pipecat's own figure for the same
architecture on Pipecat Cloud), so expect roughly 25–30 % of one vCPU for the detector alone.

## Usage

### In Pipecat (recommended)

```python
from eot.pipecat_analyzer import StreamingEOTAnalyzer, StreamingEOTParams
from eot.pipecat_eager_strategy import TurnAnalyzerEagerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy

analyzer = StreamingEOTAnalyzer(
    encoder_onnx="smart-turn-v3.2-encoder-pooled.onnx",
    heads_onnx="eot-heads-v32.onnx",
    params=StreamingEOTParams.from_rule("far15", eager="far15_gen30"),   # commit and eager tables
)
strategies = UserTurnStrategies(
    start=[VADUserTurnStartStrategy()],
    stop=[TurnAnalyzerEagerUserTurnStopStrategy(turn_analyzer=analyzer)],   # or pipecat's TurnAnalyzerUserTurnStopStrategy
)
```

`eot/` is in the repository linked above; the analyzer needs 16 kHz audio and Pipecat's default VAD
`stop_secs` of 0.2 s. `StreamingEOTParams.from_rule("far10")` without an eager table gives the
one-stage rule.

### Standalone streaming (numpy + onnxruntime)

```python
from eot.streaming import StreamingEOT

eot = StreamingEOT("smart-turn-v3.2-encoder-pooled.onnx", "eot-heads-v32.onnx", worker=True)
for frame, is_speech in audio_frames():        # 20 ms float32 frames in [-1, 1] and the VAD's verdict
    out = eot.push(frame, is_speech)          # None, or {"p_end", "p_eager", "tau", "stale_s", ...} at a hop
    if out and not out["speaking"] and out["tau"] >= 1.2 and out["p_end"] >= 0.80:
        ...                                    # end of turn (the far15 table's 1.2 s threshold)
```

### One window, offline

```python
import numpy as np, onnxruntime as ort
enc = ort.InferenceSession("smart-turn-v3.2-encoder-pooled.onnx")
heads = ort.InferenceSession("eot-heads-v32.onnx")
feats = whisper_logmel(last_8_seconds)          # (80, 800); eval/ami_eval.py has a numpy implementation
logits, pooled = enc.run(None, {"input_features": feats[None]})
mel20 = feats[:, -100:].reshape(80, 20, 5).mean(-1).T[None]       # last 1 s at 50 ms steps
tau = 0.5                                                          # seconds of silence at the window end
tau_feat = np.array([[tau / 2.5, np.log1p(tau), 0.0]], np.float32)
p_end, p_eager = heads.run(None, {"pooled": pooled, "mel20": mel20, "tau_feat": tau_feat})[0][0]
```

## Limitations

- **Conversational evaluation is English-only.** The silence-aware behaviour was trained and measured on
  English meetings (AMI); the per-language check covers 23 languages but only on isolated utterances
  (`smart-turn-data`), where the heads stay within noise of the frozen model in every language and hold the decision through
  silence after an unfinished utterance in all 23 (false alarms after 2.5 s of silence 9.4 % against the frozen
  model's 22.5 % at a matched 5 % at the clip end). Whether
  the pause thresholds transfer to other languages' conversational timing is not measured.
- Meetings have a ceiling set by the labels: ending every turn at 1.2 s of silence
  gives recall 1.0 at 18 % false alarms, and below that budget about a quarter of turn ends still
  wait for the backstop, for every model tried.
- Long pauses (≥ 1.5 s) remain the weak spot: 50–60 % false alarms at the 10–15 % budgets.
- The second output (`p_eager`, "the turn ends within 0.4 s") is not usable as a free-running
  trigger during speech (AUC 0.70 on AMI; ~22 false fires per minute of speech). The eager tables use
  P(complete) at the silence-clock points instead.
- Every decision waits for an encoder refresh of its own window: the 0.2 s decision at the VAD stop for up
  to 50 ms (`wait_s`), the 0.5 s and 1.2 s decisions for up to 80 ms (`max_defer_s`); the encoder takes
  22–46 ms on an M4 and about 3× that on a cloud vCPU, so raise the allowances there. If the refresh is
  late the decision uses the cached vector, up to 320 ms old, and the analyzer records the staleness of
  every decision. With the refresh in time the streaming decisions equal the offline evaluation's within
  0.001 AUC on AMI.
- Labels are by outcome, not intent: a speaker who was cut off counts as a complete turn.

## Reproduce the AMI tables

```
pip install numpy onnxruntime soundfile
python eval/ami_eval.py --audio-dir ./ami_audio          # downloads 34 headset files (~3 GB), ~10 min on an M4
```

## Citation

If you use this model, please cite Pipecat's Smart Turn (the encoder) and the AMI Meeting Corpus
(`ATTRIBUTION.md`).
