# Stage B3: the streaming detector in Pipecat, end to end against stock Smart Turn

Status: 2026-10-24. Code: `eot/pipecat_analyzer.py` (`StreamingEOTAnalyzer`, a pipecat
`BaseTurnAnalyzer`), `scripts/b3_replay.py` (the replay harness, built on Phase 0's
`measure_pipecat_latency.py`), results in `runs/b3/` (CSV per run, `ps_samples.csv`,
`theta_tables.csv`). Pipecat 1.14.0, Apple M4, CPU only.

## What was measured

160 AMI test silences (80 turn ends, 80 mid-turn pauses, seed 0) replayed from 9 s before the VAD
stop to 3.6 s after it, 20 ms frames at real-time pace, through a real pipeline:
`VADProcessor(SileroVADAnalyzer)` → `UserTurnProcessor` with a VAD start strategy and
`TurnAnalyzerUserTurnStopStrategy(analyzer)`. Headset audio normalised to −23 dBFS (Pipecat's VAD
gates on loudness); for turn ends the audio after the next speaker's onset is replaced by the
speaker's room tone, since a deployed microphone does not hear the agent. Latency is the wall-clock
time from the last speech sample to `on_user_turn_stopped`, compute included; a turn end with no
stop is charged 3.2 s (VAD stop 0.2 s + `stop_secs` 3 s). "False alarms, strict" are stops before
the speaker resumed; "effective" also counts stops within the VAD's 0.2 s start delay after the
resumption, when Pipecat could not have noticed it yet.

Four configurations, same silences:

| analyzer | rule | θ at 0.2 / 0.5 / 1.2 s |
|---|---|---|
| stock `LocalSmartTurnAnalyzerV3`, v3.2 fp32 model | one query at the VAD stop, then `stop_secs` | 0.5 / – / – |
| our heads (v3.2 encoder + B1 heads), streaming | the same one-shot rule, for a like-for-like comparison | 0.5 / – / – |
| our heads, streaming | re-query table fitted on AMI dev, 10 % false-alarm budget | 0.92 / 0.98 / 0.86 |
| our heads, streaming | re-query table, 15 % budget | 0.98 / 0.88 / 0.80 |

## Results

| analyzer and rule | turn ended within 3.6 s | false alarms on pauses, strict / effective | latency on turn ends: p50 / p90 / mean | committed at 0.2 / 1.2 s · by `stop_secs` · never |
|---|---|---|---|---|
| **stock Smart Turn v3.2, one-shot** | 0.97 | **42.5 % / 42.5 %** | 0.23 / **3.20** / 1.09 s | 56 / – · 22 · 2 |
| our heads, one-shot θ 0.5 | 0.97 | 35.0 % / 37.5 % | 0.25 / 3.02 / 1.21 s | 51 / – · 27 · 2 |
| streaming, 10 % budget | 0.97 | 2.5 % / 5.0 % | 1.21 / 3.00 / 1.56 s | 5 / 57 · 16 · 2 |
| **streaming, 15 % budget** | 0.99 | **5.0 % / 8.8 %** | 1.19 / **1.27** / 1.12 s | 2 / 65 · 5 · 1 |

False alarms (strict) by pause length, 38 / 37 / 5 pauses under 0.5 s, 0.5–1.5 s, ≥ 1.5 s:

| rule | < 0.5 s | 0.5–1.5 s | ≥ 1.5 s | recall before the timeout |
|---|---|---|---|---|
| stock one-shot | 0.37 | 0.49 | 0.40 | 0.70 |
| our heads, one-shot | 0.32 | 0.38 | 0.40 | 0.64 |
| streaming, 10 % | 0.00 | 0.00 | 0.40 | 0.78 |
| streaming, 15 % | 0.00 | 0.05 | 0.40 | 0.93 |

Pipeline overhead per commit, measured from Pipecat's own VAD offset: +30 ms for the stock query,
+30 to +48 ms for ours (the hop plus one 20 ms frame). Pipecat's VAD placed the offset within 60 ms
of the dataset's on the median replay; 51 of 440 commits differ by more than 0.3 s.

## CPU and memory

Whole replay process (pipeline, Silero VAD, analyzer), one process at a time on the machine,
40 silences each, `getrusage` CPU time over wall time per replay and `ps` samples every 5 s:

| configuration | CPU share of one core (mean / p95) | process RSS |
|---|---|---|
| stock Smart Turn | 0.087 / 0.098 | – (dominated by the harness's audio cache) |
| streaming, 15 % budget | 0.265 / 0.272 | – |

So the streaming analyzer adds about 18 percentage points of one M4 core over the stock pipeline
while audio flows. That is more than the 9 % that B2's per-call accounting gave, because the
whole-process figure includes the Python frame handling around the calls, the silence-clock
refreshes on every pause in the 9 s of pre-roll (seven 22 ms encoder runs per pause) and the
speech-time timer; the four-way concurrent run read 22–23 % per process. All of it runs on the
pipeline task in this analyzer; a production version moves the encoder refresh to a worker thread
and could halve the encoder part with four ORT threads (B2: 22 → 13 ms). The design budget
(≤ 30 % of one core) holds on the M4 with margin; a cloud vCPU is about 3× slower, so expect
roughly 50–60 % of one vCPU there, or run the refresh with two threads.

Analyzer memory, measured in isolation (`psutil` RSS before and after construction plus a 12 s
replay, then 60 s more of audio):

| analyzer | resident memory added | growth over 72 s of audio |
|---|---|---|
| stock Smart Turn v3.2 (fp32 model) | +76 MB | – |
| streaming (encoder + heads sessions, mel ring, feature state) | +114 MB | +1 MB (bounded ring buffers) |

The first measurement showed +265 MB: `eot/streaming.py` imported `transformers` for the mel
filterbank, ~190 MB of resident memory. The filterbank is now computed in numpy (Slaney mel scale
and normalisation, identical to transformers' to 1e-9), so the inference path imports only numpy
and onnxruntime; neither transformers nor torch is loaded.

## Reading

1. **The stock rule's problem on real conversations is false alarms, not speed.** Smart Turn v3.2
   at θ 0.5 answers in 0.23 s when it fires, but on AMI pauses it fires 42 % of the time, and it
   leaves 22 of 80 real turn ends to the 3 s timeout. The same rule on our heads is a little better
   (35 %) but of the same kind: a single query at 0.2 s with θ 0.5 is the wrong operating point for
   this data (Phase 0 saw 13 % on Seamless, which is easier).
2. **The streaming rules move the operating point to where the budget says.** At the 15 % budget:
   5 % strict false alarms, 93 % of turn ends committed by the model before the timeout, p90 latency
   1.27 s instead of 3.20 s, mean 1.12 s against 1.09 s. The cost is the median: 1.19 s instead of
   0.23 s, because the dev-fitted θ(0.2 s) is so strict that only 2 of 80 turn ends commit at the
   VAD stop. The 10 % budget trades 15 more points of recall-before-timeout for 2.5 points of FAR.
3. **Long pauses are still the floor.** On the 5 pauses ≥ 1.5 s every rule fires on 2; the rules only
   remove the false alarms on short and medium pauses (49 % → 5 % at 0.5–1.5 s).
4. **The offline simulation predicted this.** B1's dev-fitted rule on the full AMI test set gave
   recall 0.80 at FAR 9.4 % and mean latency 1.48 s for the 10 % budget; the live pipeline on 80 + 80
   silences gives 0.78 recall-before-timeout at 2.5 % and 1.56 s. The sample's pauses are shorter
   than the full set's (median 0.5 s, only 5 of 80 above 1.5 s), which is where the FAR difference
   comes from; the pipeline adds 30–48 ms per commit and nothing else.
5. **What a product gets to choose.** The budget is the knob: 10 % for an agent that must not
   interrupt, 15–20 % for one where a fast reply matters more. Stock Smart Turn offers no such knob.
6. **Not measured here**: the eager signal. The analyzer records `p_eager` at every decision point
   but Pipecat's speculative path (`EagerUserTurnStopStrategy`, `SpeculationGate`) is driven only by
   STT services; a `TurnAnalyzerEagerUserTurnStopStrategy` is the upstream change of Stage C.
   *Done in C1 (`c1_eager.md`): with 0.8 s of response latency only 7.5 % of the stock rule's 42.5 %
   false alarms above are audible (1.3 % of the streaming rule's 5 %); the two-stage rule built on
   that observation is measured there with the same replays.*

## Reproduce

```
.venv/bin/python scripts/b3_replay.py --analyzer stock --rule one_shot --n-complete 80 --n-incomplete 80 --out runs/b3/stock.csv --quiet
.venv/bin/python scripts/b3_replay.py --analyzer streaming --rule one_shot --n-complete 80 --n-incomplete 80 --out runs/b3/streaming_oneshot.csv --quiet
.venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far10 --n-complete 80 --n-incomplete 80 --out runs/b3/streaming_far10.csv --quiet
.venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far15 --n-complete 80 --n-incomplete 80 --out runs/b3/streaming_far15.csv --quiet
.venv/bin/python scripts/b3_replay.py --report runs/b3/stock.csv runs/b3/streaming_oneshot.csv runs/b3/streaming_far10.csv runs/b3/streaming_far15.csv
sh runs/b3/sample_ps.sh &    # CPU / RSS samples of running replays -> runs/b3/ps_samples.csv
```
Each replay is real-time (12.6 s of audio); 160 silences take about 35 min per configuration. The
per-replay CSV rows carry `cpu_share` and `max_rss_mb` from `getrusage`.
