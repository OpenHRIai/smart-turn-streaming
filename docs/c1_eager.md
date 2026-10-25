# Stage C1: the two-stage eager-then-commit rule on Pipecat's speculation path

Status: 2026-10-24. Code: `eot/pipecat_eager_strategy.py` (`TurnAnalyzerEagerUserTurnStopStrategy`),
`eot/pipecat_analyzer.py` (`eager_thresholds`, `pop_eager()`, `EAGER_RULES`), `scripts/c1_eager_sweep.py`
(offline fit and sweep), `scripts/b3_replay.py --strategy eager` (end-to-end replay, `--report-two-stage`),
`tests/test_eager_strategy.py`. Results in `runs/c1/` (`sweep.md`, `eager_tables.csv`, one CSV per replay).
Pipecat 1.14.0, Apple M4, CPU only.

## Why a second stage

B3 measured the stock rule's problem as false alarms: Smart Turn v3.2 at θ 0.5 ends the turn on 42 % of
AMI mid-turn pauses. But a turn end is only *heard* if the bot starts talking before the speaker resumes.
Re-scoring the B3 replays with a response latency L (inference trigger → first bot audio):

| rule (B3 replays, 80 pauses) | turn ends on pauses | audible at L = 0.5 / 0.8 / 1.2 s | turn ends left to the 3 s backstop |
|---|---|---|---|
| stock one-shot θ 0.5 | 42.5 % | 12.5 / 7.5 / 2.5 % | 22 of 80 |
| streaming far15 | 5.0 % | 1.3 / 1.3 / 1.3 % | 5 of 80 |

So with 0.8 s of response latency only 7.5 % of the stock rule's false alarms are audible; the rest are
paid for in LLM calls (a generation on 42 % of pauses) and in context hygiene (the user's turn is split
in two), and the stock rule still leaves a quarter of the real turn ends to the timeout. The streaming rule
removes the splits but spends its median turn end at 1.2 s of silence, and only then starts the LLM.

Pipecat already has the mechanism to separate the two costs. An STT service can report an *eager* end of
turn: `EagerUserTurnStopStrategy` starts a speculative inference against a provisional context, the LLM
service's `SpeculationGate` holds the response, `UserStoppedSpeakingFrame` releases it when the turn
commits and `EagerEndOfTurnCancelFrame` discards it when the user resumes. Only STT services drive it
today (`EagerEndOfTurnSTTServiceMixin`). A turn analyzer can drive the same path with two tables on the
same decision points:

```
eager    p_end(τ) ≥ θ_eager(τ)    start generating (speculation, response held)      budget: generations on pauses
commit   p_end(τ) ≥ θ_commit(τ)   end the turn, release the held response            budget: turn splits (strict FAR)
cancel   VAD reports speech       EagerEndOfTurnCancelFrame, response discarded
```

Timelines of the three rules on one turn end and on one mid-turn pause: `figures/timeline_turn_end.svg`,
`figures/timeline_pause.svg`; wins and costs against stock Smart Turn, one row per metric:
`figures/wins_and_costs.svg`; the response-time distributions of stock and the two-stage rule on the 80
replayed turn ends (stock is bimodal at 1 s and 4 s, the two-stage rule puts 92 % between 1 and 2 s):
`figures/response_time_distribution.svg`.

with θ_eager ≤ θ_commit. The response reaches the user at `max(t_commit, t_eager + L)` instead of
`t_commit + L`. A false alarm is audible if that time comes before the speaker resumes, so the eager stage
makes the committed false alarms *more* audible (the answer is ready when the commit lands); the eager
budget is a third product knob next to the commit budget and L.

## Implementation

**Analyzer** (`StreamingEOTAnalyzer`): `StreamingEOTParams.eager_thresholds`, one per rule offset. At each
silence-clock decision point the commit test runs first; if it fails and the eager table passes, the
analyzer raises the eager flag once per silence; `pop_eager()` hands it over and a speech resumption clears
it. No extra compute: the eager test reads the same `p_end` the commit test used.

**Strategy** (`TurnAnalyzerEagerUserTurnStopStrategy`, a subclass of pipecat's
`TurnAnalyzerUserTurnStopStrategy`): after `append_audio`, an eager flag with no speculation in flight
fires `on_user_turn_inference_triggered` with a `UserTurnSpeculation` on the latest transcript (final or
interim). The commit goes through `trigger_user_turn_stopped`: if the committed transcript matches the
speculated one (`NormalizedMatch`, as the STT strategy) it finalizes with `confirms_speculation=True` and
no second inference; otherwise it cancels and runs inference again. `VADUserStartedSpeakingFrame`, a turn
end without a commit, and `speculation_timeout` (5 s) all withdraw the speculation. An analyzer without
`pop_eager()` gets the base strategy's behaviour unchanged. Seven unit tests with a scripted analyzer
cover these paths. The proposed upstream shape is `BaseTurnAnalyzer.pop_eager() -> bool` (default False)
plus this strategy; `TurnAnalyzerEagerUserTurnStopStrategy` is the name `plan_online_eot.md` §6 reserved.

## Offline sweep (B1 heads `runs/b1/knob_ami07_auc`, AMI; `runs/c1/sweep.md`)

Commit tables as in B3 (dev-fitted, 10 % and 15 % false-alarm budgets). Eager tables fitted on AMI dev
(3,929 silences, 734 turn ends) by exhaustive grid, each θ_eager ≤ θ_commit, minimising the mean perceived
response time on turn ends at L = 0.8 s subject to a budget of generations spent on pauses; scored on AMI
test (4,331 silences with a decision point before the resumption, 511 turn ends; misses charged 3 s).

| rule | θ_commit 0.2 / 0.5 / 1.2 s | θ_eager | generations on pauses | turn splits | audible FAR | eager on turn ends / held | response time p50 / p90 / mean |
|---|---|---|---|---|---|---|---|
| stock one-shot θ 0.5 | 0.50 / – / – | – | 27.0 % | 25.5 % | 9.5 % | – | 1.00 / 3.80 / 2.01 s |
| far15, one stage | 0.98 / 0.88 / 0.80 | – | 14.5 % | 13.3 % | 5.9 % | – | 2.00 / 3.80 / 2.01 s |
| far15 + eager 20 % | | 0.92 / 0.84 / 0.68 | 18.8 % | 13.3 % | 6.5 % | 0.95 / 0.15 | 2.00 / 3.00 / 1.85 s |
| **far15 + eager 30 %** | | **0.84 / 0.56 / 0.42** | **28.8 %** | **13.3 %** | **8.7 %** | 0.99 / 0.58 | **1.30 / 3.00 / 1.57 s** |
| far15 + eager 40 % | | 0.72 / 0.24 / 0.04 | 38.6 % | 13.3 % | 9.3 % | 1.00 / 0.75 | 1.30 / 3.00 / 1.46 s |
| far10, one stage | 0.92 / 0.98 / 0.86 | – | 11.1 % | 9.4 % | 4.3 % | – | 2.00 / 3.80 / 2.28 s |
| far10 + eager 30 % | | 0.92 / 0.54 / 0.20 | 28.4 % | 9.4 % | 7.0 % | 0.99 / 0.60 | 1.30 / 3.00 / 1.71 s |

Reading:

1. **At the stock rule's generation cost the two-stage rule halves the turn splits and answers faster.**
   far15 + eager 30 % spends a generation on 28.8 % of pauses (stock: 27.0 %), splits 13.3 % of them
   (stock: 25.5 %), is heard on 8.7 % (stock: 9.5 %) and brings the mean perceived response time on turn
   ends from 2.01 s to 1.57 s, p90 from 3.80 s to 3.00 s. Against the one-stage far15 rule it buys 0.44 s
   of mean response time for 14 more points of generations and 2.8 points of audible false alarms; the
   turn splits do not move, since the commit table is unchanged.
2. **The 0.5 s eager point is the one that matters.** With the eager allowed only at the VAD stop
   (0.2 s), the 30 % budget gives a mean response time of 1.65 s; adding 0.5 s gives 1.56 s; adding 1.2 s
   changes nothing (1.57 s). The pattern: the eager at 0.5 s has the answer ready when the commit lands at
   1.2 s, and that is where most turn ends commit.
3. **The eager head (p_eager during speech) is not usable as a free-running trigger yet.** Oracle-timed at
   the trained δ grid it matches the p_end eager (mean response 1.60 s at 30 %), but at that threshold
   3.7 % of random mid-speech windows fire, about 22 false eagers per minute of speech at a 100 ms hop.
   It stays off until its AUC on conversational data improves (level 2 or Seamless training).
4. **Response latency sets how much the stage is worth.** At L = 0.3 s the gain is 0.21 s of mean
   response time (1.51 → 1.30 s), at 0.8 s it is 0.44 s, at 1.6 s it is 0.55 s; the audible false alarms
   fall with L for every rule (stock 17.7 % at 0.3 s, 4.4 % at 1.6 s) and the two-stage rule stays below
   stock at every L.

## End to end in Pipecat (160 AMI test silences, the B3 harness)

Same replays as B3 (80 turn ends, 80 pauses, seed 0, real-time 20 ms frames, Silero VAD,
`UserTurnProcessor`), with `TurnAnalyzerEagerUserTurnStopStrategy` and the fitted eager tables. The trace
records the speculative inference, its withdrawal and whether the commit confirmed it; the harness has no
STT, so the transcript match always holds (the withdrawal rate from transcript mismatch needs a run with an
STT service). Response time = max(commit, eager + L) when the commit confirmed a speculation, else
commit + L, L = 0.8 s; a turn end with no stop is charged 3.2 s as in B3.

| rule | generations on pauses | turn splits (strict FAR) | audible FAR | turn ends committed by the model | eager on turn ends / held / median lead to the commit | commit latency p50 / p90 / mean | **response time p50 / p90 / mean** |
|---|---|---|---|---|---|---|---|
| stock Smart Turn v3.2, one-shot (B3) | 42.5 % | 42.5 % | 7.5 % | 0.70 | – | 0.23 / 3.20 / 1.09 s | 1.03 / **4.00** / 1.89 s |
| streaming far15, one stage (B3) | 5.0 % | 5.0 % | 1.2 % | 0.93 | – | 1.19 / 1.27 / 1.12 s | 1.99 / 2.07 / 1.92 s |
| **streaming far15 + eager 30 %** | 18.8 % | **5.0 %** | **2.5 %** | 0.93 | 0.64 / 0.64 / 0.69 s | 1.19 / 1.26 / 1.12 s | **1.31 / 2.04 / 1.55 s** |
| streaming far15 + eager 40 % | 38.8 % | 5.0 % | 2.5 % | 0.93 | 0.86 / 0.85 / 0.69 s | 1.18 / 1.26 / 1.12 s | 1.24 / 2.02 / 1.41 s |
| streaming far10, one stage (B3) | 2.5 % | 2.5 % | 0.0 % | 0.78 | – | 1.21 / 3.00 / 1.56 s | 2.01 / 3.80 / 2.36 s |
| streaming far10 + eager 30 % | 20.0 % | 2.5 % | 2.5 % | 0.78 | 0.82 / 0.80 / 0.69 s | 1.20 / 3.00 / 1.56 s | 1.34 / 3.00 / 1.79 s |

Response time at other response latencies (far15 commit):

| L | stock one-shot: audible FAR, response p50 / p90 / mean | far15 one stage | far15 + eager 30 % |
|---|---|---|---|
| 0.5 s | 12.5 %, 0.73 / 3.70 / 1.59 s | 1.2 %, 1.69 / 1.77 / 1.62 s | 2.5 %, 1.20 / 1.74 / 1.35 s |
| 0.8 s | 7.5 %, 1.03 / 4.00 / 1.89 s | 1.2 %, 1.99 / 2.07 / 1.92 s | 2.5 %, 1.31 / 2.04 / 1.55 s |
| 1.2 s | 2.5 %, 1.43 / 4.40 / 2.29 s | 1.2 %, 2.39 / 2.47 / 2.32 s | 1.2 %, 1.71 / 2.44 / 1.92 s |

Where the eager fired (far15 + eager 30 %, 80 turn ends): 21 at the VAD stop (0.2 s), 27 at 0.5 s, 3 at
1.2 s, none on 29; every eager was held (the harness has no STT, so no transcript mismatch), median lead
to the commit 0.69 s, p90 0.97 s. Four of the five turn ends left to the 3 s backstop had an eager, so
their response comes at 3.2 s instead of 4.0 s. On the 80 pauses, 14 eagers fired before the speaker
resumed and were withdrawn by the VAD start; the 4 commits (5 %) are the same four as in B3. Whole-process
CPU share 24.5 % of one M4 core against 26.5 % in B3 (three replays concurrent in both): the eager test
costs nothing, it reads the `p_end` the commit test already computed.

Reading:

1. **The live pipeline confirms the offline picture.** far15 + eager 30 % keeps the one-stage rule's
   5 % turn splits and 93 % model commits, and brings the mean perceived response time from 1.92 s to
   1.55 s (p50 1.99 → 1.31 s) at a cost of 2.5 % audible false alarms (one more pause than the one-stage
   rule, out of 80) and a generation on 18.8 % of pauses. Against stock Smart Turn: fewer generations
   (18.8 % vs 42.5 %), an eighth of the turn splits (5.0 % vs 42.5 %), a third of the audible false
   alarms (2.5 % vs 7.5 %), the same median response time within 0.3 s (1.31 vs 1.03 s) and half the
   p90 (2.04 vs 4.00 s).
2. **The generation rate on this sample is lower than the offline fit predicted** (18.8 % vs 28.8 %),
   for the reason B3 gave for its FAR: the 80 sampled pauses are shorter than the test set's (median
   0.5 s, 5 above 1.5 s), so fewer of them are still going on at the 0.5 s eager point. The 40 % table
   lands at 38.8 %, close to stock's 42.5 %, and buys another 0.14 s of mean response time.
3. **The lead is what the eager point says it is.** The median eager-to-commit lead is 0.69 s (the
   0.5 → 1.2 s pair) and the p90 0.97 s (0.2 → 1.2 s): with L = 0.8 s the response is ready when the
   commit lands, which is why the p50 response time sits at the commit (1.31 s) rather than at
   commit + L (1.99 s). At L = 1.2 s the lead no longer covers the whole generation and the gain shrinks
   to 0.4 s of mean response time; at L = 0.5 s it is 0.27 s, with the p50 at 1.20 s.
4. **What a product configures.** Three knobs: the commit budget (turn splits), the eager budget
   (LLM calls spent on pauses), and the response latency it actually has. Stock Smart Turn has none of
   them; the one-stage streaming rule has the first; the two-stage rule adds the second, and the audible
   false-alarm rate follows from all three.

Not measured here: the transcript-match withdrawal rate (needs an STT service in the pipeline; an
interim transcript at the 0.2–0.5 s eager point that differs from the final one discards the response
and re-runs inference, which costs a generation but no latency beyond the one-stage rule), and the
LLM-side cost of a withdrawn speculation (Pipecat cancels the inference on `EagerEndOfTurnCancelFrame`).

## Reproduce

```
.venv/bin/python scripts/c1_eager_sweep.py --run runs/b1/knob_ami07_auc --out runs/c1/sweep.md          # ~1 min
.venv/bin/python -m pytest -q tests/test_eager_strategy.py
.venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far15 --strategy eager --eager-rule far15_gen30 \
    --n-complete 80 --n-incomplete 80 --out runs/c1/far15_gen30.csv --quiet                              # ~35 min each
.venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far15 --strategy eager --eager-rule far15_gen40 \
    --n-complete 80 --n-incomplete 80 --out runs/c1/far15_gen40.csv --quiet
.venv/bin/python scripts/b3_replay.py --analyzer streaming --rule far10 --strategy eager --eager-rule far10_gen30 \
    --n-complete 80 --n-incomplete 80 --out runs/c1/far10_gen30.csv --quiet
.venv/bin/python scripts/b3_replay.py --report-two-stage runs/b3/stock.csv runs/b3/streaming_far15.csv runs/b3/streaming_far10.csv runs/c1/*.csv
```
