# Online end-of-turn v2: results so far

Updated 2026-10-25. One page; details and reproduction commands in the linked stage documents.
Plan and gates: `plan_online_eot.md`. Architecture proposal: `design_streaming_eot.md`.

## Where we are

| stage | state | document |
|---|---|---|
| Phase 0: audio-only export, silence-aware re-query in Pipecat | done 2026-10-17 | `design_streaming_eot.md` §9, `phase0_*.md` |
| A1: smart-turn-data offsets, τ-padded and pre-offset windows | done 2026-10-24 | `a1_smart_turn_data_offsets.md` |
| A2: AMI Meeting Corpus end-of-turn events (CC BY 4.0) | done 2026-10-24 | `a2_ami.md` |
| A3 + B1: feature cache, τ-aware heads on a frozen encoder | done 2026-10-24 | `b1_results.md` |
| B2: streaming inference on the CPU | done 2026-10-24 | `b2_streaming.md` |
| B3: streaming analyzer in Pipecat, end-to-end replay | done 2026-10-24 | `b3_pipecat.md` |
| R2: Seamless dev split, NC model, AV oracle | done 2026-10-24, merged | `r2_seamless.md` |
| C1: two-stage eager-then-commit rule on Pipecat's speculation path | done 2026-10-24 | `c1_eager.md` |
| C2: encoder refresh on a worker thread, R1 release files staged | done 2026-10-25 | `c2_release.md` |
| C2 upstream Pipecat PR (`pop_eager()` + eager strategy) | on the fork branch `maxipesfix/pipecat:turn-analyzer-eager-stop-strategy`; upstream PR #6374 opened and closed, reopening is the user's call | – |
| C3: official Smart Turn benchmark, calibration, per-language behaviour | done 2026-10-25 | `c3_benchmark.md` |
| R1 release: `smart-turn-streaming` | released 2026-10-25: code at github.com/OpenHRIai/smart-turn-streaming (fresh repo, not a fork; encoder pinned in `UPSTREAM.md`), models at huggingface.co/OpenHRI/smart-turn-streaming | `c2_release.md`, figures in `figures/` |
| R3: video from company recordings | later | – |

Data: Modal volume `smart-turn-v2` (pipecat parquet shards, AMI annotations and 682 headsets,
derived offsets, events, items, 5 GB feature cache); same layout under `data/` locally.

## Headline numbers

**Phase 0, Seamless test (2,300 silences), the existing model with a rule change only.**
Re-querying at 0.6 s and 1.2 s of silence instead of waiting for the 3 s timeout: recall at
FAR 10 % 0.43 → 0.93, p90 trigger latency 3.0 s → 1.2 s offline; 3.20 s → 1.23 s measured end to
end in Pipecat. Thresholds were fitted on the test set; to be refit on a dev split before shipping.

**A1, the training data problem.** 79 % of pipecat's clips end within 0.2 s of the speech,
35 % exactly at it, 82 % are TTS. The published models never saw silence, which is why their
false-alarm rate on incomplete clips rises from 5 % at the offset to 19 % after 2.5 s of
synthesised silence. Windows cut 0.3 s before the end of speech already separate complete from
incomplete at AUC 0.90: the anticipatory signal is in the encoder.

**A2, conversational data with a commercial licence.** AMI: 52,569 labelled silences (train
43,339 / dev 4,415 / test 4,815; 13 % complete), VAD from the headset audio gated by the
transcript, pause distribution matching Seamless (median 0.5–0.6 s). Smart Turn v3.2 scores AUC
0.67 at the VAD stop on this set, against 0.79 on Seamless: meetings are harder.

**B1, the first trained model** (heads on pipecat's frozen encoder, trained on both sources,
3 seeds; AMI test):

| | AUC at the stop | AUC at 1.2 s of silence | recall at FAR 10 % (re-query rule) | stock rule θ 0.5: recall at FAR |
|---|---|---|---|---|
| Smart Turn v3.2, frozen | 0.67 | 0.58 | 0.70 at 10.6 % | 0.51 at 30 % |
| v3.2 encoder + our heads | 0.76 ± 0.01 | 0.73 ± 0.00 | 0.75 ± 0.02 at 8.0 % | 0.50–0.58 at 17–21 % |
| our encoder + our heads | 0.78 ± 0.01 | 0.72 ± 0.01 | 0.73 ± 0.01 at 7.7 % | 0.52–0.56 at 18–19 % |

No regression on pipecat's own test set (AUC 0.977 vs 0.976 at the stop; 0.968 vs 0.955 at 2.5 s
of silence). Eager head: AUC 0.84 at 0.4 s before the end of speech on pipecat data (frozen 0.76),
0.67 on AMI.

**B2, the streaming detector.** One object on one CPU thread: incremental Whisper features
(exact to 0.003 on real audio), the heads every 100 ms (0.7 ms), the encoder every 320 ms during
speech and at the trained τ grid during silence (22 ms). 9.1 % of an M4 core, 33 MB of models,
streaming AUC within 0.01 of the offline evaluation. A free-running refresh timer would have cost
0.04–0.06 AUC (stale encoder vectors); the silence-clock schedule removes that at the same budget.

**B3, end to end in Pipecat** (160 AMI test silences, real-time replay, M4): stock Smart Turn v3.2
fires on 42 % of mid-turn pauses and leaves 22 of 80 turn ends to the 3 s timeout (p90 3.20 s); the
streaming analyzer at the 15 % budget fires on 5 %, commits 93 % of turn ends itself and has p90
1.27 s, at a median of 1.19 s instead of 0.23 s. CPU: +18 points of one M4 core over the stock
pipeline (26 % vs 9 % whole process); memory +114 MB vs +76 MB for the analyzer.

**R2, Seamless** (separate session, merged from `seamless-r2`): dyadic data is easier for every model
(frozen v3.2 AUC 0.79 at the stop vs 0.67 on AMI); NC heads trained with Seamless reach 0.853 ± 0.006
vs 0.791 frozen, recall 0.88 vs 0.83 at 8 % FAR with mean latency 1.10 vs 1.32 s; AMI and pipecat's
test set unchanged. The audio-visual oracle with Meta's own face features misses its gate (every
number moves the right way but by less than the bar): the NC release is audio-only.

**C1, the eager stage.** Re-scoring the B3 replays with 0.8 s of response latency: only 7.5 % of the stock
rule's false alarms are audible (the bot speaks before the user resumes); the other 35 points are wasted
generations and split turns. Pipecat's speculation path (`UserTurnSpeculation`, `SpeculationGate`) lets a
turn analyzer separate the two costs: a looser eager table on the same decision points starts the LLM, the
dev-fitted commit table ends the turn and releases the held response, the VAD start discards it.
Offline on AMI test (511 turn ends / 4,331 pauses), at the stock rule's generation cost (29 % of pauses)
the far15 commit + 30 % eager rule splits 13 % of pauses instead of 26 %, is heard on 8.7 % instead of
9.5 %, and brings the mean perceived response time from 2.01 s to 1.57 s (p90 3.8 → 3.0 s). The 0.5 s
eager point carries the gain; the speech-time eager head would misfire ~22 times per minute and stays off.
End to end in Pipecat (the B3 replays, 80 + 80 silences): far15 + eager 30 % keeps 5 % turn splits and
93 % model commits, spends a generation on 19 % of pauses (stock 42.5 %), is heard on 2.5 % (stock 7.5 %)
and answers in 1.31 / 2.04 / 1.55 s (p50 / p90 / mean) against stock's 1.03 / 4.00 / 1.89 s and the
one-stage rule's 1.99 / 2.07 / 1.92 s; median eager-to-commit lead 0.69 s; no extra CPU.

**C2, production threading and the release.** The encoder refresh runs on one worker thread; a silence-clock
decision waits up to 80 ms for the vector of its own window (mel and τ snapshotted with it), the 0.2 s decision
at the VAD stop awaits its refresh for 50 ms on the event loop. Streaming AUC equals the inline detector's within
0.001 (B2's gate met: 0.765 / 0.760 / 0.872 vs offline 0.774 / 0.767 / 0.877). In Pipecat the pipeline-task burst
falls from 45 ms (p99) to 4–5 ms, the 0.5 / 1.2 s decisions cost 0.2 ms and land 40–60 ms after the point, CPU
share (16–19 % of an M4 core, whole process) and decisions are unchanged. Deciding at the point on a vector
computed 60 ms earlier would cost up to 0.035 AUC at 1.2 s, and a free-running hop must never decide a rule
offset (it can precede the grid hop on the previous vector). At real-time pace on an idle laptop the encoder costs 46 ms, not 22 ms (power management),
which is why a one-frame wait at the VAD stop misses it. Release R1 is staged (two ONNX files, model card, rule
tables, AMI evaluation script that reproduces B1 within 0.004 AUC, CC BY 4.0 attribution); not uploaded.

**C3, the official benchmark and languages.** On pipecat's own `benchmark.py` (full smart-turn-data v3.2
test, 31,527 clips, threshold 0.5) the heads, Platt-calibrated on the training split and fused into one
graph with the stock interface, score 93.2 % accuracy against Smart Turn v3.2's 93.7 % (FPR 4.2 vs
3.5 %), same ranking (AUC 0.975 vs 0.978), same per-language pattern within noise. The benchmark has no
pauses, so it is the regression guard, not the result. Per language on τ-padded clips the silence
awareness transfers to all 23 languages: at a matched 5 % false-alarm rate at the clip end, false alarms
after 2.5 s of silence are 9.4 % for the heads against 22.5 % for the frozen model, lower in every
language. What is still English-only is the conversational hazard (a pause that resumes).

## What we learned

1. The 3 s timeout, not inference time, is the latency problem; a rule change alone removes most
   of it (Phase 0).
2. Real conversational pauses are what teach silence awareness. τ-padding isolated utterances does
   not transfer to meetings (std-only heads = frozen model on AMI); AMI data alone does, but breaks
   the pipecat domain (AUC 0.98 → 0.66), so the commercial model trains on both.
3. The GRU over the last second of mel is the component that carries silence awareness (AUC at
   1.2 s 0.74 with it, 0.66 without); the explicit τ number adds nothing. This is the branch the
   streaming design runs per 100 ms hop at under 1 ms.
4. CPU is not the obstacle, measured end to end in B2: 9.1 % of an M4 core single-threaded for the
   whole detector, 0.7 ms per decision, 22 ms per encoder refresh; a cloud vCPU is ~3× slower.
5. Meetings have a ceiling set by the labels: waiting 1.2 s with no model gives recall 1.0 at
   FAR 18 %, and below that budget a quarter of turn ends still time out for every model. Seamless
   shows the ceiling is mostly the data: the same frozen encoder with heads reaches 0.85 there.
6. In the live pipeline the stock rule's failure is false alarms (42 % on AMI pauses), and the
   false-alarm budget is the product knob stock Smart Turn does not have.
7. A false alarm has two costs, a wasted generation and a split turn, and only becomes audible if the
   bot speaks before the user resumes. Pipecat's speculation path separates them: the eager budget buys
   response time, the commit budget protects the turn, and the response latency decides how much of a
   commit error is ever heard (C1).
8. On a laptop the cost of sparse real-time work is twice its tight-loop cost (46 vs 22 ms per encoder
   refresh): power management, not the code. Budgets measured in a tight loop are optimistic by that factor
   for a lightly loaded machine; a worker thread removes the burst from the frame path but not that factor (C2).

## Gates and open items

- B1 gates: recall ≥ 0.75 at FAR 10 % met (0.75); std regression met; FAR on pauses ≥ 1.5 s
  46 % against a 40 % target (frozen: 61 %), eager AUC on AMI 0.67–0.74 against 0.75: missed.
- Cheap knobs tried (14 runs, `b1_results.md` "Training knobs"): learning rate, early stopping on
  dev AUC, AMI weight, long-pause negatives, dropout / weight decay. Stop-AUC improves to 0.785
  (AMI weight 0.7, select by dev AUC); the rule metrics and the two missed gates do not move
  beyond seed noise. Level 1 has plateaued on AMI.
- Encoder fine-tuning (level 2) needs the baseline retrain from whisper-tiny: the PyTorch
  checkpoints were lost with the old Modal volume.
- Seamless dev split still to build; Phase 0 thresholds to be refit there.
