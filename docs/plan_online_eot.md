# Online end-of-turn detector v2: audit and plan

Status: plan, revised 2026-10-24 after discussion. Companion to `design_streaming_eot.md` (the
architecture proposal of 2026-10-17 and its Phase 0 results). This document records where the
work stands, the order agreed on (commercial audio-only release first, Seamless non-commercial
benchmark and model second, video last from our own recordings), the CPU answer with
measurements, and an experiment ladder with go/no-go gates so that each step is decided by a
number on a held-out split.

Working repo: `smart-turn-multimodal-private` (origin). Nothing goes to the public fork until a
release is ready.

## 1. Audit: what exists today

### 1.1 Done (2026-10-17)

| item | where | result |
|---|---|---|
| int8 CPU model fixed (real calibration, backbone only) | `export_onnx_multimodal.py`, `requantize_modal.py`, HF `-cpu.onnx` | matches fp32 within 0.023; was returning 0.50 on 41/45 inputs |
| audio-only ONNX by graph surgery | `export_audio_only_onnx.py`, local `smart-turn-multimodal-audio.onnx` | 33 MB, bit-exact with the fp32 audio-only path |
| Seamless end-of-turn test set | OpenVideoFDB branch `ws9-endpointing-private`: `ovfdb/endpoint/{rules,build}.py`, `manifest/endpoint_naturalistic_test.jsonl` | 2,300 silences, 133 speakers, 67 conversations, 18 % complete; media local only (CC BY-NC) |
| one-shot evaluation of all models on it | `OpenVideoFDB/runs_nv/smartturn_endpoint_test` | AUC: ours audio 0.801, ours face 0.798, pipecat v3.2 fp32 0.792 / int8 0.787 |
| silence-aware re-query simulation | `eval_silence_requery.py`, `docs/phase0_silence_requery.md` | recall at FAR 10 %: 0.43 → 0.93; p90 latency 3.0 → 1.2 s |
| Pipecat 1.12 analyzer with re-query | `local_smart_turn_multimodal.py` | end to end: p50 0.57 s / p90 1.23 s vs stock 0.24 / 3.20 s |
| Pipecat replay harness | `measure_pipecat_latency.py`, `docs/phase0_pipecat_latency.md` | wall-clock trigger latency per silence; +55–60 ms pipeline overhead per commit |

### 1.2 Not done

Everything from Phase 1 of the design onward: no training tracks, no dev split, no τ-aware
training, no streaming model, no eager head, no upstream Pipecat change. The Phase 0 thresholds
were fitted on the Seamless test set (the 2-fold cross-fit is the honest estimate).

### 1.3 Findings that change the plan

1. **The Modal volume `endpointing` no longer exists** (workspace `susurobo` has only
   `nanochat-cache` and `smart-turn-quant`). It held the PyTorch checkpoints and the Casual
   Conversations copies; HF has ONNX only. Retraining the encoder would start from
   `openai/whisper-tiny` + `smart-turn-data-v3.2`. The plan below avoids needing that until a gate
   says the frozen encoder is not enough.
2. **Casual Conversations is dropped.** Prompted monologues carry no turn-taking signal (the
   evaluation showed the video branch adds nothing); the dataset is not on disk anyway. The
   published multimodal ONNX stays as is.
3. **Video is deferred.** The visual branch will be trained later on our own company Zoom
   recordings (deployment domain, commercially usable, consent handled internally). Until then
   every model is audio-only.
4. **Only Seamless test media is local** (16 GB). Train and dev builds fetch from the public
   mirror (per file: audio 48 MB, VAD 2 KB, transcript small).

## 2. Order of releases and the data behind each

| release | training data | evaluation data | licence of the model |
|---|---|---|---|
| **R1. Commercial audio-only** | `pipecat-ai/smart-turn-data-v3.2-train` (τ-padded, pre-offset cuts) + AMI Meeting Corpus conversational silences | `smart-turn-data-v3.2-test` (regression vs v3.2) + AMI dev/test (latency, false alarms, eager lead) | BSD-2, as Smart Turn |
| **R2. Seamless NC benchmark + NC model** | R1 data + Seamless naturalistic train | Seamless dev (fitting) and test (reported once) | CC BY-NC for the model; benchmark manifests CC BY-NC |
| **R3. Video** (later) | company Zoom recordings, same labelling rules | a held-out slice of the same | internal first, then decide |

**AMI Meeting Corpus** (Edinburgh, ~100 h of meetings) is released under CC BY 4.0, which allows
commercial use with attribution (corpus licence page, checked 2026-10-24; older mirrors such as
OpenSLR 16 carry an outdated NC-SA licence, so the licence file in the download has to be kept
with the data). It ships per-speaker close-talking headset audio and word-level forced
alignments, which is exactly what the WS9 rules need (per-speaker VAD from the alignments, partner
onsets, backchannel filtering). It also has per-participant close-up cameras, usable for R3 if the
Zoom recordings are too few. Meetings are multi-party, not dyadic: "B took the floor" becomes
"anyone else took the floor", and the rules need that one generalisation.

**Seamless** stays evaluation-only for the commercial model and training data for the NC model.
Publishing numbers for a commercial product on NC data is usually treated as research use, but
it is a judgement call to be comfortable with (it was already done for pipecat v3.2).

## 3. What "beating Smart Turn v3.2" means

Pipecat's model is audio-only, Whisper-tiny on 8 s, queried once at the VAD stop, with a 3 s
`stop_secs` fallback in Pipecat. We beat it on numbers that Pipecat users feel, on data they can
reproduce (AMI for R1, Seamless for R2).

| # | metric | why | pipecat v3.2 baseline |
|---|---|---|---|
| P1 | trigger latency from the acoustic offset on completes at matched false-alarm rate on pauses (5 / 10 / 15 %): p50, p90, mean; misses charged 3 s | the number users feel | one-shot on Seamless: p50 0.20 / p90 3.0 / mean 1.54 s at FAR 13 % |
| P2 | recall at matched FAR at fixed τ = 0.2 s (one-shot) | comparability with published numbers | 0.40 at FAR 10 % on Seamless (ours 0.43) |
| P3 | FAR by pause-length bin (< 0.5, 0.5–1.5, ≥ 1.5 s) at the P1 operating point | the ≥ 1.5 s bin is where audio alone fails today | 25–28 % on ≥ 1.5 s |
| P4 | eager lead: ms before the acoustic offset at which `p_eager` fired on completes; fraction withdrawn | the anticipatory part | n/a |
| P5 | CPU: ms per hop single-threaded, share of one core at 10 Hz, decision-path burst, file size | deployability | 20 ms per query (M4), 8 MB int8 |
| P6 | accuracy per language on `smart-turn-data-v3.2-test` | regression guard for R1 | v3.2's published table |

Every table also contains **pipecat v3.2 under our re-query rule** with thresholds fitted on the
dev split, so the model has to win on top of the rule. Thresholds are never fitted on a test split
again.

Reference points from Seamless: humans start their turn at a median of 0.80 s after the offset,
15 % within 0.2 s and 33 % within 0.5 s. Ending every turn at 1.2 s of silence with no model gives
recall 1.00 at FAR 12 %; a model earns its keep by committing earlier at the same FAR and by
holding FAR down on long pauses.

## 4. The CPU question

**Yes, with one condition: the pretrained encoder stays off the per-hop path.** Measured
2026-10-23 (Apple M4, onnxruntime 1.31, audio-only graph, 8 s input):

| configuration | ms per call |
|---|---|
| Whisper-tiny encoder + head, 1 thread | 20.3 (p90 20.6) |
| same, 4 threads | 10.1 |

Pipecat's own figure for the same encoder architecture (Smart Turn v3, int8) on a Pipecat Cloud 1x
instance is about 65 ms, so a cloud vCPU is budgeted at roughly 3× the M4 single-thread cost.

| option | per-hop work at 10 Hz | M4 core | cloud vCPU (×3) | verdict |
|---|---|---|---|---|
| A. full encoder every 100 ms | 20 ms | 20 % | ~60 % | no |
| B. encoder every 320 ms on 8 s, cached pooled vector + per-hop GRU step (< 0.5 ms) | 20 ms / 3.2 + 0.5 ms | ~7 % | ~20 % | **yes, the design** |
| C. as B with a 4 s window | ~10 ms / 3.2 | ~4 % | ~10 % | yes, if lexical context survives (ablation) |
| D. fully causal student (conv + GRU at 100 Hz, no Whisper), distilled from B | ~1–2 ms per hop | ~2 % | ~5 % | later, for edge boxes |
| plus incremental log-mel (ring buffer, each column computed once) | ~1 ms per 100 ms | 1 % | 3 % | replaces the 30 ms feature extractor per query |

Both cadences run on the CPU, in one process, inside one analyzer object. The slow path (encoder
refresh) runs on the analyzer's thread and hands one 384-float vector to the fast path; the fast
path (mel columns, τ, GRU step, heads) runs on the Pipecat frame callback in about 1 ms. The
encoder vector is up to 320 ms stale; training samples it with random staleness in [0, 320] ms so
the model expects that. Exported as one stateful ONNX (GRU state in and out) plus one encoder
ONNX. No GPU anywhere.

Supporting evidence: the real-time VAP model (Inoue, Jiang, Ekstedt, Kawahara, Skantze, 2024)
keeps a 1 s transformer context with no accuracy loss against 20 s (balanced accuracy 76.2 % vs
74.2 %) and still costs 14.6 ms per 50 Hz frame on a Xeon Gold 6128 because it recomputes a
transformer per frame; a GRU step is two orders of magnitude cheaper, so our only expensive piece
is the pretrained encoder and option B amortises it.

## 5. Training levels, cheapest first

1. **Frozen encoder, train the new small parts only.** Tap the pooled 384-d vector out of the
   existing audio-only ONNX graph (or run the encoder once per training window and cache). Train
   the GRU, the fusion layer and the two heads (~100–200k parameters) on the sampled windows. Minutes
   per run on a laptop or a small GPU; no checkpoint recovery needed. **Start here.**
2. **Fine-tune the encoder** at a low learning rate if level 1 plateaus. Needs the audio baseline
   retrain from whisper-tiny on smart-turn-data first (upstream `train.py`, a few GPU hours on
   Modal, checkpoint saved to a new volume and a private HF repo).
3. **Distil a causal student** (option D) only when edge hardware demands it.

## 6. Experiment ladder for R1 (commercial, audio-only)

Each stage has a gate on a **dev** split. A stage that fails its gate ends that branch.

### Stage A. Data

**A1. smart-turn-data with τ and pre-offset variants** (done 2026-10-24, see `docs/a1_smart_turn_data_offsets.md`). For every clip find the acoustic offset
(energy / Silero on the clip). Build windows ending at τ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8,
2.5} s after it by padding with the clip's own noise floor (and, for the v3.2 noisy subset, its
noise track), label unchanged. Build pre-offset windows at δ ∈ (0, 0.8] s before the offset of
complete clips (label `p_end` 0, `p_eager` 1 if δ ≤ 0.4 s) and of incomplete clips (both 0). This
teaches "trailing silence after an incomplete utterance is still incomplete" and the eager head,
but not the hazard of a real thinking pause that resumes, because the clips stop at the cut.

**A2. AMI conversational tracks** (dev/test built 2026-10-24, train building; see `docs/a2_ami.md`). Download headset audio and the NXT word alignments; derive
per-speaker VAD from the alignments (gaps ≥ 0.2 s); port the WS9 rules with the multi-party
generalisation (floor holder vs *any* other participant); write per-meeting tracks (`audio_<spk>.wav`
16 kHz, `vad.json`, `events.jsonl` including excluded silences with reasons). Split by meeting into
train / dev / test (AMI has official splits; keep speakers disjoint). Gate: ≥ 5,000 labelled
silences in dev+test from ≥ 40 speakers, both classes ≥ 15 %.

**A3. Sampler** (done 2026-10-24 as `scripts/b1_items.py` + cached features; see `docs/b1_results.md`). On-the-fly windows from tracks: complete at τ, incomplete at τ ≤ gap, pre-offset
δ, random mid-turn. Class weighting as `train.py`.

### Stage B. Model

**B1. τ-aware heads on the frozen encoder** (level 1), trained on A1 + A2. *Done 2026-10-24, `docs/b1_results.md`: AMI test AUC 0.67 → 0.76 at the stop and 0.58 → 0.73 at 1.2 s; recall at FAR 10 % 0.75 (gate met), FAR on ≥ 1.5 s pauses 46 % (gate missed), eager AUC 0.67–0.74 on AMI (missed), std test unchanged.* Inputs: cached encoder
vector, GRU over the last 1 s of mel, τ, speaking flag. Gates on AMI dev: AUC at τ = 1.2 s ≥ AUC
at τ = 0.2 s (today the score degrades with τ); recall at FAR 10 % under the re-query rule ≥ 0.75;
FAR on pauses ≥ 1.5 s ≤ 40 % at that operating point; P6 within 1 point of v3.2.

**B2. Streaming export and cost** (done 2026-10-24, `docs/b2_streaming.md`: 9.1 % of an M4 core, 0.7 ms per hop, 22 ms per refresh, streaming AUC within 0.01 of offline with the silence-clock refresh). Stateful ONNX for the fast path, encoder ONNX for the slow
path, incremental mel. Gates: per-hop ≤ 1 ms, encoder refresh ≤ 25 ms single-threaded on M4,
decision-path burst ≤ 5 ms, `p_eager` AUC ≥ 0.75 on AMI dev.

**B3. Rules and the full streaming evaluation** (done 2026-10-24, `docs/b3_pipecat.md`: in Pipecat, 15 % budget gives 5 % false alarms and p90 1.27 s vs stock 42 % and 3.20 s). Fit θ(τ) and θ_eager on AMI dev; produce P1–P6
on AMI test; replay through Pipecat with `measure_pipecat_latency.py` extended to the streaming
analyzer. Ablations: 8 s vs 4 s context; with and without AMI; with and without τ; with and
without the eager head. If B1 misses its gates, go to level 2 (encoder fine-tune) and repeat.

### Stage C. Pipecat and release R1

Streaming analyzer (`append_audio` drives everything, `analyze_end_of_turn` returns the current
verdict); the eager strategy as an upstream PR (`TurnAnalyzerEagerUserTurnStopStrategy` feeding
`EagerTranscriptionFrame` / `EagerEndOfTurnCancelFrame`); README, HF upload of the two ONNX files,
AMI evaluation script and manifests.

*C1 done 2026-10-24 (`docs/c1_eager.md`): the strategy exists in `eot/pipecat_eager_strategy.py` and drives
Pipecat 1.14's speculation path directly (`UserTurnSpeculation` → `SpeculationGate`), no eager transcription
frames needed; the eager signal is a second p_end table on the silence clock, fitted under a budget of
generations spent on pauses. C2 (2026-10-25, `docs/c2_release.md`): the encoder refresh runs on a worker thread
(pipeline-task burst 45 → 4 ms p99, decisions and CPU unchanged); the R1 release is staged for the Hub (two ONNX
files, model card, rule tables, AMI evaluation script, CC BY 4.0 attribution), upload and public-fork push pending
the user's decision. The upstream PR is in its own session.*

## 7. R2: Seamless benchmark and NC model

1. Build the Seamless dev split (`ovfdb endpoint build --split dev --pairs 100`, ≈ 3,400
   silences). Refit the Phase 0 tables on it and re-report test, which also makes the Phase 0
   analyzer honest and shippable as an interim.
2. Evaluate the R1 model on Seamless dev/test (P1–P5) against pipecat v3.2.
3. Seamless train tracks (1,000 pairs ≈ 34k silences, 96 GB fetch, 10 GB on disk, built on Modal
   CPU) and fine-tune the R1 heads (level 1) or encoder (level 2) → the NC model.
4. Release: manifests + evaluation script as the public benchmark, NC model as a separate HF file.

## 8. R3: video from company recordings (later)

Label the Zoom recordings with the same rules (per-speaker audio tracks are needed; Zoom's
separate-audio-per-participant option gives them). Face features via a landmarker at 10 fps, a
GRU over 3 s, fused in the slot the R1 architecture reserves. Gate: FAR on pauses ≥ 1.5 s drops
by ≥ 5 points at equal recall, or `p_eager` AUC rises by ≥ 0.02, on a held-out slice. The
landmarker's per-frame CPU cost is measured before any of this (model file not local yet).

## 9. Order of work and budget

| step | depends on | effort | Modal |
|---|---|---|---|
| A1 τ/pre-offset variants of smart-turn-data | – | 2 days | – (CPU, could be local) |
| A2 AMI download, rules port, tracks, splits | licence file check | 3–4 days | – |
| B1 frozen-encoder heads + gates | A1, A2 | 3–4 days | small GPU, < $20 |
| B2 streaming export + cost | B1 | 3–4 days | – |
| B3 rules, eval tables, Pipecat replay | B2 | 3 days | – |
| level 2 encoder fine-tune (only if B1 fails) | B1 | 2–3 days | $50–150 |
| C analyzer, upstream PR, R1 release | B3 | 1 week | – |
| R2 Seamless dev split, eval, train tracks, NC fine-tune, benchmark release | C | 1–2 weeks | $20–60 CPU, < $100 GPU |

About three to four weeks to R1, one to two more to R2. Modal spend for the audio path is well
under $300 unless the encoder fine-tune is needed.

## 10. Risks

- **Labels by outcome, not intent**: a speaker cut off counts as complete. Mitigation: the eager
  head takes only clean positives (next speaker within 1 s); report P1 on that slice too.
- **AMI is meetings, not dyadic calls**: overlap and multi-party floor dynamics. Mitigation: the
  rules exclude collisions; report numbers per overlap condition; Seamless (dyadic) is the second
  check in R2.
- **Synthetic τ padding** on smart-turn-data differs from real room tone. Mitigation: pad with the
  clip's own noise floor; AMI supplies real pauses.
- **Licence hygiene**: R1 never sees Seamless; manifests and tracks stay out of git and the public
  fork; AMI attribution and licence file shipped with the data card.
- **Encoder staleness** (≤ 320 ms) could hurt the eager head. If B2's `p_eager` misses its gate,
  try a 160 ms refresh before option D.
- **Lost artifacts**: every run writes its checkpoint to a volume and a private HF repo; commands
  recorded in `README_PRIVATE.md`. Raw data (smart-turn-data shards, AMI annotations and headsets) is
  fetched directly into the Modal volume `smart-turn-v2` by `modal_data.py`; derived artifacts
  (`data/a1`, `data/a2`) are pushed from local with `modal volume put`. Training runs there.

## 11. Decisions taken

- 2026-10-24: commercial audio-only release first (smart-turn-data + AMI), Seamless NC second,
  video last from company Zoom recordings. Casual Conversations dropped. Frozen-encoder training
  first; encoder fine-tune only if a gate fails.
- 2026-10-24: Stage A (A1 smart-turn-data windows, A2 AMI events, A3 items + feature cache) and B1 (heads
  on the frozen encoder) done; results in `docs/results_so_far.md` and `docs/b1_results.md`. Raw data on
  Modal volume `smart-turn-v2`. B1 meets 2 of 4 gates on AMI; next: cheap training knobs, then B2.
- 2026-10-24: B1 training knobs (lr, selection, AMI weight, long-pause negatives, regularisation) tried;
  recipe set to lr 1e-3 / 8 ep / select by dev AUC / AMI 0.7; no knob closes the long-pause FAR or eager
  gates on AMI. Next: B2 streaming export; level 2 and Seamless for the gates.
- 2026-10-24: B2 done. Encoder refresh runs on the silence clock (VAD stop + trained tau grid), not a free timer:
  a timer alone cost 0.04-0.06 AUC from stale vectors. Seamless R2 + AV oracle started in its own session (branch
  `seamless-r2`). Next here: B3 Pipecat analyzer and end-to-end replay.
- 2026-10-24: B3 done (streaming analyzer in Pipecat; end-to-end tables). R2 done on branch `seamless-r2`: NC heads
  0.853 AUC on Seamless test; AV oracle misses its gate, video stops. Inference path is numpy + onnxruntime only.
  `seamless-r2` merged. Next: Stage C (eager strategy upstream, production threading, release files).
- 2026-10-24: C1 done (`docs/c1_eager.md`). The eager signal is not the p_eager head (misfires ~22×/min of speech on
  AMI) but a second, looser p_end table on the same silence-clock points, driving Pipecat's speculation path through
  `TurnAnalyzerEagerUserTurnStopStrategy` (`eot/pipecat_eager_strategy.py`); eager tables fitted on AMI dev under a
  budget of generations spent on pauses. Next: C2, the upstream PR (`BaseTurnAnalyzer.pop_eager()` + the strategy),
  the encoder refresh on a worker thread, HF upload and the AMI evaluation script.
- 2026-10-25: C2 threading done (`docs/c2_release.md`). The encoder runs on a worker thread; a silence-clock decision
  waits up to 80 ms for the vector of its own window, the VAD-stop decision awaits its refresh for 50 ms in
  `analyze_end_of_turn` (a 20 ms wait misses the 22 ms encoder every time, measured). Streaming AUC equal to inline
  within 0.001 (gate met); burst on the pipeline task 45 → 4–5 ms p99, CPU share and decisions unchanged.
  R1 staged in `runs/c2/hf_stage/`; the release evaluation script reproduces B1 from public AMI audio. Next: the
  user's call on the Hub repository name, the upload and the public-fork push; the upstream PR session.
