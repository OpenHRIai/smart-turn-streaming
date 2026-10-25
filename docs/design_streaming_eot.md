# Design: a streaming, anticipatory Smart Turn Multimodal (v2)

Status: proposal, 2026-10-17. Based on the int8 re-export (commit f0f65d4), the NVIDIA probe
(`OpenVideoFDB/runs_nv/smartturn_probe`) and the Seamless end-of-turn set
(`OpenVideoFDB` branch `ws9-endpointing-private`, `runs_nv/smartturn_endpoint_test`).

## 1. What the evaluations say

### 1.1 The fix that landed

The published int8 CPU model was calibrated on Gaussian noise, so its final logit could only
represent [0, 2.41] and 41 of 45 real inputs collapsed to 0.50. The re-export calibrates on real
`(feats, pix)` pairs, quantises only the R3D-18 backbone and keeps the Whisper encoder and head in
fp32. It now matches fp32 within 0.023 and runs 245–345 ms per call on CPU instead of 540–640 ms.
That is a correctness fix, not an accuracy gain: the CPU model is now as good as the GPU model,
and the GPU model is the thing that needs to improve.

### 1.2 Seamless end-of-turn set (2,300 silences, 133 speakers, 18 % complete)

| model | AUC | recall @ thr 0.5 | false alarms on pauses | recall @ FAR 10 % |
|---|---|---|---|---|
| multimodal fp32, audio only | 0.801 | 0.52 | 13 % | 0.43 |
| multimodal fp32, face video | 0.798 | 0.38 | 9 % | 0.42 |
| pipecat Smart Turn v3.2 fp32 | 0.792 | 0.51 | 17 % | 0.40 |

Three findings drive the design:

1. **Video adds nothing** (ΔAUC −0.002, CI centred on zero; identical recall at matched false-alarm
   rate). The face input only lowers the model's confidence. The R3D-18 branch is not extracting a
   turn-taking signal.
2. **Half the real turn ends are missed**, and a miss in Pipecat is not "slightly late": the
   analyzer is asked once at the VAD stop, answers INCOMPLETE, and the turn then ends on the
   `stop_secs` timeout (3 s by default). So roughly half of all bot responses in a face-to-face
   setting are delayed by about 3 s. That is the dominant latency problem, far larger than inference
   time.
3. **False alarms grow with pause length** (10 % under 0.5 s, 16 % at 0.5–1.5 s, 29 % at ≥ 1.5 s).
   Thinking pauses look like turn ends to audio alone. In dyadic data these pauses come with gaze
   aversion, which is exactly the cue the video branch was supposed to learn.

Two more numbers from the same manifest matter for an anticipatory design. The partner starts
speaking at a median of 0.80 s after the speaker's offset, 15 % within 0.2 s and 33 % within 0.5 s:
humans decide partly before the silence and partly from the silence. And a baseline that scores
"complete" when the reference transcript ends in `.`, `?` or `!` reaches AUC 0.78 on its own, so
most of the audio model's 0.80 is lexical and syntactic completeness, not prosody or timing.
(The reference punctuation was produced with knowledge of the whole utterance, so that baseline
is an upper bound on what a text cue alone gives, not something deployable as is.)

### 1.3 Why the current video branch cannot work

- **Training data has no turn-taking in it.** Casual Conversations are prompted monologues to a
  camera. Labels came from segment position within a recording, so nothing in the video correlates
  with yielding or holding the floor to another person.
- **Wrong features.** R3D-18 is a Kinetics action classifier on 112 px frames. Gaze direction, a
  mouth closing, an in-breath or a small nod are a few pixels at that resolution and are not what
  Kinetics features encode.
- **Wrong window.** 32 frames at 30 fps is about 1.07 s ending 0.2 s after the VAD stop. That window
  is mostly the last word. The informative behaviour (gaze return, mouth closing, head stilling)
  happens around the offset and during the first second of silence.
- **Wrong cost.** The branch is 95 % of CPU latency (628 ms fp32, 165 ms int8) and runs on a zero
  tensor even in audio-only mode.

### 1.4 Why the pipeline cannot be fast

Today the decision happens only after `VAD stop_secs` (0.2 s) + mel features (34 ms) + video
preprocessing (63 ms) + inference (245–345 ms int8): about 0.55–0.65 s after the acoustic offset
when the model says COMPLETE, and about 3 s when it does not. The model never sees how long the
silence has lasted (the input always ends 0.2 s after the stop), and it is never re-asked as the
silence grows. Silence duration is the single strongest cue a human uses to resolve ambiguous
cases, and the model has no access to it.

## 2. Goals for v2

| goal | target | how it is measured |
|---|---|---|
| Not triggered by VAD silence | model runs continuously at 10 Hz on audio + face features | streaming eval on the Seamless set |
| Predict before silence | an *eager* end-of-turn signal 0–400 ms before the acoustic offset on confident turn ends, withdrawn if the user continues | detection latency relative to the speaker's VAD offset |
| Fewer 3 s timeouts | recall ≥ 0.75 at 10 % false alarms on pauses, measured with the silence-aware decision rule, against 0.43 today | recall at matched FAR; p50 / p90 of trigger latency including timeouts |
| Video that helps | false-alarm rate on pauses ≥ 1.5 s halved (29 % → ≤ 15 %) with face features vs audio only | per-pause-length slices |
| CPU | ≤ 30 % of one core continuous, no burst over 40 ms on the decision path, model files ≤ 40 MB | ORT profiling on the Pipecat thread |

## 3. Architecture

```
 audio 16 kHz ──► mel (incremental, 10 ms hop) ──► Whisper-tiny encoder on the last 8 s,
                                                   run every 320 ms, pooled 384-d context  ─┐
                                                                                            │
 audio 16 kHz ──► causal conv + GRU over the last 1 s of mel (per-hop, stateful)  ─────────┤
                                                                                            │
 camera 10 fps ──► face landmarker (blendshapes 52 + head pose 6 + gaze proxy 4) ──► GRU ──┤
                                                                                            ├─► fusion MLP
 VAD state ──► silence duration τ (ms since offset), speech-in-turn so far, turn length ───┘      │
                                                                                                   ▼
                                                       two heads, every 100 ms:
                                                       p_eager = P(turn ends within 400 ms | now, still speaking)
                                                       p_end   = P(turn has ended | audio, face, τ)
```

### 3.1 Audio: keep the pretrained encoder, stop running it on the decision path

The Whisper-tiny encoder is where the lexical and syntactic signal comes from (it is the reason
the multimodal audio path edges pipecat v3.2), so keep it. It costs 26 ms fp32 on CPU for 8 s, and
the int8 experiments showed it must stay fp32. Run it on a timer (every 320 ms) over the ring
buffer and cache the attention-pooled 384-d vector. Everything that has to be fresh at 100 ms
granularity (the last 1 s of mel, the silence counter, the face features) goes through small
stateful recurrent layers that cost well under 1 ms per hop. The decision path at any hop is then
"GRU step + MLP", not "encoder on 8 s".

The encoder is retrained on windows that end at arbitrary points, not only at VAD stops (see §4),
so its embedding is meaningful mid-utterance. Later option: distil Whisper-tiny into a causal
student so the context is also per-hop; not needed for v2.

### 3.2 Video: face features, not pixels

Replace R3D-18 with a per-frame face landmarker (MediaPipe Face Landmarker: 52 blendshapes, head
transform, ~5–8 ms per frame on CPU, Apache-2.0; an ONNX equivalent is acceptable) sampled at
10 fps from the camera stream. Derived inputs per frame:

- gaze proxy: eyeLookIn/Out/Up/Down left/right blendshapes and head yaw/pitch (gaze aversion vs
  gaze at the partner/camera);
- mouth: jawOpen, mouthClose, mouthFunnel/Pucker (in-breath, about-to-speak, closed-and-resting);
- head motion: yaw/pitch/roll deltas (nods, head stilling at a turn end);
- brow and smile channels as given.

A GRU over the last 3 s (30 steps × ~64 features) produces a 64-d visual state. Compared with
pixels this is speaker-, lighting- and resolution-invariant, needs no face-crop video pipeline
in the analyzer, and the training data for it can be stored as 10 fps feature tracks (a few KB
per minute) rather than per-event face videos. Seamless Interaction ships its own 30 Hz movement
features (FAUValue, gaze_encodings, head_encodings, emotion valence/arousal); those cannot be
computed at inference time, but they are useful as a consistency check on the open extractor
during data building.

Explicit non-goal for v2: learned visual embeddings from pixels. Revisit only if the feature
model shows a clear visual gain and a gap to a pixel model is suspected.

### 3.3 Silence as an input and a time-varying decision rule

Feed the model the elapsed silence τ (0 while speaking) and the turn-so-far statistics. Train
with windows ending at τ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5} s after the offset for each
labelled silence, so `p_end(τ)` is a calibrated hazard that rises as the silence grows for genuine
turn ends and stays low for thinking pauses with averted gaze. The deployed rule is

```
COMPLETE  when p_end(τ) ≥ θ(τ)          evaluated every 100 ms during silence
```

with θ(τ) a short monotone-decreasing table tuned on the Seamless test set for a target
false-alarm rate per pause-length bin (e.g. 5 % under 0.5 s, 10 % at 0.5–1.5 s, 15 % above). This
replaces the single 0.5 threshold and the 3 s fallback with something that commits early when it
is sure and keeps waiting when the face says "thinking".

### 3.4 The eager head

`p_eager` is trained on frames while the speaker is still talking, label 1 if the current turn
ends (complete) within the next 400 ms. It fires an *eager end of turn* when it crosses θ_eager
(set for ≤ 10 % withdrawn predictions). Pipecat 1.12 already has the machinery for this:
`EagerTranscriptionFrame` / `EagerEndOfTurnCancelFrame`, `EagerUserTurnStopStrategy` and the
LLM service's `SpeculationGate`, which holds the speculative response until
`UserStoppedSpeakingFrame` confirms it and discards it on cancel. Today only STT services drive
it; §5 describes the small extension needed for a turn analyzer to drive it.

What the eager signal buys: STT finalisation and the LLM's prefill and first tokens run during
the user's last word and the first few hundred milliseconds of silence instead of after the
committed decision. The committed decision still comes from `p_end`; the eager path never writes
to the context.

### 3.5 Budget

| component | cost | cadence | share of one core |
|---|---|---|---|
| incremental mel | ~1 ms per 100 ms of audio | continuous | ~1 % |
| Whisper-tiny encoder, 8 s, fp32 | 26 ms | every 320 ms | ~8 % |
| face landmarker | 5–8 ms per frame | 10 fps | 5–8 % |
| audio GRU + visual GRU + fusion + heads | < 1 ms | every 100 ms | < 1 % |
| **total** | | | **~15–20 %** |

Model files: encoder ~32 MB fp32, landmarker ~4 MB, heads < 1 MB. No int8 needed, so the
calibration pitfalls of the current export go away; a dynamic-int8 encoder can be tried later
if a smaller file matters. Decision-path burst (the work done at the hop where a decision is
made) is under 2 ms, against 350–450 ms today.

## 4. Data and labels

### 4.1 Source

Seamless Interaction, naturalistic split, train (~45k pairs). The WS9 builder
(`ovfdb/endpoint/{rules,build}.py`) already labels every VAD stop of a floor-holding speaker by
outcome: *complete* (partner takes a real turn within 3 s and keeps it 1 s), *incomplete*
(speaker resumes within 5 s first), everything else excluded. Keep those rules. The licence is
CC BY-NC 4.0, so this model is the NC-licensed one already planned; the commercial audio-only
model keeps training on pipecat's smart-turn-data v3.2.

### 4.2 From per-event clips to per-file tracks

Instead of one 8 s wav + 2 s mp4 per event, build per speaker file:

- `audio.wav` 16 kHz (already fetched);
- `face.npy` 10 fps feature track from the landmarker on the face-cropped video (the one 4K
  decode per file the builder already pays; the 224 px face video is no longer stored);
- `events.jsonl` the labelled silences with `t`, label, `gap_s`, `b_onset_s` (from `rules.py`);
- `vad.json` the speaker's VAD regions (Seamless ships them).

Training samples are drawn on the fly from the tracks:

| sample type | window end | label for p_end | label for p_eager |
|---|---|---|---|
| complete silence at τ | t + τ, τ in the set of §3.3 | 1 | n/a |
| incomplete silence at τ | t + τ, τ ≤ gap_s | 0 | n/a |
| pre-offset of a complete | t − δ, δ ∈ (0, 0.8] s | 0 (still speaking) | 1 if δ ≤ 0.4 |
| pre-offset of an incomplete | t − δ | 0 | 0 |
| random mid-turn | any point with ≥ 1 s of speech ahead | 0 | 0 |

Class weighting as in `train.py`'s batch pos-weight; the Seamless set is 18 % complete at the
event level and far lower at the frame level. Collisions and excluded reasons stay out, as now.

### 4.3 Audio-only pretraining stays

Stage 1 trains the encoder + audio GRU + `p_end` head on smart-turn-data v3.2 plus the Seamless
audio tracks (τ-augmented). Stage 2 adds the face GRU and `p_eager` on Seamless only, encoder at
a low learning rate, as the current two-stage recipe does. The audio-only ONNX exported from
stage 1 is the drop-in replacement for pipecat's `smart-turn-v3.2` and is commercially usable.

### 4.4 Domain gap to webcams

Seamless is studio, face-to-face, high resolution. Webcam users look at a screen, not at the
partner's eyes, and are often off-centre. Face features are much less sensitive to this than
pixels, but gaze-aversion statistics still shift. Plan: record 1–2 hours of webcam conversations
with the existing `app_multimodal.py` capture path (or the OpenVideoFDB review page), label with
the same outcome rules, and use them for threshold calibration and a held-out slice, not for
training, in v2.

## 5. Pipecat integration

1. **Streaming analyzer.** Implement `LocalSmartTurnMultimodalAnalyzer(BaseTurnAnalyzer)` in the
   style of `KrispVivaTurn`: `append_audio(buffer, is_speech)` steps the audio GRU, maintains τ
   from `is_speech`, and returns `COMPLETE` itself when `p_end(τ) ≥ θ(τ)`.
   `TurnAnalyzerUserTurnStopStrategy` already handles analyzers that complete inside
   `append_audio`. `analyze_end_of_turn()` returns the current verdict without new work.
2. **Video in.** Add `append_video(frame: InputImageRawFrame)` to the analyzer and forward
   `UserImageRawFrame`s to it from the stop strategy (one `isinstance` branch in
   `process_frame`, gated on `video_in_enabled`). Frames are decimated to 10 fps and run through
   the landmarker on the analyzer's own thread, never on the pipeline task. Without a camera the
   face GRU state stays at its "no face" embedding, which stage 2 trains with a third of samples
   masked, as the int8 calibration already does for the zero video tensor.
3. **Eager end of turn from an analyzer.** Upstream change, small: a
   `TurnAnalyzerEagerUserTurnStopStrategy` that pushes `EagerTranscriptionFrame` with the
   interim transcript it already tracks in `_text` when the analyzer reports `p_eager ≥ θ_eager`,
   and `EagerEndOfTurnCancelFrame` when VAD reports the user speaking again or the committed
   decision does not follow within the speculation timeout. The existing `EagerUserTurnStopStrategy`
   and `SpeculationGate` do the rest. Until that lands, the analyzer still gives the earlier
   committed decision from step 1, which is where most of the latency gain is.
4. **Keep `stop_secs` as the backstop**, but it should rarely fire.

## 6. Evaluation protocol

Extend `scripts/smartturn_eval_endpoint.py` to a streaming harness over the per-file tracks:

- **Trigger latency on completes**: time from the speaker's VAD offset to the committed
  COMPLETE (p50, p90), including the 3 s timeout for misses. This is the number Pipecat users
  feel; report it for the current model under the current one-shot rule as the baseline.
- **Eager lead**: for completes, how far before the offset `p_eager` fired, and the fraction of
  eager signals withdrawn.
- **False alarms on incompletes** with the time-varying rule, per pause-length bin, and recall at
  matched FAR as the NOTES already do.
- **Fixed-τ ROC** at τ = 0, 0.2, 0.5, 1.0 s so the model stays comparable with the one-shot
  numbers.
- **Ablations**: audio only; audio + τ; audio + τ + face; with and without the eager head.
- **External checks**: the 45-row NVIDIA probe (unchanged script) and a VideoFDB run through the
  OpenVideoFDB harness with a Pipecat agent using the new analyzer, for TOR-Alignment and latency
  against the API models.

## 7. Phasing

**Phase 0, immediate (days), no retraining.**
- Export an audio-only ONNX of the current fp32 model. Video adds nothing and the backbone runs
  on zeros anyway: CPU latency drops from ~250–345 ms to ~35 ms with identical predictions.
- In `BaseSmartTurn`, re-run the model at 0.6 s and 1.2 s of silence (not only at the VAD stop)
  with thresholds per silence bin tuned on the Seamless set. This attacks the 3 s timeout path
  with the model we have. Measure trigger latency before and after.

Both were done on 2026-10-17; the numbers are in §9, the Pipecat analyzer and its measurement in §9.4.

**Phase 1, data (2–3 weeks).** Seamless train build to per-file tracks with 10 fps face features;
streaming eval harness; webcam calibration recordings.

**Phase 2, model (2–3 weeks).** Stage 1 audio + τ; stage 2 face + eager head; threshold tables;
stateful ONNX export (GRU states as inputs and outputs); ablations in §6.

**Phase 3, Pipecat (1–2 weeks).** Streaming analyzer with `append_video`; eager strategy PR
upstream; VideoFDB harness run; README and HF update (the NC model and the commercial audio-only
model as separate files).

## 8. Risks and open questions

- **Eager gains depend on an upstream Pipecat change.** Phases 0–2 do not; the committed
  silence-aware decision is where most of the latency goes anyway.
- **Face landmarker dependency** in a Pipecat analyzer. MediaPipe is a heavy wheel; an ONNX
  landmark + blendshape model (several exist) keeps the analyzer on onnxruntime only. Decide in
  phase 1 after measuring both on the Seamless face videos.
- **Outcome labels are noisy** (a speaker cut off counts as complete). The τ-augmented training
  dilutes this, and the eager head only takes the clean cases (b_onset ≤ 1 s) as positives.
- **Studio to webcam shift** for the visual features (§4.4). If the webcam slice shows no visual
  gain, ship audio + τ and keep the face branch behind a flag.
- **Whisper encoder every 320 ms on an 8 s window** is 8 % of a core; fine on a server, worth
  checking on small edge boxes. A 4 s window halves it at some cost in lexical context.

## 9. Phase 0 results (2026-10-17)

### 9.1 Audio-only export

`export_audio_only_onnx.py` folds the video branch of the fp32 export away by graph surgery: the
R3D-18 backbone is run once on the zero tensor, its 256-d output is frozen as an initializer, the
`pixel_values` input is removed and the dead nodes are pruned. No retraining, no re-export from the
checkpoint. On 64 Seamless endpoint clips (`build_compare_inputs.py` + `compare_onnx_outputs.py`,
Apple M4, one call per row) and on all 2,300 silences of the test set through the stored clips:

| model | inputs | size | max abs Δp vs fp32 (audio-only) | ms/call, CPU |
|---|---|---|---|---|
| fp32 `-gpu.onnx`, zero video | features + pixels | 166 MB | – | 466 |
| int8 backbone `-cpu-v2.onnx`, zero video | features + pixels | 66.5 MB | 0.014 | 200 |
| **`-audio.onnx`** | features only | **33.1 MB** | **0.0 (bit-exact; 0.0001 vs the OpenVideoFDB `eval.csv` on 2,300 rows)** | **17** (32 single-threaded) |

Details in `docs/int8_requantization.md`, "Audio-only export". The file is local (repo root, untracked),
not uploaded to HF. Anything that calls the session must stop passing `pixel_values` (the graph no
longer declares it); `compare_onnx_outputs.run` shows the one-line change.

### 9.2 Silence-aware re-query

`eval_silence_requery.py` re-cuts the 8 s window for every silence of the test set from the
speaker's source audio so that it ends 0.2, 0.6 and 1.2 s after the VAD stop (the 0.2 s window
matches the stored clip: max |Δp| 0.036, mean 0.0001), scores it with the audio-only graph, and
simulates decision rules. A query at τ only happens while the silence is still going on (the speaker
has not resumed: `gap_s` ≥ τ); a miss ends the turn on the 3 s `stop_secs` timeout. Latency is
measured from the VAD stop and excludes compute (~50 ms with the audio-only graph). Full tables:
`docs/phase0_silence_requery.md`.

What the model sees at each offset:

| offset | completes still silent | incompletes still silent | AUC | mean p, complete | mean p, incomplete |
|---|---|---|---|---|---|
| 0.2 s | 417 | 1,883 | 0.801 | 0.52 | 0.14 |
| 0.6 s | 417 | 785 (42 %) | 0.779 | 0.63 | 0.23 |
| 1.2 s | 416 | 226 (12 %) | 0.760 | 0.60 | 0.24 |

Decision rules on the 2,300 silences (417 complete). Thresholds fitted on the whole set by exhaustive
grid search under the FAR constraint; "max recall" and "min latency" are the two objectives (a miss
costs 3 s, so the latency objective trades a few misses for earlier commits):

| rule | θ at 0.2 / 0.6 / 1.2 s | recall | FAR on pauses | p50 latency | p90 latency | mean latency |
|---|---|---|---|---|---|---|
| **one-shot, θ 0.5 (current Pipecat rule)** | 0.50 / – / – | 0.52 | 13.3 % | 0.20 s | 3.00 s | 1.54 s |
| re-query, θ 0.5 at every offset | 0.50 / 0.50 / 0.50 | 0.69 | 16.6 % | 0.20 s | 3.00 s | 1.17 s |
| silence only, commit at 0.6 s (no model) | – / any / – | 1.00 | 41.7 % | 0.60 s | 0.60 s | 0.60 s |
| **silence only, commit at 1.2 s (no model)** | – / – / any | 1.00 | 12.0 % | 1.20 s | 1.20 s | 1.20 s |
| one-shot, FAR ≤ 5 % | 0.978 / – / – | 0.27 | 5.0 % | 3.00 s | 3.00 s | 2.25 s |
| re-query, FAR ≤ 5 %, max recall | 0.997 / 0.998 / 0.008 | 0.79 | 4.9 % | 1.20 s | 3.00 s | 1.56 s |
| re-query, FAR ≤ 5 %, min latency | 0.997 / 0.997 / 0.012 | 0.78 | 4.9 % | 1.20 s | 3.00 s | 1.53 s |
| one-shot, FAR ≤ 10 % | 0.858 / – / – | 0.43 | 10.0 % | 3.00 s | 3.00 s | 1.79 s |
| re-query, FAR ≤ 10 %, max recall | 0.997 / 0.996 / 0.001 | 0.96 | 10.0 % | 1.20 s | 1.20 s | 1.17 s |
| **re-query, FAR ≤ 10 %, min latency** | 0.991 / 0.991 / 0.001 | 0.93 | 10.0 % | 1.20 s | 1.20 s | 1.06 s |
| one-shot, FAR ≤ 15 % | 0.317 / – / – | 0.56 | 15.0 % | 0.20 s | 3.00 s | 1.42 s |
| re-query, FAR ≤ 15 %, max recall | 0.997 / 0.983 / 0.000 | 1.00 | 14.9 % | 1.20 s | 1.20 s | 0.95 s |
| **re-query, FAR ≤ 15 %, min latency** | 0.971 / 0.636 / 0.001 | 0.94 | 14.9 % | 0.60 s | 1.20 s | 0.82 s |

2-fold cross-fit by conversation (thresholds fitted on half of the 67 conversations, scored on the
other half) moves recall by at most 0.06 and FAR by at most 0.7 points, except the 5 % min-latency
table (recall 0.72, FAR 5.6 %); the ranking of the rules is unchanged.

Where the fitted rules commit (completes / false alarms per offset, in-sample tables):

| table | completes at 0.2 / 0.6 / 1.2 s / missed | false alarms at 0.2 / 0.6 / 1.2 s |
|---|---|---|
| FAR ≤ 10 %, min latency | 68 / 67 / 254 / 28 | 46 / 34 / 108 |
| FAR ≤ 15 %, min latency | 122 / 137 / 132 / 26 | 109 / 100 / 72 |

False alarms by pause length, incompletes (n = 891 / 846 / 146):

| rule | pause < 0.5 s | 0.5–1.5 s | ≥ 1.5 s |
|---|---|---|---|
| one-shot, θ 0.5 | 9 % | 15 % | 28 % |
| re-query, FAR ≤ 10 %, min latency | 2 % | 9 % | 66 % |
| re-query, FAR ≤ 15 %, min latency | 5 % | 16 % | 72 % |

### 9.3 Reading

1. **The 3 s timeout, not the model, is the latency problem.** Ending the turn after 1.2 s of
   silence with no model at all beats the current one-shot rule on every number: recall 1.00 vs
   0.52, false alarms 12.0 % vs 13.3 %, p90 latency 1.2 s vs 3.0 s. Only 12 % of the thinking pauses
   in this set last 1.2 s; 42 % last 0.6 s.
2. **Re-query with the current model roughly doubles recall at the same false-alarm rate** (0.43 →
   0.93–0.96 at 10 %, 0.56 → 0.94–1.00 at 15 %) and removes the 3 s tail (p90 3.0 s → 1.2 s). The
   gain comes almost entirely from the late query: the fitted tables set θ(0.2 s) ≈ 0.97–0.99, i.e.
   the model's answer at the VAD stop is trusted only when it is nearly certain, and θ(1.2 s) ≈ 0,
   i.e. the 1.2 s query is a catch-all. At 15 % FAR the model buys a p50 of 0.6 s against 1.2 s for
   the silence-only rule, with 259 of 417 completes committed at 0.2 or 0.6 s.
3. **The model does not improve as the silence grows.** AUC falls from 0.801 at 0.2 s to 0.760 at
   1.2 s and the mean score on *incomplete* pauses rises from 0.14 to 0.24: the model was trained on
   windows ending 0.2 s after the stop and reads trailing silence as completion. That is why the
   fitted thresholds collapse to "always" at 1.2 s and why the false-alarm budget of the re-query
   rules is spent on pauses ≥ 1.5 s (66–72 %, against 28 % today). The τ-augmented training of §3.3
   is what should fix this; §4.2's windows at τ ∈ {0.3, 0.5, 0.8, 1.2, 1.8} are the data for it.
4. **Recommendation for Phase 0 in Pipecat.** Re-run the model at 0.6 s and 1.2 s of silence and
   use the "FAR ≤ 15 %, min latency" table (θ = 0.97 / 0.64 / 0) or, when fewer interruptions matter
   more than speed, the 10 % table (0.99 / 0.99 / 0, which in practice is `stop_secs = 1.2` plus an
   early exit for near-certain cases). Both keep false alarms on short pauses at or below today's
   level (2–5 % vs 9 %) and move them to long pauses, where interrupting a thinking speaker after
   1.2 s is the same behaviour a 1.2 s `stop_secs` would give. Measured end to end in Pipecat in §9.4.
5. **Caveats.** Thresholds are fitted on the test set (the cross-fit above is the honest estimate);
   the simulation assumes the partner stays silent until the bot answers, uses the speaker's own
   resumption to decide whether a later query happens, and ignores compute time; labels are by
   outcome, so a speaker cut off after 1.2 s counts as complete.

### 9.4 Pipecat measurement

`local_smart_turn_multimodal.py` implements the rule as a pipecat 1.12 `BaseSmartTurn`:
`LocalSmartTurnMultimodalAnalyzer` runs the audio-only graph (`input_features` only, preprocessing of
`inference_multimodal.py`), answers the stop strategy's query at the VAD stop with θ(0.2), then
schedules the 0.6 s and 1.2 s queries itself from `append_audio` once the counted silence reaches
τ − `vad_stop_secs`, runs them on the analyzer's thread and returns COMPLETE from `append_audio` on the
next audio frame when p ≥ θ(τ); `TurnAnalyzerUserTurnStopStrategy` already honours that. Speech
reported by the VAD cancels pending re-queries; `stop_secs` stays as the backstop. The tables are in
`RULES` (`far15_min_latency` = 0.971 / 0.636 / 0, the default; `far10_min_latency` = 0.991 / 0.991 / 0;
`one_shot` = the stock rule). No change to pipecat itself.

`measure_pipecat_latency.py` replays the speaker's source audio of a silence (9 s before `t_raw` to
3.6 s after) as 20 ms frames at real-time pace through `VADProcessor(SileroVADAnalyzer)` →
`UserTurnProcessor(VAD start, TurnAnalyzerUserTurnStopStrategy(analyzer, wait_for_transcript=False))`
and takes the wall-clock time from the acoustic offset to `on_user_turn_stopped`. 60 complete and 40
incomplete silences of the test set (seed 0), replay level normalised to −23 dBFS RMS (the Seamless
recordings sit at −25 to −56 dBFS and pipecat's VAD gates on loudness; without the normalisation it
dropped soft endings and missed quiet speakers entirely, see `docs/phase0_pipecat_latency.md`).
Apple M4, CPU. A complete with no stop in the window is charged 3.2 s (VAD `stop_secs` + `stop_secs`):

| rule | turn ended within 3.6 s | ended by the model at 0.2 / 0.6 / 1.2 s · by `stop_secs` · not at all | false alarms on 40 pauses, strict / effective | latency from the acoustic offset, completes: p50 / p90 / mean | from pipecat's own VAD offset: p50 / p90 |
|---|---|---|---|---|---|
| **one-shot, θ 0.5 (stock rule)** | 0.90 | 35 / – / – · 19 · 6 | 2.5 % / 2.5 % | 0.24 / **3.20** / 1.50 s | 0.26 / 3.20 s |
| **re-query, FAR ≤ 15 %, min latency** | 0.97 | 21 / 17 / 20 · 0 · 2 | 12.5 % / 15.0 % | **0.57 / 1.23 / 0.74 s** | 0.66 / 1.26 s |
| re-query, FAR ≤ 10 %, min latency | 0.97 | 14 / 9 / 35 · 0 · 2 | 12.5 % / 15.0 % | 1.16 / 1.23 / 0.93 s | 1.25 / 1.26 s |

"Strict" false alarms are stops before the speaker resumed; "effective" ones also count stops within the
VAD's `start_secs` (0.2 s) after the resumption, when pipecat could not yet have noticed it (one case
per re-query rule, both at the 1.2 s query). The per-offset overhead of the pipeline over the nominal
offset, measured from pipecat's own VAD offset, is 55 ms at 0.2 s (the model run) and 59 ms at 0.6 and
1.2 s (model run + the one-frame poll), with 50 ms per query of which ~30 ms is the
`WhisperFeatureExtractor` and ~17 ms the ONNX run. Full tables, the false-alarm breakdown, the
un-normalised run and a synthetic `say` check: `docs/phase0_pipecat_latency.md`.

Reading:

1. **The offline numbers hold in the pipeline.** Against §9.2 (same rules, 2,300 silences, no compute):
   one-shot p50 0.20 → 0.24 s, p90 3.0 → 3.2 s; FAR ≤ 15 % table p50 0.60 → 0.57 s, p90 1.20 → 1.23 s,
   mean 0.82 → 0.74 s; FAR ≤ 10 % table p90 1.20 → 1.23 s. The differences are the ~60 ms pipeline
   overhead and the sample (60 completes; pipecat's VAD places the offset ~80 ms before `t_raw`).
   Recall of the re-query rules is 0.97 here against 0.93–0.94 offline; the one-shot rule's 0.90
   counts the 19 turns it ended by the 3 s timeout, and in six it never ended the turn because the
   speaker spoke again (backchannel, or continuing once the partner had started) before the timeout
   ran out.
2. **False alarms match the silence-only rule.** 5 of 40 pauses (12.5 %) ended early under both re-query
   tables; the offline estimate was 14.9 % and 10.0 %, the sample has 3 pauses ≥ 1.2 s and 13 ≥ 0.6 s.
   Most are the 1.2 s catch-all firing on pauses of 1.1–1.7 s, as §9.3 predicts (three of the five
   under the 15 % table, four under the 10 % table; the rest are one at 0.6 s on a 0.77 s pause and
   one at 0.2 s on a 0.32 s pause); the one-shot rule's 2.5 % is below its offline 13.3 % on this sample.
3. **What the pipeline adds.** Each commit costs the model run plus one 20 ms frame, 55–60 ms, so the
   0.6 s query answers at 0.66 s and the 1.2 s one at 1.26 s measured from pipecat's VAD offset.
   Pipecat's vendored numpy log-mel (`pipecat.audio.turn.smart_turn._whisper_features`) would cut the
   feature extraction from ~30 ms to a few ms; not done here to keep the repo preprocessing. The VAD's
   `start_secs` means a re-query can commit up to 0.2 s after the speaker has resumed (one of the
   five false alarms per table); the eager cancel in §5 item 3 is the fix. One more difference from
   the offline eval: after any committed turn `BaseSmartTurn` empties its buffer, so the next decision
   sees audio from the next VAD start (minus `pre_speech_ms`) rather than the full 8 s, and a false
   alarm inside a long turn also shortens the context for the decision at its real end. The replays
   inherit this (pauses in the 9 s pre-roll are decided like any other), the offline windows did not.
4. **Recording level matters to the VAD, not to the rule.** At the recordings' own level pipecat's
   Silero (confidence 0.7, `min_volume` 0.6) placed the offset 0.7–1.2 s early on a fifth of the
   turns and never started 6 of 300; recall fell to 0.82–0.88 for every rule. Anyone replaying
   archival audio through pipecat should normalise it first, or lower `min_volume`.
