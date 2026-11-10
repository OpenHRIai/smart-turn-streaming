# End-of-turn detection on the Parakeet Realtime EOU encoder: plan and results

Status: **stopped 2026-11-10 after step 1.** On English eot-bench the head ties with
smart-turn-streaming (33.5% vs 31.1% false cutoffs at 300 ms), which the pre-set rule says is a
stop; on CANDOR it beats both baselines clearly, recorded as a finding. Steps 2 to 4 not run.

## Goal and the bar

Find out whether a small causal streaming ASR encoder gives an end-of-turn detector that clearly
beats smart-turn-streaming, and if so, whether it can be made multilingual.

The measure is the eot-bench leaderboard's: **false cutoffs at an average response delay of 300 ms**.
A false cutoff is a mid-turn pause (0.2 to 5 s) during which the agent would have started talking.
The response delay is measured from the start of silence on true turn ends, averaged over all of
them. The benchmark sweeps the score threshold, an action delay (0.2 to 1.0 s) and a timeout (1.0 to
3.5 s) and keeps the setting with the fewest false cutoffs that stays within the delay budget.

On English eot-bench, smart-turn-streaming scores 31.1% at 300 ms and 14.8% at 600 ms (scoring every
100 ms, taking the first threshold crossing); Smart Turn v3.2 scored the same way gets 32.8% / 14.2%.
The two are tied within the bootstrap interval. The leaderboard leader, LiveKit Turn Detector v1,
scores 9.9% / 4.5%; turn-1-mini, the closest small open model, reports 22.6% / 9.6%.

**Decision rule at 300 ms on English eot-bench, fixed before scoring:**

- clearly below 22.6%: go on to step 2 (other languages), then step 3;
- 22.6% to about 30% and significantly better than 31.1%: try continued self-supervised training of
  the encoder with Voice Activity Projection (VAP, arXiv 2205.09812), as ooma-ai/turn-detector
  describes;
- about 30% or worse: stop this line.

## The model

`nvidia/parakeet_realtime_eou_120m-v1`: cache-aware FastConformer RNN-T, 17 layers, d_model 512,
109.5M-parameter encoder, 8x subsampling (one encoder frame per 80 ms), attention context [70, 1]
(5.6 s back, one frame ahead), 160 ms chunks. English only (model card); vocabulary of 1,024
English subwords plus `<EOU>` (end of utterance) and `<EOB>` (end of backchannel). NVIDIA Open Model
License. Its training list includes AMI, so AMI is not independent for this encoder.

## Steps

1. **A head on the frozen English encoder, scored on English eot-bench.** In progress; phases below.
2. **The same model on eot-bench's other 13 languages** (344 to 400 turns each). A stress test: the
   encoder hears no meaningful words outside English, so this measures how much comes from sound
   (pitch, pace, breath, silence) rather than words. Estimated under $5 on Modal GPUs, or about 12 h
   on the laptop CPU.
3. **The same head on the multilingual Nemotron 3.5 ASR Streaming encoder**
   (`nvidia/nemotron-3.5-asr-streaming-0.6b`, 600M), with its CPU cost in this role measured.
4. **Distil a 120M-parameter student from Nemotron 3.5** by matching encoder outputs, then train the
   head on the student.

### Phases of step 1

| phase | what | status |
|---|---|---|
| 0 | setup checks: causality, repeatability, transcripts, CPU cost | done |
| 1 | the native `<EOU>` token as a detector, no training | done |
| 2 | encoder features for every training window, on Modal | done |
| 3 | train the head; checkpoint chosen on AMI dev only | done |
| 4 | the one chosen model on English eot-bench, plus CANDOR as a second test | done |

## Results so far

### Phase 0: setup checks (passed)

| check | result |
|---|---|
| causality | changing the audio after 3.0 s changed no earlier encoder output; later output did change |
| repeatability | identical output on repeated runs |
| transcripts | streaming text matched NeMo's offline text on 3 eot-bench turns |
| CPU (PyTorch, one thread, M4) | 34-39 ms per 160 ms chunk, 22-25% of one thread; smart-turn-streaming measured 9.1% of a core |
| memory | about 1.7 GB peak |

NeMo's `transcribe()` leaves the encoder in training mode (dropout on); anything that runs the
encoder after it must call `model.eval()`. The adapters never call it.

### Phase 1: the native `<EOU>` token on English eot-bench

Score = the highest P(`<EOU>`) so far in each silence (`eot_harness/parakeet_eou_adapter.py`).

| average response delay | native `<EOU>` | smart-turn-streaming | Smart Turn (all scores) |
|---|---|---|---|
| 300 ms | not reachable | 31.1% | 32.8% |
| 600 ms | not reachable | 14.8% | 14.2% |
| 750 ms | 11.2% | 9.6% | 8.8% |
| 1000 ms | 5.4% | 5.2% | 5.4% |

AUC 0.886 (detector 0.881, Smart Turn 0.895). The token is late and often silent on real speech:
71% of all scores are exactly 0; by 0.3 s of silence only 4.0% of true turn ends have passed even
0.01; 33.5% never pass 0.5 within the 1.5 s span. A third of turn ends therefore wait for the
timeout (at least 1.0 s), so the fastest reachable average is 730 ms.

**The adapter is not the cause.** On 12 clean TTS sentences (macOS `say`) with 3 s of silence
appended, the model card's setup, the token fired on all 12, with a median of 167 ms and a 90th
percentile of 230 ms by the frame it sits on (card: 160 / 280 ms), or 247 / 310 ms by when that
chunk has arrived, which is the clock eot-bench uses. Streaming and offline decoding placed `<EOU>`
on all 12. The token works on clear finished sentences and holds back on real conversational ones.

### Phase 2: encoder features

One forward pass over a 777-frame segment gives the same output as the chunk-by-chunk streaming
encoder (max difference 7e-5 on values around 0.15), so each 8 s training window costs one GPU pass.
Each window is trimmed by a seeded random 0 to 160 ms at its end (in a live stream the newest chunk
is up to one chunk old); the segment then ends on a streaming chunk boundary.

Per window: the last 13 encoder frames (1.04 s, fp16, 13 x 512), the mean of all 98 frames (512)
and the trim. All 1,137,921 windows of `data/b1/items.parquet`, 207 tasks on up to 8 L4 GPUs, no
failures, 16.3 GB at `data/parakeet/feats.npz` on the `smart-turn-v2` volume. Billed $3.67 (plus
$0.09 for the one-shard measurement).

### Phase 3: the head

Same head as smart-turn-streaming's (`eot/model.py` `Heads`, 355,906 parameters): a GRU over the 13
frames, a projection of the window mean, the silence clock [t / 2.5, log(1 + t), speaking]. Two
outputs trained with binary cross-entropy: `p_end` (the turn is over) on every window, `p_eager`
(a finished turn ends within 0.4 s) on pre-offset windows only. Training data: smart-turn-data
(616,484 windows, 23 languages, 57.5% English clips) and AMI train (317,448), AMI at 70% of each
epoch. The recipe of the released heads (B1 run `knob_ami07_auc`): AdamW, lr 1e-3 one-cycle, 8
epochs, batch 1024, dropout 0.1, checkpoint by mean AMI dev AUC at 0.2 s and 1.2 s.

Run `runs/parakeet/pk_ami07_auc` (`modal_parakeet_train.py`, 6 min on one L4, $0.33). Epoch 2 was
chosen; dev AUC falls after it while training loss keeps dropping.

| | AUC at 0.2 s | 0.5 s | 1.2 s |
|---|---|---|---|
| AMI dev, Parakeet head (epoch 2) | 0.761 | | 0.655 |
| AMI dev, released detector recipe (3 seeds) | 0.785 | | 0.730 |
| AMI test, Parakeet head | 0.790 | 0.778 | 0.720 |
| AMI test, released detector | 0.775 | 0.776 | 0.742 |

### Phase 4a: CANDOR, a second independent test

CANDOR (BetterUp; 1,656 two-person video calls; CANDOR Corpus License v2.1, commercial use allowed,
no redistribution) is already processed for OpenVideoFDB: per-speaker 16 kHz audio and Silero VAD on
the `candor` Modal volume, and 1.07M labelled silences from the same turn rules as our AMI set (turn
end = the partner took a real turn within 3 s; pause = the same speaker resumed within 5 s). The test
uses conversation half B of OpenVideoFDB's split (md5 of the conversation id, odd), leaving half A
free for training: 200 conversations sampled, 10 turn ends and 20 pauses each, scored on eot-bench's
grid (every 100 ms; turn-end spans 1.5 s with room tone after the partner's onset; pauses until the
speaker resumes). 65,299 windows. Every model scores the same windows. Intervals: 2,000 paired
bootstrap draws over conversations.

| model | AUC | false cutoffs @300 ms | @600 ms | latency @5% cutoffs | @10% |
|---|---|---|---|---|---|
| smart-turn-streaming | 0.761 | 70.1% | 33.9% | 1887 ms | 1215 ms |
| Smart Turn v3.2 | 0.743 | 81.4% | 34.4% | 1819 ms | 1225 ms |
| Parakeet head | **0.863** | **53.3%** | **27.4%** | 1724 ms | 1173 ms |

Parakeet head minus smart-turn-streaming: -17.4 points at 300 ms (95% interval -27.6 to -13.9) and
-6.7 at 600 ms (-8.5 to -5.0). By silence length the head's AUC lead is largest early (0.756 vs 0.679
at 0.2 s) and gone by 1.5 s (0.606 vs 0.598).

CANDOR is much harder than eot-bench for every model: labels come from rules over VAD and ASR words,
not human annotation; the speakers are strangers on video calls. Parakeet's training list does not
name CANDOR, but about 10,000 h of it is an internal NVIDIA set whose contents are not published.

Caveats: CANDOR windows give smart-turn-streaming a freshly computed encoder vector at every grid
point (its live version refreshes less often); Smart Turn's score comes from the fp32 GPU export,
not the int8 CPU one.

### Phase 4b: English eot-bench (the decision)

`eot_harness/parakeet_head_adapter.py`: at each grid time the segment ends at the newest chunk
boundary that has arrived, the chunk grid running from the turn start; silence clock from the
annotated spans, as for smart-turn-streaming. 1 h 09 min on the laptop CPU for the 400 turns.
Output: `eot-bench/output/.../en/parakeet_head_adapter__parakeet_eou_120m_head_pk_ami07_auc/`.

| model | AUC | false cutoffs @300 ms | @600 ms | latency @5% cutoffs | @10% |
|---|---|---|---|---|---|
| smart-turn-streaming | 0.881 | 31.1% | 14.8% | 1012 ms | 745 ms |
| Smart Turn v3.2 (all scores) | 0.895 | 32.8% | 14.2% | 1045 ms | 723 ms |
| Parakeet head | 0.865 | 33.5% | 14.2% | 1074 ms | 734 ms |

Parakeet head minus smart-turn-streaming: +1.7 points at 300 ms (95% interval -2.5 to +5.9) and
0.0 at 600 ms (-2.2 to +2.3). A tie with both baselines.

**Under the rule fixed before scoring, 33.5% at 300 ms is in the "about 30% or worse: stop" band.**
Steps 2 to 4 are not justified by English eot-bench.

### Reading the two tests together

The same head is level with smart-turn-streaming on eot-bench (people talking to voice agents,
human-annotated pauses) and well ahead on CANDOR (strangers talking to each other, rule-labelled
pauses). Explanations not yet separated: the encoder's representations suit human-to-human talk
better than talk to an agent; CANDOR's rule-based labels agree better with what an ASR encoder
captures; or CANDOR-like audio is in the unpublished part of Parakeet's training data. eot-bench is
the target setting for Pipecat voice agents, so it decides; the CANDOR result is recorded as a
finding.

This work uses the CANDOR Corpus from BetterUp Labs: Reece, A., Cooney, G., et al. (2023). The CANDOR
corpus: Insights from a large multimodal dataset of naturalistic conversation. *Science Advances*
9(13), eadf3197. https://doi.org/10.1126/sciadv.adf3197. No CANDOR audio, labels or features are in
this repository, only the code that builds the test and the aggregate scores above.

## Files

| what | where |
|---|---|
| native token adapter | `eot-bench/eot_harness/parakeet_eou_adapter.py` |
| head adapter | `eot-bench/eot_harness/parakeet_head_adapter.py` |
| feature extraction (training windows, CANDOR) | `smart-turn-streaming/modal_parakeet_features.py` |
| Whisper-side CANDOR features (Smart Turn, detector) | `smart-turn-streaming/modal_features.py` (`--what candor`) |
| training | `smart-turn-streaming/scripts/b1_train.py --backbone parakeet`, `modal_parakeet_train.py` |
| CANDOR test windows and predictions | `smart-turn-streaming/scripts/candor_eot_items.py`, `scripts/candor_eot_predict.py` |
| comparison with paired bootstrap | `smart-turn-streaming/scripts/eot_compare.py` (reproduces the committed eot-bench numbers exactly) |
| CANDOR outputs (licence: never leaves these places) | `OpenVideoFDB/data/candor/eot_test/` (git-ignored), `candor` volume `eot_test/` |

Nothing in this line is committed yet.

## Costs

Modal so far, from `modal billing report`: about $5.02 (training-window features $3.77 including the
one-shard measurement, CANDOR features $0.69, training $0.56 including the one-epoch timing run).
