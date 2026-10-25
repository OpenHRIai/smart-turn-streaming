# Stage A3 + B1: sampler, frozen-encoder features, and the first τ-aware heads

Status: 2026-10-24. Plan: `docs/plan_online_eot.md` §5–6. Code: `scripts/b1_items.py` (A3 item
enumeration), `eot/features.py` + `modal_features.py` (feature cache), `eot/model.py`,
`eot/rules_eval.py`, `scripts/b1_train.py`. Runs under `runs/b1/<name>/` (`report.md`,
`heads.pt`, `predictions.npz`; git-ignored). Data: Modal volume `smart-turn-v2` and local `data/b1`.

## What was built

**Items (A3).** `scripts/b1_items.py` enumerates 1,137,921 windows from the two sources, each an
8 s window ending at a defined point with two labels (`p_end`, `p_eager`; −1 = undefined):

| source | split | clips / events | tau windows | pre-offset | mid-speech | p_end = 1 | p_eager = 1 |
|---|---|---|---|---|---|---|---|
| smart-turn-data | train (77,573 clips: all 47.6k human-recorded + 30k TTS; 4 of 9 taus, 3 of 6 deltas, 1 mid per clip) | 77,573 | 310,292 | 232,200 | 73,992 | 153,912 | 76,858 |
| smart-turn-data | test (6,000 clips, balanced; full sets) | 6,000 | 54,000 | 35,908 | 5,851 | 27,000 | 11,969 |
| AMI | train (every labelled silence; 4 taus, 3 deltas, 1 mid) | 43,339 | 157,522 | 128,480 | 31,446 | 22,428 | 11,664 |
| AMI | dev (full sets) | 4,415 | 23,842 | 25,371 | 3,251 | 6,606 | 2,871 |
| AMI | test (full sets) | 4,815 | 25,000 | 27,444 | 3,322 | 4,599 | 1,958 |

For AMI incompletes a tau window exists only while the silence is still going on (τ < gap). For
AMI completes, a tau window that runs past the next speaker's onset is cut there and continued
with the speaker's own room tone (14,452 windows): the deployed model never hears the reply, it
only sees that nobody has answered yet.

**Feature cache.** `modal_features.py` cut every window on L4 GPUs (207 tasks, ≤ 8 containers,
about 25 min), computed Whisper log-mel exactly as `train.py` does, ran two frozen backbones
through ONNX Runtime with the pooled 384-d vector exposed as an extra output (`eot.features.
expose_pooled`, bit-exact probabilities), and stored per window: `pooled_v32` and `pooled_mm`
(fp16), `mel20` = the last 1 s of log-mel at 50 ms steps (20×80, fp16), and each backbone's own
P(complete). `feats.npz` is 5.0 GiB. Backbones: pipecat `smart-turn-v3.2-gpu.onnx` (v32) and our
audio-only export `smart-turn-multimodal-audio.onnx` (mm).

**Heads (B1, training level 1).** `eot.model.Heads`: LayerNorm + Linear on the pooled vector;
GRU(80→128) over the 20 mel steps; τ features [τ/2.5, log1p τ, speaking]; fusion MLP; two logits.
172,514 trainable parameters, encoder frozen. Masked BCE with batch pos-weights (as upstream).
Per-source balanced sampling, 6 epochs, OneCycle 2e-3, batch 1024. One run takes about 3 min on
the laptop (MPS). Best epoch by dev score (recall at FAR 10 % minus mean latency / 10).

**Evaluation** (`eot.rules_eval`): AUC per τ; eager AUC per δ; the re-query rule queried at
0.2 / 0.5 / 1.2 s of silence with θ per offset fitted on **AMI dev** by grid search under a
false-alarm budget (min-latency objective) and scored on **AMI test**; misses charged 3 s;
the frozen backbone's own probability under the same protocol as the baseline. smart-turn-data
test is the regression guard.

## Results on AMI test (511 complete / 4,304 incomplete silences; 16 meetings)

Three seeds per backbone (mean ± sd), frozen backbones as reference:

| model | AUC τ = 0.2 s | AUC τ = 1.2 s | AUC τ = 2.5 s | eager AUC δ = 0.1 / 0.4 s |
|---|---|---|---|---|
| frozen Smart Turn v3.2 | 0.671 | 0.575 | 0.619 | 0.676 / 0.609 |
| frozen multimodal (audio) | 0.688 | 0.582 | – | – / 0.638 |
| **v3.2 encoder + heads** | **0.759 ± 0.006** | **0.732 ± 0.003** | 0.714 | 0.738 / 0.671 |
| **our encoder + heads** | **0.780 ± 0.007** | 0.715 ± 0.006 | 0.719 | 0.758 / 0.689 |

Re-query rule, thresholds fitted on dev, scored on test, FAR-budget sweep (seed 1 runs; the
frozen baseline is pipecat v3.2 under the same rule):

| FAR budget | v3.2 + heads: recall / FAR / mean latency | frozen v3.2: recall / FAR / mean latency | our encoder + heads |
|---|---|---|---|
| 5 % | 0.43 / 0.038 / 2.12 s | 0.37 / 0.055 / 2.34 s | 0.40 / 0.030 / 2.25 s |
| 7.5 % | 0.64 / 0.058 / 1.83 s | 0.58 / 0.086 / 1.96 s | 0.56 / 0.050 / 1.96 s |
| 10 % | 0.77 / 0.082 / 1.58 s | 0.70 / 0.106 / 1.74 s | 0.72 / 0.073 / 1.71 s |
| 12.5 % | 0.84 / 0.106 / 1.42 s | 0.80 / 0.125 / 1.56 s | 0.78 / 0.094 / 1.50 s |
| 15 % | 0.88 / 0.126 / 1.27 s | 0.87 / 0.137 / 1.44 s | 0.86 / 0.120 / 1.34 s |
| 20 % | 0.97 / 0.182 / 1.03 s | 1.00 / 0.180 / 1.20 s | 0.94 / 0.173 / 1.04 s |

Over three seeds at the 10 % budget: v3.2 + heads recall 0.75 ± 0.02 at FAR 0.080, mean latency
1.60 s, FAR on pauses ≥ 1.5 s 0.46; frozen v3.2 0.70 at 0.106, 1.74 s, 0.61. p90 latency stays
3.0 s for every rule below a 20 % budget: in meetings a quarter of the turn ends still time out.
Stock rule (θ 0.5 at the VAD stop): heads recall 0.50–0.58 at FAR 17–21 %; frozen v3.2 0.51 at 30 %.

Ablations (single seed; seed sd ≈ 0.01 AUC, 0.02 recall):

| configuration | AUC 0.2 s | AUC 1.2 s | recall @ 10 % budget (actual FAR) | std test AUC 0.2 s / 2.5 s |
|---|---|---|---|---|
| std + AMI (reference) | 0.754 | 0.736 | 0.73 (0.074) | 0.977 / 0.968 |
| AMI only | 0.777 | 0.735 | 0.78 (0.083) | **0.662 / 0.495** |
| std only | 0.665 | 0.564 | 0.68 (0.102) | 0.977 / 0.971 |
| no mel branch | 0.774 | **0.664** | 0.72 (0.087) | – |
| no τ input | 0.771 | 0.730 | 0.79 (0.091) | 0.970 / 0.969 |
| 3 epochs | 0.785 | 0.686 | 0.70 (0.090) | 0.976 / 0.966 |

## smart-turn-data test (regression guard, 6,000 clips)

| τ | AUC: v3.2 + heads / frozen v3.2 | recall at 5 % FAR: heads / frozen |
|---|---|---|
| 0.0 | 0.976 / 0.978 | 0.89 / 0.89 |
| 0.2 | 0.977 / 0.976 | 0.88 / 0.86 |
| 0.5 | 0.975 / 0.968 | 0.87 / 0.82 |
| 1.2 | 0.975 / 0.964 | 0.86 / 0.77 |
| 2.5 | 0.968 / 0.955 | 0.82 / 0.72 |

Per-source AUC at τ = 0.2 is unchanged within noise on every source (chirp3 0.981/0.981,
liva 0.986/0.980, midcentury 1.000/1.000). Eager AUC on pre-offset windows: 0.84 at δ = 0.4 s
(frozen 0.76), 0.96 at δ = 0.1 s (0.97). At threshold 0.5 the heads sit at higher recall and
higher FAR than the frozen model (0.97 / 0.13 vs 0.94 / 0.08 at τ = 0.2): a calibration shift from
the class-weighted loss, not a ranking change; operating points are set by the fitted rule.

## Reading

1. **Heads on a frozen encoder beat the frozen model everywhere on conversational data.** On AMI
   test the AUC at the VAD stop rises from 0.67 to 0.76–0.78, and at 1.2 s of silence from 0.58 to
   0.73: the frozen model's ranking collapses as the silence grows (it was never shown silence),
   the heads' does not. Under the same re-query rule the heads give +7 points of recall at the
   10 % false-alarm budget with lower actual FAR, and at the stock threshold they halve the false
   alarms at equal recall.
2. **The conversational data is what does it.** Heads trained on pipecat's data alone, even with
   τ-padding, land exactly on the frozen model on AMI. AMI alone gives the best AMI numbers but
   drops smart-turn-data AUC from 0.98 to 0.66, so the commercial model must train on both; the
   mix keeps the pipecat domain intact (AUC 0.977 vs 0.976 at the stop) and improves it at long
   silences (0.968 vs 0.955 at 2.5 s).
3. **The mel branch is the silence-aware part.** Without it the AUC at 1.2 s falls from 0.74 to
   0.66; the explicit τ scalar adds nothing on top of it (the GRU sees the silence in the mel).
   This is good news for the streaming design: the per-hop GRU over fresh mel is exactly the
   component that carries the τ information, and it costs well under 1 ms.
4. **Backbone choice is a trade.** Our encoder ranks better at the stop (0.780 vs 0.759), pipecat's
   holds better at 1.2 s (0.732 vs 0.715); rule results are equal within seed noise. For the
   commercial release pipecat's v3.2 encoder is the clean choice (trained only on CC BY data, and
   the comparison isolates what the heads add); ours is the natural backbone for the NC model.
5. **Meetings are hard, and the ceiling is the data.** Silence-only at 1.2 s gives recall 1.00 at
   FAR 18 % on AMI; at budgets below that every rule times out on a quarter of the completes. In
   four-party meetings the "complete" label (someone else took the floor) is weaker evidence of the
   speaker's intent than a dyadic reply, and the headset channel carries crosstalk. The Seamless
   dyadic test in R2 will show how much of the gap is the data.
6. **Training recipe.** Dev AUC at τ = 0.2 drifts down over epochs while the rule metric improves
   slightly; 3 epochs gives a higher stop-AUC (0.785) and lower long-silence AUC (0.686). A lower
   learning rate, more AMI weight or encoder fine-tuning (level 2) are the next knobs.

## Gates (plan §6, B1), measured on AMI dev / test

| gate | target | result | status |
|---|---|---|---|
| AUC at 1.2 s not below AUC at 0.2 s | ≥ 0 | −0.03 (heads) vs −0.10 (frozen) | restated: the remaining incompletes at 1.2 s are the hard long pauses, so AUC falls for any model; the heads lose 0.03 where the frozen model loses 0.10 |
| recall at FAR 10 % under the re-query rule | ≥ 0.75 | 0.75 ± 0.02 (test), 0.76 (dev) | met, narrowly |
| FAR on pauses ≥ 1.5 s at that operating point | ≤ 40 % | 46 % (frozen: 61 %) | not met |
| eager AUC | ≥ 0.75 | 0.74 at δ = 0.1 s, 0.67 at 0.4 s (AMI); 0.84 at 0.4 s (std) | not met on AMI |
| std test within 1 point of v3.2 | yes | AUC 0.977 vs 0.976 | met |

Two gates miss on AMI. Before going to level 2 (encoder fine-tune, which needs the baseline
retrain), the cheap knobs are: a lower learning rate with early stopping on dev AUC, heavier AMI
weighting, and using the excluded `no_floor` / `long_silence` silences as extra negatives. The
B2 streaming export does not depend on closing these gaps and can proceed in parallel.

## Reproduce

```
.venv/bin/python scripts/b1_items.py --out data/b1/items.parquet
modal volume put smart-turn-v2 data/b1/items.parquet data/b1/items.parquet
.venv/bin/python -c "from eot.features import expose_pooled as e; e('data/models/smart-turn-v3.2-gpu.onnx','data/models/smart-turn-v3.2-gpu.pooled.onnx'); e('data/models/smart-turn-multimodal-audio.onnx','data/models/smart-turn-multimodal-audio.pooled.onnx')"
modal volume put smart-turn-v2 data/models/smart-turn-v3.2-gpu.pooled.onnx data/models/smart-turn-v3.2-gpu.pooled.onnx
modal volume put smart-turn-v2 data/models/smart-turn-multimodal-audio.pooled.onnx data/models/smart-turn-multimodal-audio.pooled.onnx
.venv/bin/modal run --detach modal_features.py          # ~25 min on 8 L4s
modal volume get smart-turn-v2 data/b1/feats.npz data/b1/feats.npz
.venv/bin/python scripts/b1_train.py --backbone v32 --sources std,ami --epochs 6 --out runs/b1/v32_std_ami
```

## Training knobs (2026-10-24, after the first B1 round)

Tried, each as a 3-minute run on the cached features, scored on AMI test with thresholds fitted
on AMI dev (seed noise: ±0.01 AUC, ±0.03 recall at the 10 % budget):

| recipe | AUC 0.2 s | AUC 1.2 s | eager AUC 0.4 s | recall @ 10 % budget (FAR, mean latency) | FAR ≥ 1.5 s |
|---|---|---|---|---|---|
| original: lr 2e-3, 6 ep, select by rule score, AMI 0.5 (3 seeds) | 0.759 | 0.732 | 0.67 | 0.75 (0.080, 1.60 s) | 0.46 |
| lr 1e-3, 8 ep, select by dev AUC | 0.789 | 0.693 | 0.71 | 0.76 (0.095, 1.55 s) | 0.51 |
| lr 5e-4, 8 ep, select by dev AUC | 0.791 | 0.694 | 0.71 | 0.74 (0.086, 1.62 s) | 0.48 |
| **lr 1e-3, 8 ep, select by dev AUC, AMI 0.7 (3 seeds)** | **0.785 ± 0.009** | 0.730 ± 0.010 | 0.71 | 0.76 ± 0.03 (0.087, 1.56 s) | 0.46 |
| same, AMI 0.85 | 0.792 | 0.722 | 0.71 | 0.76 (0.097, 1.53 s) | 0.48 |
| AMI 0.7 + 22,491 long-pause negatives (2 seeds) | 0.788 | 0.722 | 0.70 | 0.72 (0.083, 1.60 s) | 0.43 |
| AMI 0.5 + long-pause negatives | 0.776 | 0.722 | 0.69 | 0.69 (0.082, 1.60 s) | 0.39 |
| AMI 0.7, dropout 0.3, weight decay 0.05 | 0.788 | 0.709 | 0.71 | 0.71 (0.082, 1.66 s) | 0.44 |
| AMI 0.7, dropout 0.3, wd 0.05, lr 5e-4, 4 ep | 0.796 | 0.705 | 0.71 | 0.71 (0.087, 1.64 s) | 0.45 |

Curves: `runs/b1/figures/lr_and_dev_curves.png`, `loss_curves.png`, `step_curves.png`, and from
now on `runs/b1/<run>/curves.png` for every run (loss per step split by head, learning rate, dev
ranking and rule metrics per epoch, selected epoch marked).

What the curves show: training loss falls monotonically through 8 epochs while dev AUC at the
stop peaks at epoch 1–2 and declines by 0.03–0.04 afterwards; dev AUC at 1.2 s rises until epoch
3–4 and flattens; dev eager AUC peaks early. The heads overfit after about two passes, and the
two ranking targets move against each other: selecting early buys stop-AUC (0.79) and costs
long-silence AUC (0.69–0.71), selecting late does the reverse (0.75 / 0.74).

Conclusions:

1. **No cheap knob closes the two missed gates.** Across 14 runs the rule metrics stay in one
   band: recall 0.69–0.80 at an actual FAR of 7–10 %, mean latency 1.5–1.7 s, FAR on pauses ≥ 1.5 s
   0.39–0.51. The eager AUC on AMI sits at 0.69–0.72 for every recipe.
2. **Long-pause negatives from the excluded silences do not help** (only 2,526 events; the
   ≥ 1.5 s FAR moves within noise and recall drops slightly).
3. **Regularisation and lower learning rates keep the stop-AUC but do not improve the rules.**
4. **Recipe kept for B1**: lr 1e-3, 8 epochs, checkpoint by dev AUC, AMI weight 0.7, dropout 0.1:
   stop-AUC 0.785 (from 0.759) at unchanged long-silence AUC and rule metrics, std test unchanged
   (0.976 vs 0.976).
5. **Level 1 has plateaued on AMI.** What is left is the encoder (level 2 fine-tuning, which
   needs the baseline retrain from whisper-tiny) and the data (the dyadic Seamless set, where the
   labels are cleaner than four-party meetings). The B2 streaming export proceeds with the
   recipe above.
