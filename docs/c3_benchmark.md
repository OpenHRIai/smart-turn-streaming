# Stage C3: the official Smart Turn benchmark, calibration, and behaviour per language

Status: 2026-10-25. Code: `scripts/c3_languages.py` (per-language tables from the B1 predictions),
`scripts/c3_calibrate.py` (Platt calibration of the heads for the clip-end regime),
`scripts/c3_fuse_onnx.py` (one graph for pipecat's benchmark), `scripts/c3_official_benchmark.py`
(driver for the vendored, unmodified `scripts/vendor/pipecat_benchmark.py`, pipecat-ai/smart-turn
`benchmark.py` at `4786657e`). Results in `runs/c3/`.

## Why run it at all

Pipecat's `benchmark.py` is the format Smart Turn users know: one ONNX graph, `smart-turn-data-v3.2-test`,
threshold 0.5, accuracy / precision / recall / F1 / false-positive rate overall, per language and per
source dataset. It measures the clip-end decision on isolated utterances, which is the stock model's
native regime and not where the streaming detector's gains are (no pauses, so the 0.5 s and 1.2 s
decisions, the eager stage and the cancel path never run). It is still the right regression guard:
anyone swapping the model in will ask whether it is at least as good at the thing Smart Turn does.

Two things had to be settled to make the comparison honest:

1. **Calibration.** The heads were trained with a class-weighted loss, so at the same ranking their
   probabilities sit above the frozen model's; at 0.5 they have more recall and more false positives
   (B1 regression guard). A monotone map fitted on the **training** split's clip-end windows (τ = 0 and
   0.2 s, 69,116 windows), Platt scaling `p' = sigmoid(a · logit(p) + b)` with a = 0.8251,
   b = −1.2576, changes no AUC and no fitted table (a table threshold θ maps to sigmoid(a · logit(θ) + b))
   but puts 0.5 at a comparable operating point. Both the raw and the calibrated heads are reported.
2. **One graph.** `scripts/c3_fuse_onnx.py` composes the pooled encoder, the last-second mel summary
   computed inside the graph from the same `input_features`, zeros for the τ features (τ = 0, not
   speaking), the heads, and the calibration into one ONNX with the stock interface (`input_features`
   (B, 80, 800) → `probability` (B, 1)); it reproduces the two-graph path to 5e-7. The upstream script
   is used as is for the metrics and the report; the data path it leaves to `train.py` (last 8 s or
   zero-padded at the start, `WhisperFeatureExtractor(chunk_length=8)`, `do_normalize=True`) is
   reproduced in the driver, and features are extracted once so every model scores the same batches.
   The one upstream helper that assumes a CUDA box (`device="cuda"` in the perf-only feature extraction)
   is replaced by the same call on the CPU.

## Calibration (smart-turn-data v3.2 test, 6,000 clips, `runs/c3/calibration.json`)

| τ of silence | model | AUC | accuracy at 0.5 | recall at 0.5 | FPR at 0.5 |
|---|---|---|---|---|---|
| 0 | heads, raw | 0.975 | 0.922 | 0.965 | 0.120 |
| 0 | **heads, calibrated** | 0.975 | **0.926** | 0.935 | 0.082 |
| 0 | frozen Smart Turn v3.2 | 0.978 | 0.925 | 0.907 | 0.056 |
| 0.2 | heads, calibrated | 0.976 | 0.929 | 0.951 | 0.093 |
| 0.2 | frozen v3.2 | 0.976 | 0.929 | 0.939 | 0.082 |
| 1.2 | heads, calibrated | 0.972 | **0.919** | 0.960 | 0.121 |
| 1.2 | frozen v3.2 | 0.964 | 0.899 | 0.963 | 0.166 |
| 2.5 | heads, calibrated | 0.965 | **0.899** | 0.967 | 0.168 |
| 2.5 | frozen v3.2 | 0.955 | 0.871 | 0.980 | 0.237 |

At the clip end the calibrated heads and the frozen model are equal to within 0.1 point of accuracy;
after 1.2 and 2.5 s of silence the heads are 2 to 3 points ahead at the same threshold, which is the
silence awareness showing through even in this regime.

## Official benchmark (full `smart-turn-data-v3.2-test`, 31,527 clips, Apple M4, CPU)

Upstream's metrics at threshold 0.5 on the full test set (49.8 % complete), three graphs on the same
feature batches (`runs/c3/official/`):

| model | accuracy | precision | recall | F1 | FPR | FNR | CPU inference p50 (batch 1) |
|---|---|---|---|---|---|---|---|
| Smart Turn v3.2 (fp32) | **93.71 %** | 0.931 | 0.944 | 0.937 | 3.51 % | 2.78 % | 10.2 ms |
| streaming heads, calibrated (`smart-turn-streaming-clipend.onnx`) | 93.23 % | 0.919 | 0.947 | 0.933 | 4.15 % | 2.62 % | 10.3 ms |
| streaming heads, raw probabilities | 92.33 % | 0.885 | 0.972 | 0.927 | 6.28 % | 1.38 % | 10.3 ms |

Per language, accuracy / FPR, Smart Turn v3.2 vs the calibrated heads:

| language | n | v3.2 | heads | | language | n | v3.2 | heads |
|---|---|---|---|---|---|---|---|---|
| English | 7820 | 94.7 / 3.0 | 94.1 / 3.3 | | Polish | 974 | 95.2 / 2.6 | 95.4 / 3.1 |
| Spanish | 1783 | 91.5 / 4.5 | 91.3 / 5.7 | | Indonesian | 971 | 95.5 / 3.1 | 94.9 / 4.0 |
| Russian | 1468 | 94.4 / 3.4 | 94.1 / 4.3 | | Turkish | 966 | 97.0 / 1.7 | 97.3 / 1.2 |
| Portuguese | 1398 | 95.5 / 2.6 | 95.1 / 3.7 | | Arabic | 947 | 89.1 / 7.1 | 87.5 / 8.2 |
| Dutch | 1398 | 96.9 / 1.8 | 96.6 / 2.1 | | Chinese | 929 | 90.5 / 5.3 | 89.7 / 6.5 |
| German | 1322 | 96.6 / 2.2 | 96.1 / 2.6 | | Ukrainian | 929 | 95.0 / 2.7 | 94.5 / 3.9 |
| Hindi | 1284 | 92.8 / 3.7 | 92.2 / 5.1 | | Korean | 889 | 97.6 / 1.1 | 97.4 / 1.3 |
| French | 1252 | 94.7 / 3.0 | 95.0 / 3.0 | | Japanese | 834 | 97.1 / 1.3 | 96.6 / 1.8 |
| Norwegian | 1014 | 93.0 / 3.6 | 91.9 / 3.9 | | Italian | 782 | 95.0 / 2.6 | 94.8 / 3.1 |
| Finnish | 1010 | 95.2 / 2.5 | 94.9 / 2.6 | | Danish | 779 | 93.6 / 3.6 | 93.3 / 3.3 |
| Vietnamese | 1004 | 82.5 / 9.6 | 82.1 / 11.0 | | Marathi | 774 | 88.1 / 6.8 | 86.7 / 8.5 |
| Bengali | 1000 | 85.1 / 7.5 | 84.6 / 9.1 | | | | | |

Reading:

1. **On the official benchmark the calibrated heads trail Smart Turn v3.2 by 0.5 points of accuracy**
   (93.2 % vs 93.7 %), with 0.6 points more false positives and 0.2 fewer misses. The ranking is the
   same (AUC 0.975 vs 0.978 on the balanced subset); what remains is the operating point: Platt scaling
   fitted on the training split lands slightly to the recall side of v3.2 on the test split. Raw, without
   the calibration, the gap is 1.4 points, all of it false positives. For a user who swaps the model in and
   keeps 0.5, this is the number to expect at the clip end; the rule tables, not 0.5, are the deployed
   operating points.
2. **Per language the picture is the same within noise**: 19 of 23 languages within 0.7 points, three
   ahead (French, Polish, Turkish), the largest gaps Arabic −1.6, Marathi −1.4, Norwegian −1.1. The
   languages both models find hard are the same (Vietnamese, Bengali, Marathi, Arabic).
3. **Cost at the clip end is the stock model's plus the heads**: 10.3 vs 10.2 ms per 8 s window on the M4
   (the benchmark's own single-sample timing; the heads add 0.1 ms).
4. **This is the regression guard, not the result.** The benchmark has no pauses, so the re-query, the
   eager stage and the cancel path never run; the headline numbers stay the AMI and Pipecat replay ones
   (`b3_pipecat.md`, `c1_eager.md`). The fused clip-end graph is shipped on the Hub as a drop-in for
   anyone who wants the stock interface and the stock benchmark.

## Per language (B1 predictions, 6,000-clip balanced test subset, `runs/c3/languages.md`)

Two questions per language: does the ranking hold at the clip end (AUC at τ = 0), and does the decision
hold through silence after an unfinished utterance. For the second, each model's threshold is set to
5 % false alarms at τ = 0 over the whole set (heads 0.954, frozen 0.594) and the false-alarm rate is read
at τ = 2.5 s, where the clip has been padded with its own room tone. Cells: heads / frozen v3.2.

| language | clips | AUC τ 0 | AUC τ 2.5 | recall τ 0 | FAR τ 0 | **FAR τ 2.5** |
|---|---|---|---|---|---|---|
| English | 1505 | 0.977 / 0.979 | 0.974 / 0.967 | 0.86 / 0.86 | 0.045 / 0.037 | **0.072 / 0.148** |
| Spanish | 345 | 0.966 / 0.969 | 0.940 / 0.936 | 0.85 / 0.91 | 0.066 / 0.084 | 0.120 / 0.295 |
| Portuguese | 279 | 0.993 / 0.992 | 0.987 / 0.987 | 0.93 / 0.95 | 0.042 / 0.042 | 0.049 / 0.174 |
| Russian | 273 | 0.966 / 0.964 | 0.957 / 0.952 | 0.98 / 0.97 | 0.083 / 0.075 | 0.105 / 0.180 |
| German | 271 | 0.985 / 0.989 | 0.973 / 0.963 | 0.93 / 0.93 | 0.037 / 0.044 | 0.096 / 0.184 |
| Dutch | 257 | 0.979 / 0.981 | 0.979 / 0.964 | 0.95 / 0.96 | 0.049 / 0.049 | 0.130 / 0.268 |
| Hindi | 244 | 0.984 / 0.985 | 0.974 / 0.973 | 0.87 / 0.87 | 0.040 / 0.032 | 0.104 / 0.280 |
| French | 222 | 0.984 / 0.987 | 0.971 / 0.963 | 0.93 / 0.94 | 0.029 / 0.038 | 0.086 / 0.181 |
| Bengali | 197 | 0.918 / 0.926 | 0.883 / 0.896 | 0.71 / 0.70 | 0.109 / 0.082 | 0.182 / 0.409 |
| Norwegian | 187 | 0.976 / 0.978 | 0.967 / 0.961 | 0.86 / 0.93 | 0.021 / 0.032 | 0.105 / 0.347 |
| Polish | 182 | 0.977 / 0.985 | 0.977 / 0.965 | 0.94 / 0.94 | 0.072 / 0.082 | 0.072 / 0.144 |
| Korean | 182 | 1.000 / 1.000 | 0.999 / 0.997 | 0.95 / 0.97 | 0.000 / 0.000 | 0.012 / 0.071 |
| Finnish | 182 | 0.983 / 0.992 | 0.970 / 0.951 | 0.91 / 0.92 | 0.024 / 0.024 | 0.107 / 0.417 |
| Indonesian | 179 | 0.983 / 0.986 | 0.970 / 0.960 | 0.94 / 0.97 | 0.032 / 0.022 | 0.054 / 0.204 |
| Turkish | 176 | 0.997 / 0.998 | 0.994 / 0.997 | 0.96 / 0.98 | 0.012 / 0.012 | 0.058 / 0.174 |
| Vietnamese | 173 | 0.893 / 0.906 | 0.851 / 0.823 | **0.48 / 0.70** | 0.094 / 0.106 | 0.176 / 0.447 |
| Marathi | 171 | 0.945 / 0.956 | 0.915 / 0.907 | 0.67 / 0.70 | 0.026 / 0.039 | 0.156 / 0.416 |
| Ukrainian | 169 | 0.961 / 0.971 | 0.944 / 0.937 | 0.91 / 0.91 | 0.056 / 0.044 | 0.067 / 0.200 |
| Chinese | 167 | 0.934 / 0.941 | 0.928 / 0.920 | 0.78 / 0.80 | 0.119 / 0.107 | 0.119 / 0.238 |
| Arabic | 166 | 0.963 / 0.966 | 0.933 / 0.933 | 0.87 / 0.91 | 0.067 / 0.112 | 0.124 / 0.258 |
| Japanese | 163 | 0.990 / 0.996 | 0.995 / 0.997 | 0.95 / 0.97 | 0.013 / 0.013 | 0.039 / 0.066 |
| Danish | 160 | 0.952 / 0.970 | 0.935 / 0.945 | 0.80 / 0.91 | 0.093 / 0.107 | 0.173 / 0.453 |
| Italian | 150 | 0.981 / 0.976 | 0.968 / 0.960 | 0.93 / 0.93 | 0.038 / 0.038 | 0.101 / 0.190 |
| **all** | 6000 | 0.975 / 0.978 | 0.965 / 0.955 | 0.87 / 0.89 | 0.050 / 0.050 | **0.094 / 0.225** |

Reading:

1. **The silence awareness transfers to every language in the set.** At a matched 5 % false-alarm rate
   at the clip end, the frozen model's false alarms on unfinished utterances rise to 22.5 % after 2.5 s
   of silence, the heads' to 9.4 %; the heads are lower in all 23 languages, by a factor of 2 to 4 in
   most (Finnish 10.7 % vs 41.7 %, Norwegian 10.5 % vs 34.7 %, Danish 17.3 % vs 45.3 %). This is the half
   of the behaviour the isolated-utterance data can test: holding the decision through silence after an
   incomplete utterance. The other half, the hazard of a real pause that resumes, was only trained and
   measured on English (AMI).
2. **Ranking at the clip end is unchanged within noise**, as the regression guard said: 20 of 23
   languages within 0.01 AUC of the frozen model, Danish −0.018, Finnish −0.009, Bengali −0.008.
3. **At the matched threshold a few languages lose recall at the clip end**, Vietnamese most (0.48 vs
   0.70 on 173 clips), then Danish (0.80 vs 0.91), Norwegian (0.86 vs 0.93) and Spanish (0.85 vs 0.91).
   The heads' scores on those languages are shifted down relative to the global threshold; a
   per-language threshold, or more data from them in training (each is about 1,000 clips of 77k), is the
   fix if those languages matter for a deployment. The weakest languages for both models (Vietnamese,
   Bengali, Chinese, Marathi) are the same ones, which points at the data rather than the heads.

## What this does and does not say about languages

The encoder is Whisper-tiny and `smart-turn-data` covers 23 languages (60 % English, 7 % Spanish, about
1,000 clips each for the rest; 39 % TTS), so the clip-end decision is multilingual by construction and
the silence-holding behaviour, learned from AMI's English pauses and the τ-padded clips of all languages,
carries over. What has not been measured in any language but English is the conversational hazard, a
pause that resumes, because that needs per-speaker dialogue audio. Candidates are research-licensed
(TalkBank's CABank conversational corpora, CallHome subsets; CC BY-NC-SA, evaluation only, as Seamless is
for the NC model) or the company's own recordings planned for R3 if some are not in English.

## Reproduce

```
.venv/bin/python scripts/c3_languages.py --run runs/b1/knob_ami07_auc --out runs/c3/languages.md
.venv/bin/python scripts/c3_calibrate.py --run runs/b1/knob_ami07_auc --out runs/c3/calibration.json          # loads the 5 GB feature cache
.venv/bin/python scripts/c3_fuse_onnx.py --calibration runs/c3/calibration.json --out data/models/smart-turn-streaming-clipend.onnx
.venv/bin/python scripts/c3_fuse_onnx.py --no-calibration --out data/models/smart-turn-streaming-clipend-raw.onnx
.venv/bin/python scripts/c3_official_benchmark.py --model data/models/smart-turn-v3.2-gpu.onnx:smart-turn-v3.2 \
    --model data/models/smart-turn-streaming-clipend.onnx:streaming-heads-calibrated \
    --model data/models/smart-turn-streaming-clipend-raw.onnx:streaming-heads-raw --out-dir runs/c3/official   # ~30 min
```
