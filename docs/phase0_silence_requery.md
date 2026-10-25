# Phase 0: silence-aware re-query, full output

Produced on 2026-10-17 by `eval_silence_requery.py` on the Seamless end-of-turn test set (OpenVideoFDB, branch
`ws9-endpointing-private`, `manifest/endpoint_naturalistic_test.jsonl`; media under `data/endpoint` and `data/seamless`,
local only, CC BY-NC). Read with `docs/design_streaming_eot.md` §9. Thresholds in the first rules table are fitted on the
whole set; the second table re-fits them on half of the conversations and scores the other half.

```
python eval_silence_requery.py --manifest <OpenVideoFDB>/manifest/endpoint_naturalistic_test.jsonl --root <OpenVideoFDB> \
    --model smart-turn-multimodal-audio.onnx --out runs/silence_requery --check-clips
```

Model `smart-turn-multimodal-audio.onnx` (audio-only); 2300 silences (417 complete, 1883 incomplete) from 133 speaker files; offsets 0.2, 0.6, 1.2 s after the VAD stop; timeout 3 s for misses. Model compute: nan ms per window (batched, 0 windows). Latencies are measured from the VAD stop and exclude compute time.

Re-cut windows vs stored clips at offset 0.2 s: max sample |Δ| nan, max |Δp| 0.0358, mean |Δp| 0.0001, rows with |Δp| > 0.05: 0.

## Queries available per offset (silence still ongoing)

| offset | completes | incompletes | AUC | mean p (complete) | mean p (incomplete) |
|---|---|---|---|---|---|
| 0.2 s | 417 | 1883 | 0.801 | 0.52 | 0.14 |
| 0.6 s | 417 | 785 | 0.779 | 0.63 | 0.23 |
| 1.2 s | 416 | 226 | 0.760 | 0.60 | 0.24 |

## Decision rules (in-sample thresholds)

| rule | θ table (0.2 / 0.6 / 1.2 s) | recall | FAR | p50 latency | p90 latency | mean latency |
|---|---|---|---|---|---|---|
| one-shot, θ 0.5 (current) | 0.500 / – / – | 0.52 | 0.133 | 0.20 s | 3.00 s | 1.54 s |
| re-query, θ 0.5 at every offset | 0.500 / 0.500 / 0.500 | 0.69 | 0.166 | 0.20 s | 3.00 s | 1.17 s |
| silence only, commit at 0.6 s | – / any / – | 1.00 | 0.417 | 0.60 s | 0.60 s | 0.60 s |
| silence only, commit at 1.2 s | – / – / any | 1.00 | 0.120 | 1.20 s | 1.20 s | 1.20 s |
| one-shot, FAR ≤ 5% | 0.978 / – / – | 0.27 | 0.050 | 3.00 s | 3.00 s | 2.25 s |
| re-query, FAR ≤ 5%, max recall | 0.997 / 0.998 / 0.008 | 0.79 | 0.049 | 1.20 s | 3.00 s | 1.56 s |
| re-query, FAR ≤ 5%, min latency | 0.997 / 0.997 / 0.012 | 0.78 | 0.049 | 1.20 s | 3.00 s | 1.53 s |
| one-shot, FAR ≤ 10% | 0.858 / – / – | 0.43 | 0.100 | 3.00 s | 3.00 s | 1.79 s |
| re-query, FAR ≤ 10%, max recall | 0.997 / 0.996 / 0.001 | 0.96 | 0.100 | 1.20 s | 1.20 s | 1.17 s |
| re-query, FAR ≤ 10%, min latency | 0.991 / 0.991 / 0.001 | 0.93 | 0.100 | 1.20 s | 1.20 s | 1.06 s |
| one-shot, FAR ≤ 15% | 0.317 / – / – | 0.56 | 0.150 | 0.20 s | 3.00 s | 1.42 s |
| re-query, FAR ≤ 15%, max recall | 0.997 / 0.983 / 0.000 | 1.00 | 0.149 | 1.20 s | 1.20 s | 0.95 s |
| re-query, FAR ≤ 15%, min latency | 0.971 / 0.636 / 0.001 | 0.94 | 0.149 | 0.60 s | 1.20 s | 0.82 s |

## Decision rules, 2-fold cross-fitted by conversation (thresholds fitted on the other half)

| rule | θ table (0.2 / 0.6 / 1.2 s) | recall | FAR | p50 latency | p90 latency | mean latency |
|---|---|---|---|---|---|---|
| one-shot, FAR ≤ 5% | per fold | 0.26 | 0.048 | 3.00 s | 3.00 s | 2.26 s |
| re-query, FAR ≤ 5%, max recall | per fold | 0.80 | 0.049 | 1.20 s | 3.00 s | 1.57 s |
| re-query, FAR ≤ 5%, min latency | per fold | 0.72 | 0.056 | 1.20 s | 3.00 s | 1.57 s |
| one-shot, FAR ≤ 10% | per fold | 0.44 | 0.099 | 3.00 s | 3.00 s | 1.76 s |
| re-query, FAR ≤ 10%, max recall | per fold | 0.96 | 0.101 | 1.20 s | 1.20 s | 1.15 s |
| re-query, FAR ≤ 10%, min latency | per fold | 0.94 | 0.106 | 1.20 s | 1.20 s | 1.04 s |
| one-shot, FAR ≤ 15% | per fold | 0.56 | 0.152 | 0.20 s | 3.00 s | 1.44 s |
| re-query, FAR ≤ 15%, max recall | per fold | 1.00 | 0.157 | 1.20 s | 1.20 s | 0.92 s |
| re-query, FAR ≤ 15%, min latency | per fold | 0.94 | 0.150 | 0.60 s | 1.20 s | 0.83 s |

## Re-query: commits per offset (in-sample tables)

| target FAR, objective | completes @ 0.2 s | completes @ 0.6 s | completes @ 1.2 s | missed (timeout) | false alarms @ 0.2 s | false alarms @ 0.6 s | false alarms @ 1.2 s |
|---|---|---|---|---|---|---|---|
| 5%, max recall | 6 | 0 | 324 | 87 | 2 | 0 | 90 |
| 5%, min latency | 6 | 32 | 288 | 91 | 2 | 15 | 76 |
| 10%, max recall | 6 | 59 | 336 | 16 | 2 | 24 | 162 |
| 10%, min latency | 68 | 67 | 254 | 28 | 46 | 34 | 108 |
| 15%, max recall | 6 | 161 | 250 | 0 | 2 | 90 | 188 |
| 15%, min latency | 122 | 137 | 132 | 26 | 109 | 100 | 72 |

## False alarms by pause length (incompletes)

| rule | < 0.5 s (n=891) | 0.5–1.5 s (n=846) | ≥ 1.5 s (n=146) |
|---|---|---|---|
| one-shot, θ 0.5 | 0.09 | 0.15 | 0.28 |
| one-shot, FAR ≤ 5% | 0.03 | 0.06 | 0.12 |
| re-query, FAR ≤ 5%, max recall | 0.00 | 0.03 | 0.45 |
| re-query, FAR ≤ 5%, min latency | 0.00 | 0.04 | 0.42 |
| one-shot, FAR ≤ 10% | 0.06 | 0.12 | 0.25 |
| re-query, FAR ≤ 10%, max recall | 0.00 | 0.09 | 0.79 |
| re-query, FAR ≤ 10%, min latency | 0.02 | 0.09 | 0.66 |
| one-shot, FAR ≤ 15% | 0.10 | 0.17 | 0.32 |
| re-query, FAR ≤ 15%, max recall | 0.00 | 0.16 | 1.00 |
| re-query, FAR ≤ 15%, min latency | 0.04 | 0.18 | 0.66 |

