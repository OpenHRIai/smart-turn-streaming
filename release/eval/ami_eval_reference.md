# AMI end-of-turn evaluation (dev, test; thresholds shipped and re-fitted on dev)

## AMI dev: AUC of P(complete) per silence offset

| tau (s) | n complete / incomplete | AUC heads | AUC frozen Smart Turn v3.2 | mean P on incompletes: heads / frozen |
|---|---|---|---|---|
| 0.2 | 734 / 3195 | 0.748 | 0.675 | 0.251 / 0.299 |
| 0.5 | 734 / 1827 | 0.722 | 0.640 | 0.398 / 0.414 |
| 1.2 | 734 / 624 | 0.698 | 0.616 | 0.727 / 0.450 |

## AMI test: AUC of P(complete) per silence offset

| tau (s) | n complete / incomplete | AUC heads | AUC frozen Smart Turn v3.2 | mean P on incompletes: heads / frozen |
|---|---|---|---|---|
| 0.2 | 511 / 3820 | 0.775 | 0.671 | 0.256 / 0.309 |
| 0.5 | 511 / 2256 | 0.776 | 0.638 | 0.357 / 0.427 |
| 1.2 | 511 / 688 | 0.738 | 0.577 | 0.713 / 0.533 |

## Decision rule on AMI test (511 complete / 3820 incomplete silences with a decision point; misses charged 3 s)

**Heads (this release)**

| rule | theta at 0.2 / 0.5 / 1.2 s | recall | FAR | p50 | p90 | mean latency | FAR by pause length <0.5 / 0.5-1.5 / >=1.5 s |
|---|---|---|---|---|---|---|---|
| one-shot at the VAD stop, theta 0.5 (stock rule) | 0.50 / - / - | 0.64 | 0.254 | 0.20 s | 3.00 s | 1.21 s | 0.19 / 0.24 / 0.46 |
| shipped table, 5% budget | 0.94 / 0.94 / 0.94 | 0.47 | 0.046 | 3.00 s | 3.00 s | 2.05 s | 0.00 / 0.03 / 0.21 |
| shipped table, 10% budget | 0.92 / 0.98 / 0.86 | 0.79 | 0.094 | 1.20 s | 3.00 s | 1.49 s | 0.01 / 0.04 / 0.50 |
| shipped table, 15% budget | 0.98 / 0.88 / 0.80 | 0.86 | 0.133 | 1.20 s | 3.00 s | 1.21 s | 0.00 / 0.09 / 0.61 |
| re-fitted on dev, 5% budget | 0.94 / 0.94 / 0.94 | 0.47 | 0.046 | 3.00 s | 3.00 s | 2.05 s | 0.00 / 0.03 / 0.21 |
| re-fitted on dev, 10% budget | 1.00 / 0.94 / 0.86 | 0.79 | 0.090 | 1.20 s | 3.00 s | 1.49 s | 0.00 / 0.04 / 0.50 |
| re-fitted on dev, 15% budget | 0.98 / 0.88 / 0.80 | 0.86 | 0.133 | 1.20 s | 3.00 s | 1.21 s | 0.00 / 0.09 / 0.61 |

**Frozen Smart Turn v3.2 (baseline, same protocol)**

| rule | theta at 0.2 / 0.5 / 1.2 s | recall | FAR | p50 | p90 | mean latency | FAR by pause length <0.5 / 0.5-1.5 / >=1.5 s |
|---|---|---|---|---|---|---|---|
| one-shot at the VAD stop, theta 0.5 (stock rule) | 0.50 / - / - | 0.51 | 0.303 | 0.20 s | 3.00 s | 1.56 s | 0.23 / 0.30 / 0.45 |
| re-fitted on dev, 5% budget | 1.00 / 1.00 / 0.96 | 0.37 | 0.056 | 3.00 s | 3.00 s | 2.34 s | 0.00 / 0.02 / 0.31 |
| re-fitted on dev, 10% budget | 1.00 / 1.00 / 0.26 | 0.71 | 0.106 | 1.20 s | 3.00 s | 1.72 s | 0.00 / 0.04 / 0.61 |
| re-fitted on dev, 15% budget | 1.00 / 1.00 / 0.02 | 0.87 | 0.137 | 1.20 s | 3.00 s | 1.44 s | 0.00 / 0.05 / 0.78 |

**No model**

| rule | theta at 0.2 / 0.5 / 1.2 s | recall | FAR | p50 | p90 | mean latency | FAR by pause length <0.5 / 0.5-1.5 / >=1.5 s |
|---|---|---|---|---|---|---|---|
| silence only, commit at 1.2 s | - / - / 0.00 | 1.00 | 0.180 | 1.20 s | 1.20 s | 1.20 s | 0.00 / 0.08 / 1.00 |

Scores cached in `./ami_eval_scores.npz`; audio from https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus (AMI Meeting Corpus, CC BY 4.0).
