# Phase 0 in Pipecat: end-to-end trigger latency, full output

Produced on 2026-10-17 by `measure_pipecat_latency.py` (pipecat 1.12.0, Apple M4, CPU) on 60 complete and 40 incomplete
silences sampled (seed 0) from the Seamless end-of-turn test set (OpenVideoFDB, `manifest/endpoint_naturalistic_test.jsonl`,
source audio under `data/seamless/raw`, local only, CC BY-NC). Read with `docs/design_streaming_eot.md` §9.4. The analyzer
is `local_smart_turn_multimodal.py` with the audio-only graph `smart-turn-multimodal-audio.onnx`.

```
for rule in one_shot far15_min_latency far10_min_latency; do
  python measure_pipecat_latency.py --manifest <OpenVideoFDB>/manifest/endpoint_naturalistic_test.jsonl --root <OpenVideoFDB> \
      --rule $rule --n-complete 60 --n-incomplete 40 --seed 0 --normalize-dbfs -23 --out runs/pipecat_latency/$rule.csv --quiet
done
python measure_pipecat_latency.py --report runs/pipecat_latency/*.csv
```

Each silence is replayed as 20 ms frames at real-time pace from 9 s before the dataset's VAD stop `t_raw` (the acoustic offset)
to 3.6 s after it through `VADProcessor(SileroVADAnalyzer)` → `UserTurnProcessor` with a VAD start strategy and
`TurnAnalyzerUserTurnStopStrategy(analyzer, wait_for_transcript=False)`. Latency is wall-clock time from the acoustic offset to
`on_user_turn_stopped`. "Recall" is a turn stop within the 3.6 s window (a `stop_secs` stop counts); a complete with no stop is
charged 3.2 s (VAD `stop_secs` 0.2 + `stop_secs` 3.0) in the latency columns. "Strict" false alarms are stops before the speaker
resumed (the dataset's `gap_s`); "effective" ones also count stops within the VAD's `start_secs` (0.2 s) after the resumption,
when pipecat could not yet have noticed it. Rules: `one_shot` = θ 0.5 at the VAD stop then `stop_secs` (the stock rule);
`far15_min_latency` = θ 0.971 / 0.636 / 0 at 0.2 / 0.6 / 1.2 s; `far10_min_latency` = 0.991 / 0.991 / 0.

## Replay level normalised to -23 dBFS RMS (the reported run)

| rule | n complete / incomplete | recall (turn stopped in window) | false alarms, strict / effective | latency from acoustic offset, completes: p50 / p90 / mean | from pipecat's VAD offset: p50 / p90 | committed at 0.2 / 0.6 / 1.2 s / stop_secs / none | model ms per query |
|---|---|---|---|---|---|---|---|
| far10_min_latency | 60 / 40 | 0.97 | 12.5% / 15.0% | 1.16 / 1.23 / 0.93 s | 1.25 / 1.26 s | 14 / 9 / 35 / 0 / 2 | 50 |
| far15_min_latency | 60 / 40 | 0.97 | 12.5% / 15.0% | 0.57 / 1.23 / 0.74 s | 0.66 / 1.26 s | 21 / 17 / 20 / 0 / 2 | 50 |
| one_shot | 60 / 40 | 0.90 | 2.5% / 2.5% | 0.24 / 3.20 / 1.50 s | 0.26 / 3.20 s | 35 / 0 / 0 / 19 / 6 | 52 |

Latency of the committed turns by committing offset (completes and incompletes), measured from pipecat's own VAD offset (its
VAD stop minus `stop_secs`), so the overhead over the nominal offset is the pipeline's: VAD frame granularity, frame handling,
the model run and the analyzer's one-frame poll.

| rule | committed at | n | p50 | p90 | median overhead over the nominal offset |
|---|---|---|---|---|---|
| far10_min_latency | 0.2 | 16 | 0.256 s | 0.261 s | +56 ms |
| far10_min_latency | 0.6 | 9 | 0.659 s | 0.664 s | +59 ms |
| far10_min_latency | 1.2 | 46 | 1.259 s | 1.260 s | +59 ms |
| far15_min_latency | 0.2 | 23 | 0.255 s | 0.264 s | +55 ms |
| far15_min_latency | 0.6 | 20 | 0.659 s | 0.660 s | +59 ms |
| far15_min_latency | 1.2 | 29 | 1.259 s | 1.260 s | +59 ms |
| one_shot | 0.2 | 41 | 0.255 s | 0.266 s | +55 ms |
| one_shot | stop_secs | 19 | 3.200 s | 3.201 s | +0 ms |

Rows where pipecat's VAD offset differs from the dataset's `t_raw` by more than 0.3 s: 22 of 203 committed turns (median |Δ|
over all: 98 ms). Pipecat's VAD stop lands ~0.12 s after `t_raw` on agreeing clips, i.e. its acoustic offset is ~80 ms before
`t_raw`, which is why the "from pipecat's VAD offset" latencies are ~80 ms above the "from acoustic offset" ones.

Where the false alarms happened (normalised run, 40 incompletes):

| rule | at 0.2 s | at 0.6 s | at 1.2 s | of which after the speaker had resumed by < 0.2 s |
|---|---|---|---|---|
| one_shot | 1 (pause 0.32 s) | – | – | 0 |
| far15_min_latency | 1 (0.32 s) | 1 (0.77 s) | 4 (pauses 1.09, 1.22, 1.41, 1.73 s) | 1 |
| far10_min_latency | 1 (0.32 s) | 0 | 5 (0.39*, 1.09, 1.22, 1.41, 1.73 s) | 1 |

\* a 1.2 s query scheduled from a VAD stop that pipecat placed ~1 s before `t_raw`. The sample has 3 incompletes with pauses
≥ 1.2 s and 13 with pauses ≥ 0.6 s.

The six one-shot completes without any stop: in four the speaker spoke again 1.9–2.5 s after the offset (a backchannel or a
continuation after the partner started), which reset the `stop_secs` countdown, so the one-shot rule never ended the turn; in one
pipecat's VAD kept reporting speech for 3.4 s after `t_raw`; one ended before the offset. The two re-query completes without a
stop are the latter two.

## Replay at the recordings' own level (not reported in the design doc)

The Seamless source recordings sit between -25 and -56 dBFS RMS. Pipecat's VAD gates on loudness (`min_volume` 0.6 on a -110…-10
LUFS scale) as well as on Silero's confidence, so at the recordings' own level it dropped soft endings (offsets 0.7–1.2 s before
`t_raw`) and never detected 6 of the 300 replays at all; 69 of 300 had no VAD stop after the offset (6 after normalisation). The
numbers below are what a very quiet microphone would give and are kept only to show the effect.

| rule | n complete / incomplete | recall (turn stopped in window) | false alarms, strict / effective | latency from acoustic offset, completes: p50 / p90 / mean | from pipecat's VAD offset: p50 / p90 | committed at 0.2 / 0.6 / 1.2 s / stop_secs / none | model ms per query |
|---|---|---|---|---|---|---|---|
| far10_min_latency | 60 / 40 | 0.88 | 7.5% / 20.0% | 1.10 / 3.20 / 1.07 s | 1.26 / 3.20 s | 8 / 8 / 37 / 0 / 7 | 49 |
| far15_min_latency | 60 / 40 | 0.82 | 7.5% / 20.0% | 0.75 / 3.20 / 1.13 s | 1.26 / 3.20 s | 13 / 12 / 24 / 0 / 11 | 48 |
| one_shot | 60 / 40 | 0.83 | 5.0% / 5.0% | 2.58 / 3.20 / 1.89 s | 3.20 / 3.20 s | 22 / 0 / 0 / 28 / 10 | 51 |

Rows where pipecat's VAD offset differs from the dataset's `t_raw` by more than 0.3 s: 60 of 207 committed turns.

## Synthetic check (`--wav`, macOS `say`)

A complete sentence ("I think we should probably meet on Tuesday afternoon, if that works for you."): one-shot p 0.958 → stop
at 0.27 s; re-query (far15) p 0.958 < 0.971 at 0.2 s, p 0.962 ≥ 0.636 at 0.6 s → stop at 0.68 s. A trailing filler ("So what I
was going to say is, um,"): p 0.0003 at every offset → one-shot stops at 3.22 s (`stop_secs`), re-query at 1.28 s.

Note on the windows the model sees. `BaseSmartTurn` empties its audio buffer after every committed turn, so a query's window runs
from the current turn's VAD start (minus `pre_speech_ms` and the VAD's `start_secs`) to the query, capped at 8 s; it is not the
fixed 8 s before the query that `eval_silence_requery.py` scores. On the synthetic sentence above the rule committed on the
mid-sentence pause (p 0.99 at 0.2 s), so the query at the sentence's end saw only the last clause (1.3 s, p 0.958); the whole
sentence cut at the same point scores p 0.05. The replays of the Seamless silences inherit this: a premature commit on a pause in
the 9 s pre-roll shortens the context for the decision at `t_raw`, as it would in a live pipeline.
