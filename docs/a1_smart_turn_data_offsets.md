# Stage A1: acoustic offsets and τ / pre-offset windows on smart-turn-data v3.2

Status: 2026-10-24. Plan: `docs/plan_online_eot.md` §6, Stage A1. Code: `eot/offsets.py`,
`eot/windows.py`, `scripts/a1_offsets.py`, `scripts/a1_tau_baseline.py`, tests in
`tests/test_windows.py`. Data under `data/` (git-ignored).

## What A1 produces

Pipecat's `smart-turn-data-v3.2` (CC BY 4.0; train 270,946 clips / 41.4 GB, test 31,527 clips /
4.8 GB, 23 languages) is one utterance per clip with a complete / incomplete label. The clips are
downloaded as parquet shards to `data/smart_turn_data/{train,test}`. For every clip A1 records:

- `t_off`: the acoustic offset (end of the last Silero VAD speech segment, ONNX, threshold 0.5,
  min silence 100 ms, speech pad 30 ms), plus `t_on`, segment count, speech seconds;
- `trailing_s`: room tone already in the clip after `t_off`; `tail_dbfs`, `floor_dbfs`,
  `speech_dbfs`, `peak_dbfs`;
- `t_off_energy`: an energy-based offset for QA (end of the last 20 ms frame above
  max(floor + 12 dB, −50 dBFS)).

Output: `data/a1/{test,train}_offsets.parquet`, one row per clip, joined to the source by
`(shard, row)` and `id`.

```
.venv/bin/python scripts/a1_offsets.py --parquet-dir data/smart_turn_data/test  --out data/a1/test_offsets.parquet  --workers 8   # 198 s
.venv/bin/python scripts/a1_offsets.py --parquet-dir data/smart_turn_data/train --out data/a1/train_offsets.parquet --workers 8
```

Training windows are **not stored**. `eot.windows` cuts them on the fly from the clip and `t_off`:

| kind | window end | `p_end` label | `p_eager` label |
|---|---|---|---|
| `tau` | `t_off + τ`, τ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5} s | clip label | undefined |
| `pre` | `t_off − δ`, δ ∈ {0.1, 0.2, 0.3, 0.4, 0.6, 0.8} s, only while ≥ 0.5 s of speech remains | 0 | 1 if complete and δ ≤ 0.4 s |
| `mid` | random point ≥ 1 s into the speech and ≥ 1 s before the offset | 0 | 0 |

Where a `tau` window runs past the end of the clip, the tail is filled with the clip's own room
tone: the audio after `t_off` is tiled in 0.5 s pieces with random crops, random reversal and
10 ms crossfades. Clips with less than 50 ms of tail (the majority, see below) get Gaussian noise
at the level of the clip's quietest 20 ms frames, capped at −45 dBFS. Windows are 8 s, with zeros
before the clip start, as the upstream trainer does.

## Test split: what the clips look like (31,527 clips, 0 decode errors, 4 without speech)

| source | n | complete | synthetic | median duration | median trailing room tone | clips with < 50 ms tail |
|---|---|---|---|---|---|---|
| chirp3_1 | 16,254 | 0.50 | yes | 7.2 s | 0.02 s | 66 % |
| chirp3_2 | 8,428 | 0.50 | yes | 6.5 s | 0.02 s | 69 % |
| liva_1 | 3,831 | 0.50 | no | 12.2 s | 0.84 s | 1 % |
| midcentury_1 | 1,044 | 0.48 | no | 3.7 s | 0.04 s | 56 % |
| mundo_1 | 496 | 0.50 | no | 4.7 s | 0.00 s | 87 % |
| human_5 | 402 | 0.44 | no | 4.3 s | 0.27 s | 26 % |
| rime_2 | 394 | 0.42 | yes | 5.9 s | 1.43 s | 6 % |
| orpheus_* (3) | 484 | 0.49 | yes | 5–7 s | 0.02 s | 67–78 % |
| chirp3_3_short | 104 | 0.46 | yes | 1.0 s | 0.00 s | 89 % |
| human_convcollector_1 | 90 | 0.42 | no | 2.9 s | 0.15 s | 36 % |

- 81 % of clips are synthetic (TTS). Only `liva_1` and `rime_2` carry real room tone of any length.
- **Trailing room tone is almost absent**: median 0.04 s; 79 % of clips have less than 0.2 s after
  the offset and 35 % end exactly at the offset. So the published models were trained on windows
  ending *at* the offset, not 0.2 s into the silence as Pipecat queries them. Every τ > 0 variant
  of these clips is synthesised room tone, which is why real conversational pauses (AMI, Stage A2)
  are still needed for the hazard.
- Silero's offset sits a median 0.18 s after the energy offset (p10 0.06, p90 0.34 s; 4.6 % differ
  by more than 0.5 s). The VAD offset is used as `t_off` because it is what Pipecat's Silero
  reports; the energy offset stays in the table for audits.
- Speech per clip: median 6.4 s (p10 3.1, p90 10.8). 65 % of clips have 2 or more VAD segments.

## Train split (270,946 clips, 83 shards, 0 decode errors, 34 without speech; 1,327 s on 8 workers)

Same picture as the test split: 49.6 % complete, 82.4 % synthetic, 23 languages; per-source
counts scale by ≈ 9× (chirp3_1 146,627, chirp3_2 67,646, liva_1 30,922, midcentury_1 8,489,
mundo_1 3,986, human_5 3,527, rime_2 3,505, orpheus_* 4,773, chirp3_3_short 792,
human_convcollector_1 679). Trailing room tone: median 0.04 s, 79 % under 0.2 s; speech per
clip median 6.4 s; VAD offset a median 0.18 s after the energy offset (p90 0.34 s).
Rows: `data/a1/train_offsets.parquet`.

## Baseline: the current model on τ and pre-offset windows

`scripts/a1_tau_baseline.py`, 600 test clips (300 complete / 300 incomplete, seed 0), the audio-only
ONNX of the published multimodal model (`smart-turn-multimodal-audio.onnx`), Whisper features as in
`train.py`:

| window | mean P, complete | mean P, incomplete | AUC | recall @ 0.5 | FAR @ 0.5 |
|---|---|---|---|---|---|
| stored clip (last 8 s) | 0.932 | 0.063 | 0.989 | 0.95 | 0.06 |
| τ = 0.0 s | 0.892 | 0.056 | 0.985 | 0.91 | 0.05 |
| τ = 0.2 s | 0.934 | 0.073 | 0.984 | 0.94 | 0.07 |
| τ = 0.5 s | 0.928 | 0.079 | 0.980 | 0.95 | 0.08 |
| τ = 0.8 s | 0.928 | 0.100 | 0.978 | 0.95 | 0.09 |
| τ = 1.2 s | 0.937 | 0.108 | 0.978 | 0.95 | 0.10 |
| τ = 1.8 s | 0.935 | 0.161 | 0.965 | 0.96 | 0.16 |
| τ = 2.5 s | 0.937 | 0.188 | 0.963 | 0.95 | 0.19 |

| δ before the offset | mean P, complete clips | mean P, incomplete clips | AUC |
|---|---|---|---|
| 0.1 s | 0.848 | 0.045 | 0.980 |
| 0.2 s | 0.747 | 0.057 | 0.960 |
| 0.3 s | 0.585 | 0.085 | 0.900 |
| 0.4 s | 0.450 | 0.147 | 0.797 |
| 0.6 s | 0.262 | 0.220 | 0.628 |
| 0.8 s | 0.219 | 0.194 | 0.602 |

Reading:

1. **The silence drift reproduces in-domain.** With nothing but synthesised room tone appended,
   the false-alarm rate on incomplete clips at threshold 0.5 rises from 5 % at the offset to 19 %
   at 2.5 s, and AUC falls from 0.985 to 0.963. This is the same failure seen on Seamless (mean
   P on incomplete pauses 0.14 → 0.24 from 0.2 s to 1.2 s). It gives Stage B1 its in-domain gate:
   FAR at τ = 2.5 s should come back to the τ = 0 level (≈ 5 %) at unchanged recall.
2. **The eager signal is already in the encoder.** Without any training for it, windows cut
   0.3 s before the offset separate complete from incomplete clips at AUC 0.90, and 0.2 s before
   at 0.96. A head trained on the `pre` windows should push the usable horizon out; the design's
   400 ms horizon sits at AUC 0.80 today. These clips are mostly TTS with clean endings, so the
   conversational number (AMI, Seamless) will be lower.
3. **The stored clip and τ = 0.2 s agree** (0.989 vs 0.984 AUC, both at 94–95 % recall), so the
   window construction does not change what the model sees for the case it was trained on.

## Next

Train-split offsets (`data/a1/train_offsets.parquet`) complete the A1 deliverable; the sampler
(`eot.windows.enumerate_items`) then feeds Stage B1. Stage A2 (AMI) supplies the real pauses that
this data lacks.
