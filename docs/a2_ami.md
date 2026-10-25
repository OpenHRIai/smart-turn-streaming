# Stage A2: AMI Meeting Corpus end-of-turn events (commercial conversational data)

Status: 2026-10-24, dev, test and train built (all 682 headset files); raw data and derived events are on the Modal volume `smart-turn-v2`. Plan:
`docs/plan_online_eot.md` §6, Stage A2. Code: `eot/turn_rules.py` (rules), `eot/ami.py` (corpus
access), `scripts/a2_ami_mine.py`, `scripts/a2_ami_fetch.py`, `scripts/a2_ami_build.py`, tests in
`tests/test_turn_rules.py`. Data under `data/ami` and `data/a2` (git-ignored).

## Source and licence

AMI Meeting Corpus (University of Edinburgh and partners), 171 meetings, 100 h, four to five
participants each, per-speaker close-talking headset audio (16 kHz PCM16 mono), transcriber
segments with forced-aligned word times, per-participant close-up cameras. The annotations zip
(`ami_public_manual_1.6.2.zip`, release 1.7) ships `LICENCE.txt`: **Creative Commons Attribution
4.0 International**, which allows commercial use with attribution. The file is kept with the data
and must accompany any release built from it. Older mirrors (OpenSLR 16) carry an outdated
NC-SA notice; the Edinburgh distribution is the one used.

Splits follow the Kaldi AMI recipe (`egs/ami/s5/local/split_{dev,eval}.orig`): dev = ES2011a-d,
IS1008a-d, TS3004a-d, IB4001-4004, IB4010, IB4011 (18 meetings); test = ES2004a-d, IS1009a-d,
TS3003a-d, EN2002a-d (16); train = the remaining 137. Speakers recur across the a-d sessions of a
scenario but not across splits.

## Rules: the WS9 labelling, generalised to many participants

`eot/turn_rules.py` is a port of OpenVideoFDB's `ovfdb/endpoint/rules.py` with one change: "the
partner" becomes "any other participant". A silence is a VAD stop of speaker A with a gap of
≥ 0.2 s; it is **complete** when some other participant starts a non-backchannel utterance within
3 s and A stays quiet for 1 s after that onset, **incomplete** when A resumes within 5 s before
anyone else says more than a backchannel; otherwise excluded with a reason (`no_floor`: someone
else was speaking in the last 1 s; `short_turn`: fewer than 2 words and under 1 s of speech so
far; `collision`; `long_silence`; `tail`). The floor test, the next real onset and the turn cuts
all use the union of the others' speech. The dyadic tests from WS9 pass unchanged, plus
multi-party cases (a third party takes the floor, a third party blocks the floor, backchannels
from two listeners are ignored).

## Why the VAD comes from the audio, not the alignments

AMI's word times are contiguous inside a transcriber segment: each word ends where the next
starts, so pauses inside a segment are invisible. Mining the alignments alone
(`scripts/a2_ami_mine.py`, all 171 meetings, no audio) finds 25,275 labelled silences but their
incomplete pauses have a median gap of 1.71 s and 60 % are ≥ 1.5 s, i.e. only the between-segment
silences. That is not the deployed distribution (Seamless: median 0.52 s, 47 % under 0.5 s).

`scripts/a2_ami_build.py` therefore runs Silero VAD (ONNX, min silence 100 ms, pad 30 ms) on each
headset channel, which is what Pipecat's VAD would see, and **gates** the regions by the transcript:
a Silero region counts as this speaker's speech only if it overlaps one of their transcribed
segments (± 0.3 s) or one of their timed words. The gate is what removes crosstalk: raw Silero
speech on a headset is a median 1.66× the speaker's transcribed speech (the other three people
bleed in), gated speech is 0.74× (Silero splits at every 100 ms pause).

## Dev and test (built 2026-10-24; 135 headset files, 8.5 GB, 0 download failures)

| split | meetings | speaker-meetings | complete | incomplete | no_floor | short_turn | collision | long_silence | tail |
|---|---|---|---|---|---|---|---|---|---|
| dev | 18 | 72 | 734 | 3,681 | 3,688 | 898 | 232 | 764 | 12 |
| test | 16 | 63 | 511 | 4,304 | 3,807 | 1,292 | 209 | 517 | 9 |

- 9,230 labelled silences from 37 distinct participants (135 speaker-meeting pairs); 13.5 %
  complete (dev 16.6 %, test 10.6 %). The plan's gate asked for ≥ 5,000 silences from ≥ 40
  speakers with both classes ≥ 15 %: the count is met, the participant count is 37 and the test
  split sits at 10.6 % complete. Meetings have one floor holder and three listeners, so most
  stops are mid-turn; the class balance is what the task has.
- Incomplete pauses: median 0.6 s; 3,120 under 0.5 s, 3,835 at 0.5–1.5 s, 1,030 at ≥ 1.5 s (12.9 %).
  This matches Seamless (47 / 45 / 8 %) and gives the ≥ 1.5 s slice ~1,000 pauses to measure
  false alarms on.
- Completes: the next speaker starts at a median of 0.75 s after the offset, 15.5 % within 0.2 s and
  37.5 % within 0.5 s (Seamless: 0.80 s, 15 %, 33 %).
- Turn so far at the stop: median 10 words, 3.4 s of speech.

Examples (`a_text` = A's last words, `b_text` = the next speaker):

| label | A said | next |
|---|---|---|
| complete | "I'm thinking about a swordfish" | B after 0.62 s: "So what yeah it's this is bit of the swordfish…" |
| complete | "So we can decrease this one to four buttons" | 1.45 s: "Right now we have five." |
| incomplete | "Or just or is it more like" | A resumes after 0.6 s |
| incomplete | "Let's see Oh I have to say this was a little hard because" | 0.5 s |
| incomplete | "Yeah I'd uh yeah I'd probably put it uh" | 1.0 s |

Caveats carried from WS9: labels are by outcome (a speaker who was cut off counts as complete);
the gate's ± 0.3 s pad can let a loud neighbour extend a region by up to that much; meetings are
four-party, so "someone took the floor" is a weaker signal of A's intent than a dyadic partner
replying. Known AMI timing problems (EN2002a,c in test; EN2003a and TS3009c channel 3 in train)
are inherited; the audio-based VAD limits their effect to the word counts and texts.

## Train (built 2026-10-24; 547 headset files)

| split | meetings | speaker-meetings | complete | incomplete | no_floor | short_turn | collision | long_silence | tail |
|---|---|---|---|---|---|---|---|---|---|
| train | 137 | 541 | 5,607 | 37,732 | 31,175 | 9,938 | 1,998 | 5,481 | 98 |

All three splits together: 52,569 labelled silences, 13.0 % complete; incomplete pause median
0.5 s (42 % under 0.5 s, 11 % at or above 1.5 s); next speaker at a median 0.68 s after the offset
(17 % within 0.2 s, 40 % within 0.5 s). The train split is 4.6× the dev+test material and comes from
different scenario groups.

Download notes: the Edinburgh mirror serves about 6 MB/s in total; a few requests return a
transient HTTP 403 and two transfers were truncated, all recovered by re-running the resumable
fetch (every one of the 682 headset channels listed in `meetings.xml` exists). Meetings EN2001e,
EN2003a and EN2004a were rebuilt after their missing channels arrived; `a2_ami_build.py` now
re-aggregates the split manifests from every `events.jsonl` under `--out`, so a partial rebuild
never shrinks them.

## Outputs

- `data/a2/ami/<meeting>/events.jsonl`: every silence, labelled or excluded, with `t`, `gap_s`,
  `b_onset_s`, `b_speaker`, turn statistics, texts, headset `channel` and `audio` path;
  `vad_<L>.json` the gated VAD regions; `speakers.json` the per-channel statistics.
- `data/a2/ami/manifest_{dev,test,train}.parquet`: labelled events only, the input to the
  Stage A3 sampler (window = headset audio up to `t + τ` or `t − δ`, 8 s, as `eot.windows` does
  for clips; here no padding is ever needed because the recording continues).
- `data/a2/ami_events_all.parquet`: the alignment-only mining of all 171 meetings, kept for the
  comparison above.

## Where the data lives

Local: `data/ami/audio` (43 GB), `data/ami/annotations`, `data/a2/ami`. Modal volume
`smart-turn-v2`, same relative layout under `/vol/data` (headsets pushed from the local copy at
~25 MB/s, which beat the mirror's 6 MB/s; pipecat's parquet shards fetched on Modal from HF with
`hf_transfer`). See `modal_data.py` and `README_PRIVATE.md`.

## Next

Stage A3: one sampler over smart-turn-data clips (A1) and AMI tracks (A2) feeding Stage B1.
