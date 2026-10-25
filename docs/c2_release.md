# Stage C2: the encoder refresh on a worker thread, and the R1 release files

Status: 2026-10-25. Code: `eot/streaming.py` (`StreamingEOT(worker=True)`), `eot/pipecat_analyzer.py`
(`StreamingEOTAnalyzer`, worker on by default), `scripts/b2_validate.py` (worker configurations, real-time
pace), `scripts/b3_replay.py` (`--no-worker`, `--lead`, `--wait`, pipeline-task timings), `scripts/c2_report.py`,
tests in `tests/test_streaming.py` (six new) and `tests/test_release_eval.py`. Release sources in `release/`
(model card, attribution, licences, rule tables, evaluation script and manifests, `stage.py`); the staged
Hugging Face directory is `runs/c2/hf_stage/smart-turn-streaming-eot/`. Results in `runs/c2/`. Pipecat 1.14.0,
Apple M4, CPU only. The upstream Pipecat PR (`pop_eager()` + the eager strategy) is a separate session.

## 1. Worker thread

B3 measured all compute on the pipeline task: a 0.7 ms heads hop every 100 ms and a 22 ms encoder refresh
seven times per silence and every 320 ms during speech, so the frame that lands on a silence-clock point
held the event loop for 25–40 ms (`b2_streaming.md`). The design budget is a decision-path burst of ≤ 5 ms
(`design_streaming_eot.md` §2, plan §6 B2).

### What moved where

| path | thread | work | when |
|---|---|---|---|
| mel | pipeline task | one 10 ms log-mel column per frame | every frame |
| heads hop | pipeline task | mel20 + cached pooled vector + τ → p_end, p_eager | every 100 ms and at each silence-clock point |
| **encoder refresh** | **worker** (`ThreadPoolExecutor(1)`) | 8 s window → pooled vector | requested from the pipeline task; the window (80 × 800, 0.6 ms to assemble) is handed over, the result is applied at the next hop |

Three details make the hand-off cheap in decisions, not just in milliseconds:

1. **A silence-clock decision waits for its own vector.** At each grid point (τ = 0.2, 0.5, 0.8, 1.2, 1.8,
   2.5 s) the refresh is requested and the hop is postponed, frame by frame, until the result is applied or
   `max_defer_s` (80 ms) has passed: the decision is then the one the inline detector takes, 20–60 ms later
   (the encoder's time, §1.3). A free timer would give a vector 0–320 ms old instead (B2: 0.04–0.06 AUC).
   The hop output records `pending` (requests not yet applied), `stale_s` (age of the vector in audio time)
   and `grid` (a silence-clock decision); `Timings.n_stale_decisions` counts decisions taken after the
   allowance ran out. Option: `lead_s` > 0 requests a point's refresh that long ahead and decides at the point
   on the slightly older vector (no delay; measured below as the alternative).
2. **The one point that cannot be anticipated is the transition itself.** In Pipecat that is the 0.2 s
   decision at the VAD stop, taken in `analyze_end_of_turn()`, which is `async`: it requests the refresh and
   `await`s it for `wait_s` without blocking the event loop (other processors keep running; audio frames
   queue for at most that long), then decides on whatever vector it has. In the synchronous `push()` path
   (the offline validation, a non-asyncio caller) the transition point is never waited for: it is deferred
   one frame like the others.
3. **Nothing on the frame path blocks.** `push()` only snapshots, submits, collects finished futures and
   runs the heads; the speech-time timer also submits instead of running inline. A queued request that has
   not started is superseded by a newer one; one that is running completes and is applied if newer than
   the cached vector.
4. **The three inputs of a decision come from one instant.** The heads were trained on a pooled vector, a
   mel summary and a τ taken from the same window. Each request therefore snapshots mel20 and tau_feat with
   the window it hands to the encoder, and a silence-clock hop feeds the snapshot that belongs to the vector
   it decides on (`hop(consistent=True)`); the free-running 100 ms hops keep using the live mel20 and τ. With
   the consistent pair the worker reproduces the inline decision bit for bit once the vector has arrived
   (`test_worker_reproduces_inline_decisions_when_the_vector_arrives`, checked through 2.5 s of silence).
5. **Rule offsets are decided on silence-clock hops only.** The hop output marks them (`grid`). Deferring a
   hop shifts the phase of the free-running 100 ms hops, so one of them can land just before a grid point
   while that point's refresh is in flight and would see the previous vector (a trace at real-time pace
   showed a hop at 1.16 s on the 0.8 s vector, 360 ms old, with the 1.2 s refresh pending). Inline, the
   regular hop and the grid hop coincided, which is why B3 never saw this. The analyzer now ignores
   free-running hops for the rule (the backstop still fires on any hop), and `b2_validate.py` records the
   grid hop for each query instead of "the first hop after τ", which had been picking up exactly those
   early hops in the worker runs (0.02–0.04 AUC at 1.2 s, see §1.2).

`worker=False` restores the B3 behaviour (same code path, inline refresh; the silence-clock hop now also
uses the consistent snapshot, which inline is the live state anyway) and is what
`scripts/b2_validate.py`'s original configurations and `scripts/b3_replay.py --no-worker` run. The silence
clock now also skips grid points already passed when the silence starts at `vad_stop_secs` (B3 ran a
redundant second refresh 20 ms after the VAD stop). The heads session always has one ORT thread; `threads`
sets the encoder session only.

### 1.1 Tests (`tests/test_streaming.py`, 10 tests, all passing with the 4 B2 tests unchanged)

- the worker's applied vector equals an inline refresh of the same window (`np.allclose`, 1e-5), with the
  deferred hops landing one frame after each grid point;
- with lead 0 and a long defer, every silence-clock p_end of the worker equals the inline detector's at the
  same point within 1e-6 (same vector, same mel20, same τ);
- a 50 ms encoder (monkeypatched) never shows on the frame path: push p99 < 5 ms, the stale decisions are
  recorded and later hops pick the vector up, the vector never exceeds the timer's age;
- at real-time pace the 0.2 / 0.5 / 0.8 s points are decided on a vector no older than lead + two frames;
- `await_refresh()` applies the result inside an asyncio loop;
- `StreamingEOTAnalyzer` with the worker: the VAD-stop decision in `analyze_end_of_turn` is fresh, every
  rule offset is decided once, `append_audio` p99 < 5 ms, `cleanup()` closes the executor.

### 1.2 Fidelity: B2's gate re-run with the worker (`scripts/b2_validate.py --configs "worker thread"`)

Same 150 AMI test silences, same offline predictions, replay at real-time pace (the worker needs wall
time; the B2 configurations ran as fast as the CPU allowed). Gate: streaming AUC within 0.01 of offline.

| configuration | AUC offline / streaming at τ = 0.2 s | at 0.5 s | at 1.2 s | mean \|Δp\| at 0.2 / 0.5 / 1.2 s | vector age at the query | core (per-call accounting) |
|---|---|---|---|---|---|---|
| B2 reference: inline silence clock, tight loop (`runs/b2/validate.md`) | 0.774 / 0.765 | 0.767 / 0.758 | 0.877 / 0.865 | 0.061 / 0.051 / 0.020 | 0 ms | 9.1 % |
| inline silence clock, run alone on 2026-10-25 (same room-tone draws as the rows below) | 0.774 / 0.765 | 0.767 / 0.760 | 0.877 / 0.871 | 0.061 / 0.051 / 0.020 | 0 ms | 8.2 % |
| **worker, defaults: request at the point, wait ≤ 80 ms (lead 0), real time** | 0.774 / **0.765** | 0.767 / **0.760** | 0.877 / **0.872** | 0.061 / 0.051 / 0.019 | 40 ms (the decision lands 40 ms after the point, on the point's vector); none stale | 11.2 % |
| worker, option: lead 60 ms, decide at the point on the 60 ms-older vector | 0.774 / 0.770 | 0.767 / 0.748 | 0.877 / 0.836 | 0.082 / 0.086 / 0.033 | 60 ms, none stale | 16.6 % |
| earlier iterations (vector and mel from different instants, or free-running hops recorded): 1.2 s AUC 0.833–0.850 | | | | | | |

Cost in the same runs (real-time pace, see §1.3 point 2 for why these are above B2's tight-loop figures): hop
p50 1.2 ms, encoder refresh p50 31 ms / p99 49 ms (1 thread), calling-thread time per frame p50 0.08 ms /
p99 4.6 ms, **decision-path burst at the silence-clock hops p50 1.0 / p99 2.1 / max 3.1 ms** (n = 2,398; was
25–40 ms inline). 543 of the 2,398 silence-clock hops were taken before their refresh arrived: those are the
transition points (τ = 0, not a rule offset, deferred one frame), none of the 0.2 / 0.5 / 1.2 s queries.

Reading: with the defaults the worker path gives the inline decisions (AUC equal within 0.001 at every
offset, mean |Δp| equal to three decimals), i.e. B2's gate is met with the same 0.006–0.009 residual the
inline silence clock has against offline (room-tone draw and hop grid, as B2 explained). The 1.2 s row has
88 items instead of 90: two pauses ended within the 40 ms the decision was postponed, so there was no decision
to record, which is also what a deployment would see. Getting here took three iterations, each a real effect
worth knowing: (i) a vector computed 60 ms before the point with the live mel of the point, or the point's
vector with the mel of 40 ms later, costs 0.035 AUC at 1.2 s (design point 4); (ii) a free-running hop that
lands just before a grid point sees the previous vector, and both the rule and the validation have to decide
on grid hops only (design point 5); (iii) the validation's room-tone draws depend on the order of the
configurations in one run, so the reference has to be re-run in the same position (the inline row above).
The lead option, measured with the corrected recording, is 0.005 better at 0.2 s and 0.012 / 0.035 worse
at 0.5 / 1.2 s than inline, with mean |Δp| 1.4–1.7× larger: a vector whose window ends 60 ms before the
decision is not the trained one, and the loss grows with τ. Waiting ≤ 80 ms for the point's own vector is
the right default; the lead stays as a knob for a setup that cannot afford the 20–60 ms.

### 1.3 Cost in Pipecat (`scripts/b3_replay.py`, far15 + eager 30 %, 20 turn ends + 20 pauses, real time)

Whole-process CPU share is `getrusage` over each replay (all threads); the pipeline-task numbers are the
wall time of every `append_audio` call (p99 and max per replay, then the median and worst replay) and of
the `analyze_end_of_turn` call at the VAD stop. Each run is one process at a time; the B2 fidelity run and
the release evaluation were running on other cores at the same time.

| configuration | CPU share of one core, mean / p95 | `append_audio` on the pipeline task: p99 (median replay / worst) | per-replay max (median / worst) | `analyze_end_of_turn` at the VAD stop (median / max) | 0.2 s decisions taken on a stale vector | outcome: strict FAR / model commits / commit latency mean |
|---|---|---|---|---|---|---|
| inline (B3 behaviour, `--no-worker`) | 0.189 / 0.201 | **45.4 / 48.9 ms** | 51.7 / 57.8 ms | 39.8 / 50.4 ms | 0 % | 5.0 % / 0.90 / 1.20 s |
| worker, wait 20 ms at the VAD stop | 0.164 / 0.196 | **4.1 / 5.6 ms** | 6.0 / 44.1 ms | 22.9 / 27.3 ms | 100 % (vector 161 ms old on average, 340 ms max) | 5.0 % / 0.90 / 1.31 s |
| worker, wait 50 ms at the VAD stop | 0.192 / 0.206 | **4.5 / 5.4 ms** | 6.0 / 11.1 ms | 45.6 / 55.1 ms (awaited, loop not blocked) | 7 % (vector ≤ 20 ms old) | 5.0 % / 0.90 / 1.32 s |
| worker, wait 50 ms, consistent snapshot, lead 60 ms | 0.170 / 0.187 | **3.7 / 5.0 ms** | 5.8 / 41.8 ms | 37.9 / 54.2 ms | 6 % | 5.0 % / 0.85 / 1.38 s |
| **final: shipped defaults (request at the point, decision waits ≤ 80 ms; VAD-stop wait 50 ms)** | 0.188 / 0.207 | **4.8 / 5.3 ms** | 6.8 / 22.9 ms | 45.0 / 54.6 ms | 6 % | 5.0 % / 0.90 / 1.22 s (inline 1.20 s) |

Decision path by rule offset (shipped defaults): the 0.5 s and 1.2 s decisions cost **0.2 ms p50 / 0.3 ms
max** on the pipeline task (the snapshot means no mel recomputation at the point) and land 40–60 ms after
the point on the point's own vector; the 0.2 s decision takes 34 ms p50 / 55 ms max in `analyze_end_of_turn`,
of which all but ~2 ms is the await on the event loop. Inline, every rule offset cost 30–39 ms p50 and up to
53 ms on the pipeline task. The outcome of the default run is the inline run's (5 % false alarms, 18 of 20
turn ends committed by the model, commit latency p50 / p90 / mean 1.20 / 1.44 / 1.22 s against 1.19 / 1.43 /
1.20 s: the 20 ms are the postponed decisions). The worker runs shared the machine with the fidelity runs of
§1.2, which is where the p99 of 4–5 ms and the occasional 20–45 ms outlier frame come from; B3's solo runs
had the machine to themselves.

Reading:

1. **The burst is gone from the frame path.** `append_audio` p99 falls from 45 ms to 4–5 ms, the worst
   frame of a replay from 58 ms to 6–7 ms (plus a 20–45 ms outlier frame in each 40-replay run, on a frame
   with no decision, while other measurement processes shared the machine). The 4 ms p99 is above B2's tight-loop hop figure (0.7 ms) for the same reason the
   encoder is slower at real-time pace (next point): the sparse frame loop runs on a core that idles between
   frames, and a hop then costs ~2 ms and the window snapshot for a request 2–4 ms.
2. **Power management doubles the cost of sparse work.** At real-time pace on an otherwise idle machine
   the encoder takes 46 ms p50 (22 ms in a tight loop, B2) and the hop 2.2 ms (0.7 ms); thread QoS hints
   (`pthread_set_qos_class_self_np`, user-interactive) change nothing, nor does moving the work to a thread.
   This is the cost a deployment sees on a laptop; a loaded server core has no idle gaps. The lead of
   60 ms is set from this number; it is a parameter (`lead_s`), and on a 3× slower cloud vCPU it should be
   ~150 ms.
3. **The one-frame wait at the VAD stop misses the encoder by design.** With one ORT thread the refresh
   takes 22 ms when warm, so a 20 ms wait lands 2 ms short every time (100 % of 0.2 s decisions stale, the
   vector 0–340 ms old: exactly the staleness B2 measured at 0.04 AUC). A 50 ms wait makes 93 % of them
   fresh and the rest 20 ms old, at the price of the VAD-stop decision arriving ~25 ms later than inline
   would give it, and that wait is an `await` on the event loop, not CPU work on the pipeline task.
   **The analyzer therefore defaults to `wait_s` = 0.05**, a deliberate departure from the "one frame" rule
   the plan wrote for the frame path; `push()` itself never waits (it postpones). Two ORT threads for the
   encoder (15 ms) would also fit a 20 ms wait at a higher total CPU cost.
4. **CPU share is unchanged within noise** (0.16–0.19 of one core for the whole process, against 0.189
   inline in the same session and 0.265 in B3's solo run with the plain stop strategy): the thread moves
   the work, it does not add any. The eager strategy's generations on pauses and the turn outcome are the
   same across the runs (5 % false alarms, 17–18 of 20 turn ends committed by the model), and the shipped
   defaults reproduce the inline latencies within the 20 ms the postponed decisions add.

### 1.4 Gate check (plan §6 B2 and `design_streaming_eot.md` §2)

| gate | target | C2 | status |
|---|---|---|---|
| per-hop cost | ≤ 1 ms | 0.7 ms tight loop, 2.2 ms at real-time pace (power management) | met in B2's terms; the real-time figure is the machine's, not the code's |
| encoder refresh | ≤ 25 ms single-threaded | 22 ms tight loop, 46 ms at real-time pace | as above |
| decision-path burst on the pipeline task | ≤ 5 ms | 0.2 ms p50 / 0.3 ms max at the 0.5 and 1.2 s decisions; `append_audio` p99 3.7–4.8 ms over all frames (was 45 ms) | **met** |
| streaming AUC within 0.01 of offline | | 0.765 / 0.760 / 0.872 vs offline 0.774 / 0.767 / 0.877, equal to the inline detector within 0.001 | **met** |
| CPU budget | ≤ 30 % of one core | 16–19 % whole process | met |

## 2. Release files (R1, commercial, audio-only)

`release/` holds the sources; `release/stage.py` copies them with the two ONNX files into
`runs/c2/hf_stage/<repo-name>/`, checks that both graphs load with the documented inputs and outputs, writes
`SHA256SUMS` and prints the upload command. Nothing has been uploaded and nothing pushed to the public fork.

| file in the staged directory | source | size |
|---|---|---|
| `smart-turn-v3.2-encoder-pooled.onnx` | `data/models/smart-turn-v3.2-gpu.pooled.onnx` (pipecat's v3.2 with the pooled vector as a second output, weights unchanged) | 31 MB |
| `eot-heads-v32.onnx` | `data/models/eot_heads_v32.onnx` (B1 run `knob_ami07_auc`) | 0.7 MB |
| `README.md` | `release/README.md`: model card with the B1 (AMI AUC and rule tables, smart-turn-data regression guard), B3 (end to end in Pipecat) and C1 (two-stage rule) numbers, the cost table, usage in Pipecat / standalone / one window, limitations | |
| `rules/theta_tables.csv`, `rules/eager_tables.csv` | `runs/b2/theta_tables.csv`, `runs/c1/eager_tables.csv` | |
| `eval/ami_eval.py` | self-contained (numpy, onnxruntime, soundfile): downloads the AMI headset files it needs, builds the windows as the training items were built (room tone after the next speaker's onset), Whisper log-mel in numpy, both models, AUC per τ, the rule tables with the shipped thresholds and a re-fit on dev | |
| `eval/ami_dev.csv`, `eval/ami_test.csv` | the labelled silences of `data/a2/ami/manifest_{dev,test}.parquet` without the transcripts (meeting, speaker, channel, t, label, gap_s, b_onset_s, a_turn_s) | 0.5 MB |
| `eval/ami_eval_reference.md` | the script's output on 2026-10-25 (below) | |
| `LICENSE` | BSD 2-Clause (this repository's) | |
| `AMI_LICENCE.txt` | the CC BY 4.0 text shipped in the AMI annotations release 1.7 | |
| `ATTRIBUTION.md` | encoder (Daily, BSD-2), AMI corpus (CC BY 4.0, citation), smart-turn-data; what the manifests contain; Seamless and Casual Conversations not used | |

### 2.1 The evaluation script reproduces B1 (`tests/test_release_eval.py` + a full run, 7 min)

Its numpy log-mel matches `WhisperFeatureExtractor` within 1e-3 on every bin (B1's features came from the
HF extractor on Modal GPUs), its windows are byte-identical to `eot.windows.cut_window` for the same random
state, and its AUC / rule / fit code gives the same numbers as `eot.rules_eval` on random inputs. On the
real data (`runs/c2/ami_eval/ami_eval.md`):

| AMI test | B1 (`runs/b1/knob_ami07_auc/report.md`) | release script |
|---|---|---|
| AUC at τ = 0.2 / 0.5 / 1.2 s, heads | 0.775 / 0.776 / 0.742 | 0.775 / 0.776 / 0.738 |
| AUC, frozen v3.2 | 0.671 / 0.638 / 0.575 | 0.671 / 0.638 / 0.577 |
| far10 (shipped θ 0.92 / 0.98 / 0.86): recall / FAR / mean latency | 0.80 / 0.094 / 1.48 s | 0.79 / 0.094 / 1.49 s |
| far15 (0.98 / 0.88 / 0.80) | 0.86 / 0.133 / 1.21 s | 0.86 / 0.133 / 1.21 s |
| stock one-shot θ 0.5, heads / frozen | 0.64 / 0.255, 0.51 / 0.304 | 0.64 / 0.254, 0.51 / 0.303 |
| re-fit on dev at 10 %: θ, recall / FAR | 0.92 / 0.98 / 0.86, 0.80 / 0.094 | 1.00 / 0.94 / 0.86, 0.79 / 0.090 |

The residual differences are the room-tone draw (a different random tiling after the next speaker's onset)
and 1e-3 feature noise; the 10 % re-fit lands on a neighbouring grid point with the same recall. Runtime
for dev + test at the three rule offsets: 16k encoder windows, 7 min single-threaded, plus 13 s per
threshold fit; the download is 34 headset files (about 3 GB) for the two splits.

### 2.2 Upload (not run)

```
.venv/bin/python release/stage.py --repo-id susuROBO/<repo-name>        # restage after any change
.venv/bin/hf auth login
.venv/bin/hf upload susuROBO/<repo-name> runs/c2/hf_stage/<repo-name> . --repo-type model --commit-message 'R1: streaming end-of-turn heads'
```

Open decisions for the user: the repository name (the staging used `smart-turn-streaming-eot`; note that
`pipecat-ai/smart-turn-v2` already exists on the Hub, so "v2" in the name would be confusing), and the
public fork push: the model card links to `github.com/susurobo/smart-turn-multimodal` for `eot/streaming.py`,
`eot/pipecat_analyzer.py` and `eot/pipecat_eager_strategy.py`, which are not public yet. The files that would
go to the fork for R1: `eot/{streaming,pipecat_analyzer,pipecat_eager_strategy,features,model,windows,
turn_rules,ami,rules_eval}.py`, `scripts/{a2_*,b1_items,b2_validate,b3_replay,c1_eager_sweep,c2_report}.py`,
`tests/`, `release/`, `docs/{a1,a2,b1,b2,b3,c1,c2}*.md`, `docs/plan_online_eot.md`, `docs/results_so_far.md`;
nothing under `data/` or `runs/` (the ONNX files go to the Hub).

## Reproduce

```
.venv/bin/python -m pytest -q tests/                                                                   # 40 tests
.venv/bin/python scripts/b2_validate.py --run runs/b1/knob_ami07_auc --n 150 --configs "worker thread" --out runs/c2/validate_worker.md   # ~60 min, real time
sh runs/c2/replays.sh                                                                                  # three replays, ~27 min
.venv/bin/python scripts/c2_report.py runs/c2/replay_worker_wait20.csv runs/c2/replay_worker_wait50.csv runs/c2/replay_inline.csv
.venv/bin/python release/eval/ami_eval.py --audio-dir data/ami/audio --no-download --encoder data/models/smart-turn-v3.2-gpu.pooled.onnx \
    --heads data/models/eot_heads_v32.onnx --theta-tables release/rules/theta_tables.csv --cache runs/c2/ami_eval/scores.npz --out runs/c2/ami_eval/ami_eval.md
.venv/bin/python release/stage.py --repo-id susuROBO/smart-turn-streaming-eot
```
