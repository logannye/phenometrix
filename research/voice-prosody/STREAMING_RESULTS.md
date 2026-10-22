# Browser voice streaming evidence

This report preserves the original fixed-anchor runs. The later
[independent timing follow-up](TIMING_FOLLOWUP.md) replaces the operational
freshness gate with same-main-clock leases while retaining these historical
measurements and their unresolved physical interpretation.

The default voice estimator remains DSP. The learned MLP is an explicitly
unqualified research fixture; runtime success cannot reverse the failed quality
gates in [RESULTS.md](RESULTS.md). This report concerns generated-audio replay
through a real Chrome AudioWorklet, MessagePort, worker, and result receiver.
It does not measure physical microphone latency or clinical validity.

## Implemented transport and time contract

The worklet preallocates four 20 ms transferable PCM buffers. At 48 kHz each
contains 960 float32 samples. Credit exhaustion drops input with sample and block
counters instead of allocating more transport buffers. Missing input is a gap,
not observed silence; terminal diagnostics include missing quanta and partial
support. Credits remain globally bounded across epoch resets.

The worker permits four unacknowledged result batches. A batch contains at most
four frames and a diagnostic record, followed by a separate completion record;
there are at most 24 queued scalar records under this contract, plus one bounded
host assembly. Without result credit, the worker recycles and drops input,
records the backpressure episode, and resets analysis state. An acknowledgment
from a retired mapping or segment releases its credit without clearing fresh
state. Completed duplicate batches are suppressed with a bounded high-water mark.

Both input admission and host output delivery apply a provisional 100 ms age cap
when the mapped clock is consistent. Worker times use the main performance clock
coordinate. Sample support, target time, availability, worker preparation,
complete-block service, and main arrival remain distinct. Negative or inconsistent
derived durations remain unavailable with raw values retained. Unknown audio-clock
precision is null; this is not a hard physical latency guarantee.

Discontinuities reset the PCM ring, spectral and nucleus history, feature history,
and optional model state, and change the analysis segment. The DSP keeps its
`browser-voice-dsp@1.1` identity and legacy window-start `tMs`. Optional v2 transport
and timing provenance does not retroactively change acoustic measurement semantics.
The research MLP has a separate target at support end minus 20 ms and never
replaces the product's signal frame or enters its observation contract.

The research worker verifies local artifact hashes, accepts raw float32 features
because scaling is embedded in the trained graph, and uses ONNX Runtime Web
1.23.2 WASM with one thread and proxy disabled. One inference operation is active
at a time, with at most three queued PCM blocks backed by the same worklet credits.
Generation checks suppress obsolete completion and rejection after reset. Disposal
acknowledges and clears transport independently of waiting for an uncancelable
runtime promise; host worker termination ends remaining work.

## Functional and arithmetic checks

On the frozen implementation, 51 focused tests passed alongside five actual-Chrome
research tests. Regressions cover missing terminal input, retired acknowledgments,
duplicate delivery, stale inference rejection, serial bounded inference, reset,
and disposal. The shared feature path exactly matches native-grid replay, and
the complete legacy DSP frame remains equal between research and default paths.

The trained actual-stream test produced 215 frames with zero decision mismatches
against independent NumPy inference and maximum absolute numerical error
4.76837158203125e-7. All-masked candidates, raw voicing probability, and the absence
of a calibrated correctness probability retain separate meanings. A random-weight
recurrent export spike also exercises its declared graph contract; it is not
trained-model performance evidence.

Source identities and final measurements are recorded in
[streaming-summary.json](../../benchmarks/voice-prosody/streaming-summary.json).
Raw timings and complete local run manifests remain in ignored research outputs.
The standard-library `phenometrix_prosody.audit_browser` independently reconstructs
summary distributions and counters from the raw records and verifies their hash.
All five fresh audits passed without a metric mismatch; the recorded source hashes
and all run-artifact hashes also matched. No measured source changed during this audit.


## Sequential runtime series

On 22 October 2026, the frozen implementation ran locally on an Apple M4 Max
(16 CPU cores, 40 GPU cores, 128 GiB memory) in headless Chrome 155. The sequence
was DSP, MLP, MLP, DSP for 60 seconds each, followed by a 30-minute DSP run.
Each run introduced a deliberate 300 ms worker stall at its midpoint. Agent
training and tests did not overlap this series; ordinary desktop background load,
power and thermal conditions were not controlled. ONNX used one WASM thread.
Every original artifact retains `timingBenchmark: false`. These are descriptive
same-machine measurements with overload recovery, not an exclusive-hardware
latency benchmark.

The source is a **looped two-second** mono buffer: 96,000 float32 samples,
384,000 PCM payload bytes at 48 kHz, containing 200 Hz and 400 Hz sinusoids of
amplitude 0.12 and 0.03. Output to the speaker is zero. The orchestration manifest
incorrectly described a full-duration source buffer. Its original bytes are
preserved; an additive `metadata-correction.json`, bound to that manifest and
the fixture source, records the correction. The public summary includes it.
Browser allocation overhead and heap growth were not measured.

All five runs completed, passed their functional assertions, retained complete
raw telemetry, and acknowledged both worker and worklet stop. All delivered
DSP periodic frames reported 200 Hz. This repeated tone is a transport fixture,
not additional speech accuracy evidence. The MLP runs still deliver the unchanged
DSP frame, with separate unqualified model predictions.

| Run | Delivered frames | Complete blocks | Block service p50 / p95 / p99 / max, ms | Mapped arrival p50 / p95 / p99 / max, ms |
|---|---:|---:|---|---|
| 01-dsp | 5,964 | 2,989 | 1.50 / 3.30 / 4.30 / 6.80 | 33.67 / 42.57 / 44.07 / 46.87 |
| 02-mlp | 5,962 | 2,988 | 2.60 / 5.00 / 5.80 / 9.20 | 21.13 / 29.53 / 31.63 / 34.13 |
| 03-mlp | 5,962 | 2,988 | 2.80 / 5.10 / 6.10 / 9.70 | 24.43 / 32.83 / 35.03 / 38.13 |
| 04-dsp | 5,966 | 2,990 | 2.20 / 3.70 / 4.40 / 7.20 | 30.37 / 39.17 / 40.57 / 42.97 |
| 05-dsp-soak | 179,958 | 89,986 | 2.10 / 3.60 / 4.30 / 8.30 | 66.37 / 86.67 / 91.17 / 95.67 |

Complete-block elapsed service includes feature preparation, optional ONNX
completion, PCM recycle and the data-envelope post. It excludes waiting before
the block handler starts and the completion-envelope post. Legacy frame service
ends before ONNX inference, so it is not total model-path cost. The two MLP runs
had inference-only p50 0.10 ms, p99 0.40/0.50 ms and maximum 3.60/3.80 ms.
Their full-block p99 values were 5.80/6.10 ms, versus 4.30/4.40 ms for the short
DSP runs. MLP initialization took 220.10/215.50 ms and occurred before paced
source time; it is included in the separate setup durations.

Across the five runs, no complete block exceeded the declared 10 ms service
budget or 20 ms cadence, and no prepared frame exceeded its 10 ms cadence.
The narrower 5 ms frame-preparation budget was exceeded 1, 4, 2, 0 and 9 times
respectively. These checks do not cover the deliberate external 300 ms stall
as block service; its effect appears in drops and recovery.

Run 03 retained 30 unavailable arrival timings because main arrival was slightly
earlier than the worker-ready value after time-origin conversion. Arrival
quantiles therefore use 5,932 of its 5,962 frames. Raw values remain available;
there was no clamping into a valid duration. Its receipt distribution retains
all 5,962 separately valid receipt measurements. The lower MLP arrival numbers
cannot be read as faster whole-path execution: its complete processing cost is
higher and each run has its own fixed clock anchor.

| Run | Worklet dropped blocks | Final worklet dropped samples | Included partial samples | Worker stale blocks / samples | Analysis segments |
|---|---:|---:|---:|---|---:|
| 01-dsp | 10 | 10,048 | 448 | 4 / 3,840 | 2 |
| 02-mlp | 11 | 11,136 | 576 | 4 / 3,840 | 2 |
| 03-mlp | 11 | 12,288 | 320 | 4 / 3,840 | 2 |
| 04-dsp | 10 | 9,600 | 0 | 4 / 3,840 | 2 |
| 05-dsp-soak | 11 | 10,880 | 320 | 4 / 3,840 | 2 |

Every run reached but did not exceed four PCM credits and four result credits.
No result-backpressure input blocks, stale host results, or research queue drops
occurred in this worker-stall series. The separate blocked-main test exercises
those failure paths. Worklet sample losses include partial support and any
missing render input, so sample loss need not equal full dropped blocks times
960. Worker stale drops are a separate disposition, not included in the worklet
sample totals. Terminal snapshots do not assert delivery of every in-flight
result.

## Thirty-minute trend and unresolved clock behavior

The soak delivered 179,958 frames and 89,986 complete-block records. Its block
service p99 was 4.30 ms and maximum 8.30 ms. Reported arrival nevertheless rose:
whole-run p99 was 91.17 ms and maximum 95.67 ms. The table below independently
recomputes five-minute source windows from raw records. Frame windows use the
exclusive support endpoint; block windows use support start. Quantiles use
nearest rank without interpolation. Receipt is frame-weighted, matching the
browser summary; a block's receipt can contribute to two frames.

| Source minutes | Arrival p50 / p99, ms | Receipt p50 / p99, ms | Complete block p50 / p99 / max, ms | Receipt-to-main p50 / p99, ms |
|---|---|---|---|---|
| 0–5 | 38.97 / 49.67 | 33.37 / 37.77 | 1.90 / 3.80 / 5.90 | 2.00 / 3.90 |
| 5–10 | 49.37 / 60.67 | 44.57 / 47.97 | 2.00 / 4.20 / 7.50 | 2.10 / 4.20 |
| 10–15 | 59.27 / 74.27 | 52.87 / 62.97 | 2.20 / 4.30 / 8.30 | 2.30 / 4.40 |
| 15–20 | 71.67 / 82.57 | 65.17 / 69.77 | 2.20 / 4.20 / 6.20 | 2.20 / 4.20 |
| 20–25 | 77.27 / 88.17 | 70.67 / 75.07 | 2.30 / 4.50 / 6.30 | 2.40 / 4.60 |
| 25–30 | 81.97 / 93.07 | 74.37 / 80.37 | 2.20 / 4.60 / 8.00 | 2.30 / 4.70 |

Arrival median increased from 38.97 to 81.97 ms, and receipt median from 33.37
to 74.37 ms. Receipt-to-main median remained 2.0–2.4 ms; legacy-ready-to-main
median was 0.40 ms in every window. Thus most of the measured increase precedes
worker receipt relative to the mapped audio boundary, rather than appearing in
the measured block handler or downstream delivery. This does **not** identify
its physical cause or prove that a queue grew.

The harness takes one audio/performance anchor before starting the source and
assumes equal clock rates throughout. It does not periodically re-anchor, and
audio-clock precision remains unknown. Relative clock-rate/phase error and
upstream rendering, message transport or callback scheduling can all affect the
reported pre-receipt interval. Internal nonnegative checks do not establish
clock accuracy. The final sample counter represents 1,799.946667 seconds while
paced wall duration is 1,800.000800 seconds, a 54.13 ms difference. Start/stop
boundaries and phase also differ, so this is supporting diagnostic context,
not proof of drift and not a fitted correction.

The soak had no unavailable frame timings or host-stale discards, but its last
window approached the provisional 100 ms cap. No stable whole-path arrival bound
or extension beyond 30 minutes is established. Browser heap growth, hardware
input latency, display latency, live microphone performance, and sustained MLP
operation for 30 minutes remain unmeasured.

## Separate blocked-main and model-parity evidence

The source-identical v6 functional suite passed five browser tests and 51 focused
unit tests. Blocking the main thread for 350 ms during a 2.2015-second replay
reached four result credits, dropped 14 input blocks (13,440 samples) due to
result backpressure, discarded eight stale frames at the receiver, and recovered
in a second segment. It delivered 176 periodic frames matching the known tone;
16 arrival timings remained unavailable. Both terminal acknowledgements
completed. The sink-failure case explicitly stopped incomplete collection.

The original trained MLP's actual worklet stream matched independent NumPy
inference over 215 frames: zero decision mismatches and maximum absolute error
4.76837158203125e-7. The separate untrained recurrent export spike matched 24
steps with three all-masked cases, zero decision mismatches and maximum error
1.4901161193847656e-7. These are arithmetic and integration results, not evidence
that either model estimates pitch correctly in general. The runtime MLP is the
original fitted model, not the augmented follow-up.

The public JSON retains every run's distributions, counters, exclusions,
terminal snapshots, model/runtime hashes, source hashes, original raw-artifact
hashes, independent audit identities, and five-minute trend. Raw row streams
remain local and ignored. Scientific failure gates, no sealed test, no seed
confirmation, no calibrated F0 correctness, and unestablished voice-duration
and coverage aggregate contracts remain unchanged; see [RESULTS.md](RESULTS.md).
