# Voice measurement foundation

## Active ambient pipeline

The browser requests one mono microphone stream at 48 kHz with echo
cancellation, noise suppression, and automatic gain control disabled. These are
requests; the browser's reported settings determine the acquisition metadata.
The capture adapter may fall back to the AudioContext rate when the track does
not report a rate. Neither settings nor that fallback establish physical
hardware identity, input latency, or acquisition validity.

An AudioWorklet sends nominal 20 ms PCM blocks directly to a worker over a
MessagePort. The worker uses a two-second ring, 40 ms trailing analysis windows,
and a 10 ms hop. DSP uses the rendered AudioContext rate. Acquisition metadata
and rendered rate are recorded separately; resampling does not turn the
requested acquisition settings into verified settings.

PCM remains inside the worklet/worker boundary. The worker emits compact
`VoiceSignalFrameV1` values, timing records, diagnostics, and lifecycle
acknowledgements. Signal frames contain activity, periodicity, pitch estimates,
nucleus events, quality facts, segment identity, and processor provenance.

The ambient extractor produces exactly seven metrics:

- median F0 and F0 variability;
- speech-activity fraction;
- pause rate and median pause duration;
- median speech-run duration; and
- acoustic nucleus rate estimate.

Each metric requires its configured duration, coverage, segment/event count,
sample rate, SNR, clipping, DC offset, continuity, and estimator quality. A
failed requirement returns a specific abstention rather than zero or an
imputed value. There are no sustained-vowel, reading, rapid-syllable, or other
guided tasks in the active browser workflow.

## Version and clock contracts

| Contract | Version and meaning |
| --- | --- |
| Worker and worklet messages | `phenometric.voice-worker-message.v2` and `phenometric.voice-worklet-message.v2`; matched internal assets must ship together. |
| Compact acoustic frame | `phenometric.voice-signal-frame.v1`; legacy time semantics are preserved. |
| Numerical DSP | `voice-analysis-1.1`, processor `browser-voice-dsp@1.1`; transport and timing changes do not establish new numerical or clinical equivalence. |
| Timing | `phenometric.voice-stream-timing.v1`; sample support and explicit clock-coordinate mappings. |

`VoiceSignalFrameV1.tMs` is still the analysis-window start in the capture's
sample timeline, and `absoluteSampleIndex` is its first sample. New timing
records declare half-open support `[supportStartSample, supportEndSampleExclusive)`,
the end boundary used as the reference, and the final required sample at
`supportEndSampleExclusive - 1`. No future sample is included. The existing
`acquiredAtMs` field is an estimate of required support-end availability in the
main thread's performance-clock coordinate, not microphone hardware acquisition
time.

Worklet block endpoints use `currentFrame` plus the consumed samples within the
render quantum. A paired AudioContext/main-performance anchor maps that audio
timeline into main-thread milliseconds. Worker timestamps are converted using
both realms' `performance.timeOrigin`; direct subtraction of unrelated
`performance.now()` values is invalid. The anchor's clock-read bracket is
reported, while AudioContext clock uncertainty remains explicitly unknown.
The mapping is invalidated when the AudioContext stops running. Startup and
reset while suspended keep a null mapping until a fresh running-context anchor
arrives.

Frame timing separates support availability, worker receipt, analysis start,
worker-ready time, and main-thread arrival. Worker receipt minus block-end
availability combines transport and scheduling; it is not an independently
measured queue delay. Per-frame service time covers feature/DSP/state/quality
work through worker-ready. A separate complete-block service time covers ring
insertion, all analysis, diagnostics, buffer recycling, and the data envelope's
structured-clone/post cost. A subsequent completion envelope carries that time;
only construction and posting of this timing completion are excluded. Frame counts per
block remain explicit. The research harness compares frame service against a
5 ms budget at a 10 ms hop and complete-block service against a 10 ms budget at
a nominal 20 ms cadence; those budgets are distinct from cadence deadlines.

Inconsistent or negative mapped durations are retained as raw values and mark
timing unavailable. Published derived durations become null, never clamped to
zero or replaced by service time. Otherwise valid acoustic predictions remain
available. Worker latency p95 is a rolling statistic over at most 500 valid
frames; `latencyStatisticSampleCount` distinguishes an empty statistic from a
measured zero. These clocks do not replace the separate authenticated UTC clock
qualification used by the encounter integration.

## Buffer bounds, gaps, and resets

The worklet preallocates four transferable PCM buffers and one pending block.
It drops newly completed blocks when all four transport credits are in flight;
it does not allocate extra PCM buffers. Sequence and sample counters advance
through those drops. Credits survive capture-epoch changes, and late,
duplicate, or unknown returns cannot increase the four-buffer capacity. The
worker returns buffers to their originating port even when their epoch or
mapping is stale.

A missing input quantum is a gap, not observed silence. Once the first input
establishes a capture origin, the worklet immediately advances the sample clock
through absent render quanta and discards partial support. Terminal snapshots
include missing quanta even when observed input never resumes. The sample loss counter includes missing samples and discarded
partial samples. There is no zero-filled substitute. A terminal partial block
is also counted when the worklet stops.

The provisional worker-admission policy drops a block whose mapped block-end
age exceeds 100 ms before acoustic analysis. It records `stale-dropped`, reason
`audio-timeline-age-exceeded`, and separate stale block/sample counters. This
bounds admitted block age in the mapped audio timeline when the mapping is
consistent. It does not bound hardware delay, later result delivery, or the
age of every sample within a window. Unknown or inconsistent mappings cannot
establish actual age. The policy is recorded as
`drop-new-at-cap;drop-stale-at-worker;reset-on-gap`; it is independent of model
speed or accuracy claims.

Sample gaps, stale drops, sequence/sample regressions, large timestamp gaps,
sample-rate changes, mapping changes, and capture resets clear the PCM ring,
prior spectral state, nucleus score/refractory state, and next analysis index.
Recovery emits a new segment identity, including a reset that repeats the same
capture epoch. Sample-rate changes rebuild rate-dependent ring capacity. Track
replacement starts a new pipeline. Production remains DSP-only. The separate
research MLP hook resets feature history and explicit model state at the same
boundaries; cancelled generations cannot deliver a late prediction or failure
into a renewed epoch or mapping.

Worker-to-main results have a separate four-credit limit. One credited batch
uses two ordered envelopes: a data envelope containing at most four frames and
one diagnostic, then one block-timing completion. At most four pairs, or 24
scalar records, can remain unacknowledged. Production's nominal 20 ms blocks
normally produce at most two frames. PCM support over 40 ms is rejected and
counted, never truncated; this also bounds the larger unit-test input path.
The worker holds at most one five-record data array while processing a block,
and the main receiver holds at most four incomplete envelopes.

When no result credit is available, the worker recycles incoming PCM, counts
result-backpressure blocks/samples/episodes, retains a constant-size span for
the latest episode, and resets feature history. It produces no queued scalar
message for those drops. The span's bounds may include other sample gaps; its
counts describe only blocks discarded for result backpressure. A subsequent
sample jump and segment change preserve the continuity loss. A recovery batch
or final diagnostics expose those counters. Token IDs survive epoch changes;
only an ACK with the token's originating epoch releases its credit, and repeats
cannot create extra capacity. Tokens also retain their mapping and segment: a
discard ACK for retired support cannot reset a fresh segment. The receiver
retains a monotonic completed-ID high-water mark to suppress duplicate delivery
without an unbounded history of completed IDs.

On receipt of the completion envelope, the host rejects a frame whose valid
mapped support-end age exceeds 100 ms. It also rejects further frames from the
same segment until the worker has reset after the acknowledged loss. The ACK
contains discarded/stale frame counts and is sent before application callbacks,
so a throwing callback cannot strand credits. Retired-epoch results are
acknowledged without delivery. A retired clock-mapping identity is also rejected,
including when a reset repeats the same epoch. Unknown or inconsistent clock mappings retain
unavailable timing and cannot support an age guarantee. The age check refers
to receipt, not a later callback, paint, display, or hardware-input endpoint.

PCM/result transport queues, local batch assembly, and worker histories now
have explicit capacity bounds. This is not a measured bound on the entire
browser heap, downstream application storage, or allocation frequency. Ready,
error, and final disposal replies are control messages; terminal disposal does
not wait for a result ACK from a blocked main thread.

## Stop and terminal provenance

`stop()` is idempotent and waits at most 500 ms for worker and worklet
acknowledgements before releasing processor resources. Its optional
`onTransportStopped` callback reports a current-epoch terminal worklet snapshot,
or explicit unavailable status with `worklet-stop-timeout` or
`worklet-dispose-failed`. Missing acknowledgement never becomes zero loss.
Cleanup runs in a `finally` block, including when a stop-notification callback
throws. The pipeline does not own or stop the host's input tracks or AudioContext.

A complete transport snapshot includes final sequence and sample-clock end,
dropped blocks/samples, the partial samples discarded at disposal, and credits
still in flight. “Complete” describes that snapshot; it does not prove all
in-flight blocks reached analysis. Worker diagnostics remain a separate final
response. Late-epoch disposal or terminal messages cannot terminate or complete
a newer epoch. The worklet returns the same snapshot on repeated dispose and
rejects subsequent processing and buffer returns.

## Functional evidence and sustained-run preparation

The historical, local, ignored
`research/voice-prosody/runs/browser-check-v4-result-identity/` evidence from
2026-10-22 exercised an actual Chrome AudioBufferSource → production AudioWorklet
→ production worker graph with a synthetic 200 Hz tone. A 3.5013-second source
run with a forced 300 ms worker stall delivered 312 periodic frames within 2 Hz
of the known pitch, counted four stale block drops and eleven worklet block drops,
and recovered in a new segment. The final worklet snapshot also counted 640
discarded partial samples. All 475 frame/block timing records reached the
bounded batch sink.

A separate 2.2011-second source run blocked the main thread for 350 ms. It reached
four outstanding result batches, counted 13 backpressure input blocks (12,480
samples), discarded eight stale frames at the host, and resumed a new segment.
It delivered 178 periodic frames matching the known pitch. Both worker and
worklet terminal acknowledgements completed. All four browser tests passed,
including sink failure and 24-step WASM export parity against Python fixtures.
The exported graph spike was untrained.

`run-manifest.json` binds this evidence to source and artifact hashes, verified
stable during the run. Focused unit regressions additionally cover differing
worker/main time origins, suspended startup/reset, missing quanta, stale ports,
wrong/duplicate/retired-epoch ACKs, bounded result assembly, oversize rejection,
and credit-independent disposal. Concurrent compute was not controlled. This
is synthetic functional evidence, not sustained latency, live microphone,
natural-speech accuracy, trained-model improvement, or clinical validation.

The completed 30-minute paced run with one midpoint overload is documented in
[the streaming report](../research/voice-prosody/STREAMING_RESULTS.md), including
the unresolved rise in mapped arrival times. To reproduce it, wait for other
training, benchmark, and validation processes to finish. Record the machine,
browser/runtime versions, power conditions,
thread count, source hashes before/after, and test report alongside raw timing
batches. Use a fresh directory; the test refuses to overwrite evidence:

```sh
cd apps/capture-web  # from the repository root
export VOICE_RESEARCH_OUTPUT_DIR="$(mktemp -d "$PWD/../../research/voice-prosody/runs/browser-soak-30m-XXXXXX")"
export VOICE_RESEARCH_METHOD=dsp
export VOICE_RESEARCH_DURATION_MS=1800000
export VOICE_RESEARCH_OVERLOAD_AT_MS=900000
export VOICE_RESEARCH_OVERLOAD_DURATION_MS=300
pnpm exec playwright test --config playwright.research.config.ts \
  --grep 'real AudioWorklet/worker replay' --workers=1 --reporter=json \
  > "$VOICE_RESEARCH_OUTPUT_DIR/playwright-report.json"
```

The configured named target is headless local Chrome at 48 kHz, one Playwright
worker, a 30-minute source interval, and a test timeout 45 seconds longer.
The overload targets the worker; the separate short blocked-main test exercises
result backpressure. Retained frame/block arrays are capped at 120/80; numeric distributions are
capped at 200,000 values per metric. Raw records use batches of at most 100 with
one in-flight batch and at most 200 queued records; sink failure/backpressure
fails the run. Require complete worker/worklet stop acknowledgements, matched
raw counts, explicit drops, known-tone pitch, and segment recovery. The result
remains labelled a sustained functional soak rather than automatically becoming
a controlled latency benchmark. Repeated synthetic tone is not new speech data.

## Research-only trained MLP execution

`research-voice-trained-worker.ts` configures the production worker's optional
research hook before initialization. The normal capture worker imports no ONNX
runtime and remains DSP-only. The research hook is explicitly
`unqualified-functional-fixture`: the frozen trained MLP failed separate
synthetic accuracy gates, so successful execution or parity cannot promote it.
Predictions are siblings of the original signal frame, never replacements for
its F0, heuristic confidence, quality, processor identity, or legacy `tMs`.

The pinned configuration binds export manifest, graph, runtime module, and WASM
hashes. Only verified local bytes execute, via temporary blob URLs, using ONNX
Runtime Web 1.23.2 WASM with one thread and no proxy. Initialization time is
separate from paced source time. Shared v2 extraction supplies 57 raw float32
features and six candidate masks; fitted scaling already lives in the graph.
The 32-element state is explicit and reset, although this framewise MLP produces
zero next-state. The model targets support end minus 20 ms, retains the actual
last required input sample, and exposes uncalibrated voicing/candidate
probabilities separately from correctness probability, which remains null.

One async operation holds one PCM block and at most three queued requests.
The existing four worklet buffers and four result credits remain authoritative;
prepared features and predictions are bounded to the current block's maximum
four frames. Inference is serial. Epoch/mapping resets invalidate the active
generation, recycle active/queued buffers once, clear feature/model state, and
suppress both late success and late rejection. A current model failure emits
one error and stops research admission until a new worker is created. Disposal
acknowledges independently of inference completion and result credits; host
termination is the final bounded release barrier.

The original frame service timestamp ends after feature/baseline preparation.
Research inference-only service, frame-start-to-model-ready elapsed time,
support-end-to-model-ready duration, and delayed-target-to-model-ready duration
are reported separately. Full-block service includes model completion, and
main-result arrival includes the entire model path. Clock inconsistencies
retain raw values and null derived timing. These measurements do not establish
browser heap growth or hardware-input latency.

`prepare-voice-mlp-research.mjs` stages the frozen export under the ignored
`public/.local/voice-mlp` directory without replacing the random export spike.
The short real-browser check compares actual stream inputs/outputs against
independent NumPy equations. A separate worker regression compares every legacy
DSP frame and all shared features with native-grid `CausalVoiceReplay` using
`referenceOriginMs: 0`, whose first full support ends at 40 ms. This differs
from the provider-label replay phase used in training and is not relabeled as
natural-speech evaluation. Fresh source-bound functional artifacts are under
`research/voice-prosody/runs/browser-check-v6-trained-stream-final/`.

For a controlled comparison after all other compute stops, use the same overload
command above with `VOICE_RESEARCH_METHOD=dsp` or `trained-mlp`, duration 60000,
and overload at 30000. Run sequential DSP, MLP, MLP, DSP in four fresh evidence
directories. This changes only the research entry configuration. Source and
artifact digests must remain unchanged through all runs.

## Forbidden outputs

PCM, waveforms, spectral arrays, pitch cycles, cepstra, MFCCs, formant tracks,
spectrograms, transcripts, embeddings, voiceprints, and microphone identifiers
must not enter ObservationV3, reports, or the event journal.

## Optional WavLM research service

`services/voice-inference` retains the earlier loopback FastAPI adapter for
isolated research. It is disabled unless `PHENOMETRIC_WAVLM_ENABLED=1`, uses a
pinned local-only model revision and weight digest, and returns only layer
mean/standard-deviation summaries plus processor provenance.

The current browser has no client for this endpoint. WavLM summaries do not
enter active measurements, evidence, reports, or trajectory.
