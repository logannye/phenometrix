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
| Worker and worklet messages | `phenometric.voice-worker-message.v3` and `phenometric.voice-worklet-message.v2`; matched internal assets must ship together. |
| Compact acoustic frame | `phenometric.voice-signal-frame.v1`; legacy time semantics are preserved. |
| Numerical DSP | `voice-analysis-1.1`, processor `browser-voice-dsp@1.1`; transport and timing changes do not establish new numerical or clinical equivalence. |
| Timing | `phenometric.voice-stream-timing.v2`; sample support, unqualified fixed clock projections, and separate freshness certificates. |
| Render freshness | `phenometric.voice-render-freshness.v1`; required software render leases with a 100 ms issue-to-receipt limit. |

`VoiceSignalFrameV1.tMs` is still the analysis-window start in the capture's
sample timeline, and `absoluteSampleIndex` is its first sample. New timing
records declare half-open support `[supportStartSample, supportEndSampleExclusive)`,
the end boundary used as the reference, and the final required sample at
`supportEndSampleExclusive - 1`. No future sample is included. The existing
`acquiredAtMs` field is an estimate of required support-end availability in the
main thread's performance-clock coordinate, not microphone hardware acquisition
time.

Worklet block endpoints use `currentFrame` plus the consumed samples within the
render quantum. A paired AudioContext/main-performance anchor projects that audio
timeline into main-thread milliseconds at a fixed unit rate. This projection is
retained as an unqualified diagnostic, not a qualified availability timestamp or
an admission gate. The render clock need not remain synchronized to the
performance clock; an initial phase offset or later change is not queue delay. Worker timestamps are converted using
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
available only when their independent render-freshness lease passes. A timing
record's `status: valid` means the projected durations are internally consistent;
`projectionSemantics: unqualified-fixed-audio-clock-projection` still applies,
and `renderFreshness` is the separate operational certificate. Worker latency p95 is a rolling statistic over at most 500 valid
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

The required `required-render-lease-v1` policy uses a main-thread performance
clock for freshness. The main thread issues a lease with an epoch, mapping
identity, time origin, unique ID, issue time, and expiry exactly 100 ms later.
The worklet snapshots the current lease before the first sample of a new PCM
block. Renewal cannot refresh an earlier partial block or either of its normal
two frame endpoints. The worker checks the lease before analysis after converting
its performance coordinate into the issuing main clock. Missing, malformed,
retired, future, expired, or wrong-origin evidence is explicitly withheld, with
`freshness-withheld` block disposition, a reason, and separate block/sample
counters. Result backpressure takes precedence when result credit is exhausted;
its separate counters retain those discarded samples.

The host verifies the exact lease against a ledger of at most eight issued
identities and requires main result receipt minus main issue time to be between
0 and 100 ms. This gives a conservative software-render age bound for the last
required input; earlier samples in the 40 ms support can predate lease issue.
It does not establish microphone acquisition time, timestamp precision, or later
application callback/display time. Certificate precision is explicitly null.
Fixed audio-clock projections cannot rescue an expired lease or expire a fresh
one. Unit fixtures that preserve historical numerical behavior explicitly select
`unqualified-legacy-fixture`; they keep the previous mapped-age policy and never
receive a freshness certificate. Production has no implicit fallback.

Lease renewal targets 25 ms and allows one outstanding main-to-worklet request.
Expiry does not mint another credit. A matching accepted or retired ACK frees
only that request, including an accepted ACK already posted before reset.
Reset invalidates the issuance ledger and worklet lease; a late old ACK can
release control credit without revalidating old support. The worklet ACK path
is independent of PCM and result credits. Suspension prevents renewal, and
stop cancels its timer. A resumed lane needs a newly accepted lease.

Sample gaps, freshness withholding, sequence/sample regressions, large timestamp gaps,
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

On receipt of the completion envelope, the host rejects missing or expired
lease proof before any application callback and records the withholding reason.
It also rejects further frames from the same segment until the worker resets
after the acknowledged loss. The ACK contains discarded/expired frame counts
and is sent before application callbacks, so a throwing callback cannot strand
credits. Retired-epoch results are acknowledged without delivery. A retired
clock-mapping identity is also rejected, including when reset repeats an epoch.
The certificate uses the original main clock; unknown or inconsistent fixed
audio-clock projections remain separately unavailable and do not suppress an
otherwise certified acoustic frame. The certified endpoint is main receipt,
not a later callback, paint, display, or hardware-input endpoint.

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

## Functional evidence and sustained-run confirmation

The current lease path passed 70 focused tests and a local Chrome functional
pair on 2026-10-22. The ignored
`research/voice-prosody/runs/browser-check-v7-render-lease/` contains a 6.5-second
synthetic run with a 300 ms worker stall and a separate 350 ms main-thread stall:
all 571 delivered frames were certified, eight host frames and five worker
blocks were withheld for expired leases, and known 200 Hz pitch recovered.
A separate 2.2-second blocked-main run delivered 176 certified frames and
withheld eight expired host frames. Both terminal acknowledgements completed.
These short tests establish functional recovery for the exercised path, not a
sustained performance or physical-acquisition bound. See
[the timing follow-up](../research/voice-prosody/TIMING_FOLLOWUP.md) for the
independent clock investigation and subsequent confirmation status.

The historical worker-v2, local, ignored
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

The historical worker-v2 30-minute paced run with one midpoint overload is
documented in [the streaming report](../research/voice-prosody/STREAMING_RESULTS.md),
including its unresolved rise in mapped arrival times. It predates the lease
contract and does not validate the new gate. The command below prepares a new
worker-v3 confirmation with separate worker and main stalls. Wait for other
training, benchmark, and validation processes to finish. Record the machine,
browser/runtime versions, power conditions,
thread count, source hashes before/after, and test report alongside raw timing
batches. Use a fresh directory; the test refuses to overwrite evidence:

```sh
cd apps/capture-web  # from the repository root
export VOICE_RESEARCH_OUTPUT_DIR="$(mktemp -d "$PWD/../../research/voice-prosody/runs/browser-soak-30m-XXXXXX")"
export VOICE_RESEARCH_METHOD=dsp
export VOICE_RESEARCH_DURATION_MS=1800000
export VOICE_RESEARCH_OVERLOAD_AT_MS=600000
export VOICE_RESEARCH_OVERLOAD_DURATION_MS=300
export VOICE_RESEARCH_MAIN_STALL_AT_MS=1200000
export VOICE_RESEARCH_MAIN_STALL_DURATION_MS=350
export VOICE_RESEARCH_CLOCK_DIAGNOSTICS=1
pnpm exec playwright test --config playwright.research.config.ts \
  --grep 'real AudioWorklet/worker replay' --workers=1 --reporter=json \
  > "$VOICE_RESEARCH_OUTPUT_DIR/playwright-report.json"
```

The configured named target is headless local Chrome at 48 kHz, one Playwright
worker, a 30-minute source interval, and a test timeout 45 seconds longer.
The two distinct overloads target the worker and main thread. Independent 1 Hz
clock probes preserve main-read brackets, render-callback causal brackets, and
separate output-device timestamps without shifting the fixed anchor. Only one
probe may be outstanding; the harness retains the last 64 and streams all raw
observations. Retained frame/block arrays are capped at 120/80; numeric distributions are
capped at 200,000 values per metric. Raw records use batches of at most 100 with
one in-flight batch and at most 200 queued records; sink failure/backpressure
fails the run. Require complete worker/worklet stop acknowledgements, matched
raw counts, explicit drops, known-tone pitch, segment recovery, every delivered
lease certificate within 100 ms, and every five-minute window. The prospectively
declared performance checks require complete processed-block p99 at most 10 ms
and no processed block above its 20 ms cadence. The result
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

## Research-only hybrid execution

The explicitly selected `phenometrix.mlp-voicing-yin-pitch.v1` mode retains the
original trained MLP for voicing and supplies pitch from
`phenometrix.raw-yin-librosa011-semantics.float64.ts.v2`. Configuration bytes, original
checkpoint, exported graph, export manifest and local runtime assets are pinned.
The default/legacy candidate-MLP research mode stays distinct; neither mode
changes production DSP signal frames or their identity.

`research-voice-yin.ts` ports librosa 0.11.0 raw YIN semantics for exactly 1,920
float32 samples at 48 kHz: linear autocorrelation, cumulative mean normalized
difference, first threshold trough or global minimum, and bounded parabolic
refinement. A fixed reusable scratch set bounds its storage. Each prepared frame
owns scalar YIN evidence; subsequent preparation or reset cannot overwrite it.
The hybrid returns unvoiced below raw MLP probability 0.5, otherwise finite raw
YIN pitch in inclusive 50–700 Hz, or unavailable. The candidate head is diagnostic
only. No additional RMS, correlation or continuity gate is introduced.

The frame target remains support end minus 960 samples, with earliest input
availability at support end. This is 20 ms of future support relative to the
reported target. YIN service, model inference and complete admitted-block service
remain separate metrics. Both delivered and host-withheld research predictions
retain raw YIN, configuration/model/estimator identity and null correctness
probability for independent auditing.

The bounded parity worker uses the same hook on pinned local PCM windows. It
checks natural recordings on the exact offline provider phase, including feature
history and reset. It is arithmetic evidence, not natural audio passing through
the AudioWorklet and not a reference-accuracy test. The real worklet path instead
uses known synthetic sources, explicit stalls and a sustained local replay.
See the [frozen validation plan](../research/voice-prosody/HYBRID_VALIDATION_PLAN.md)
and [validation report](../research/voice-prosody/HYBRID_VALIDATION_RESULTS.md)
for independent natural results, numerical tolerances and runtime evidence.

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
