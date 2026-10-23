# Frozen hybrid browser and independent natural validation

Declared 22 October 2026 after the user's request, against base commit
`a0de18b520eb02f8eee787f36a826a5c577539dc`, before any reserved-speaker
microphone or reference acquisition. This is a new validation stage. It preserves
all historical failures and the original hybrid's development tradeoff.

## Candidate and decision boundary

Validate only the already selected original-MLP-voicing/raw-YIN hybrid,
`phenometrix.mlp-voicing-yin-pitch.v1`. Original weights SHA-256 is
`06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d`.
The original trained ONNX export, train-only scaling and v2 57-feature ordering
remain fixed. No fitting, threshold search, candidate revision, model selection,
secondary-hybrid switch or calibration follows these outcomes.

Raw MLP p(voiced)<0.5 means unvoiced. Otherwise raw causal librosa 0.11.0 YIN
supplies pitch if finite and within inclusive 50–700 Hz, or the result is
unavailable. Candidate logits remain diagnostics and never select hybrid pitch.
YIN uses raw float32 PCM, 48 kHz, 1,920-sample windows, 480-sample hops,
center=False and trough threshold 0.1. No extra RMS/correlation/continuity gate.
Support ending at e predicts e−960 samples, available at e: 20 ms measurement
delay. Scorer references, masks and future audio cannot influence inference.
Raw voicing probability remains uncalibrated; pitch-correctness probability is
null. Production remains DSP throughout this validation stage.

Passing establishes only the named component on the named data/device. A failed
component remains failed. Implementation bugs may be repaired and every affected
check rerun with explicit new source identity; model or acceptance-rule changes
must not reuse this test as fresh validation. No test-driven method search.

## Independent natural speech

Use the existing pinned PTDB-TUG metadata inventory and seed 20261022 assignment.
Reserved pairs 5 and 6 contain F05, M05, F06 and M06. Before content access, select
exactly 40 eligible prompts per speaker using the existing ascending
SHA256(`20261022:integer_pair:prompt`) order, excluding sa1/sa2. This yields 160
source recordings and 640 recording/condition combinations. Matching F/M speakers
share prompt groups. No sample expansion or exclusion depends on outcomes.
The pre-access audit must confirm no acquired reserved acoustic/reference files,
no cross-split prompt IDs or normalized texts, and exact inventory/selection hashes.

Apply the unchanged clean, 10 dB white noise, first-difference and amplitude-0.1
clipping conditions. The canonical acquisition split is `sealed_test`; the existing
corruption transform's explicit sealed spelling is `sealed-test`. Record that
mapping, fixed seeds and achieved SNR, without changing prior data. Corruptions
stay with their original source and are not additional independent recordings.

Use a dedicated runner that validates actual protocol/configuration/model/source/
inventory hashes, rather than merely checking nonempty digest strings. Freeze the
exact selected identities and executable sources before acquisition. Acquire MIC
only; generate and hash ALL predictions for DSP, gated YIN, fixed continuity,
original candidate MLP and original hybrid before downloading or opening any REF.
Then acquire those exact reference files and score once. Preserve and hash the
complete prediction freeze. Laryngograph content is unnecessary and remains
unacquired. Historical development/training runners retain their sealed guards.

Retain nominal provider targets 16+10*k ms, the common mask t>=64 ms and
 t+32 ms<=N, and the existing reference validity rules. Out-of-range valid F0
stays in unconditional denominators. Missing/unavailable outputs count as joint
errors, including on unvoiced frames. No fitted alignment, reference repair,
transition removal, outcome-based exclusion or per-recording time adjustment.
Provider laryngograph-derived targets are an independent sensor reference, not
infallible manually verified physiological truth; filtering history remains uncertain.

Pool counts within each speaker/condition, then weight all 16 cells equally.
The primary natural component requires hybrid joint-error reduction of at least
1 absolute percentage point against EACH of DSP, gated YIN and continuity;
reference-voiced success may be at most 2 points worse than the best of those
three; clean joint error may exceed the best classical comparator by at most
1 point. These are the same declared natural gates as the previous hybrid stage.
Report original-MLP differences without using that comparison to select a model.
Publish every recording count, speaker, condition, pair effect, coverage,
false-positive/negative rate, pitch/octave error and unavailable count.

Four speakers in two matched prompt groups cannot support a broad population
claim. Report both pair effects and their range. As supplemental conditional
uncertainty, use 2,000 paired prompt-cluster bootstrap draws, seed 20261022,
resampling the 40 prompts within each fixed pair while keeping both sexes and
all conditions together. Percentile 2.5/97.5 bounds concern prompt sampling
conditional on these four speakers, not participant-population uncertainty.
Do not bootstrap individual frames or treat 640 conditions as independent people.
The component decision uses the predeclared point estimates, not a posthoc CI rule.

## Browser implementation and parity

Add the exact hybrid to the separate research worker path. Keep the default DSP
and product observation semantics unchanged. Bind the model, manifest, local
WASM/module assets and explicit hybrid/YIN identities. Reuse single-threaded
ONNX Runtime Web 1.23.2 WASM; do not fetch executable assets from a remote CDN.
The research hook retains bounded input/result storage, same-clock freshness,
segment reset, serial inference, cancellation, disposal and fail-closed loading.
Raw YIN belongs to its exact prepared frame and cannot be overwritten by another.

Before new parity outcomes, raw YIN tolerance is fixed at
max(0.01 Hz, 1e-4*abs(Python Hz)); float32 librosa and double-precision browser
FFT arithmetic need not be bit-identical. Existing trained-MLP logit/probability
tolerances remain unchanged. Require consistent voiced/unvoiced/unavailable
decisions outside p=0.5 plus/minus 1e-5 and the 50/700 Hz endpoints plus/minus
that pitch tolerance. List every boundary exception individually; it is not
unconditional parity. Independently
recompute Python references for actual browser PCM/windows. Test the original
300 Hz regression, other analytic signals, irregular chunks, reset/cancellation,
and development natural PCM. Later reserved-speech checks use the same unchanged
implementation and tolerance; no tuning after reserved reference access.
An exact finalized tolerance/configuration artifact is required before those runs.
Parity is an implementation check, not a new natural-accuracy estimate.

## Local complete-stream performance

All computation runs on this Apple M4 Max/128 GiB machine. Reuse installed tools,
one research Python environment and local Chrome. No hosted jobs, remote model
inference, paid compute or microphone capture. Check free disk before acquisition;
retain bounded useful evidence and keep corpus, raw telemetry and weights ignored.

After browser correctness passes and measured sources freeze, run sequentially
DSP, hybrid, hybrid, DSP for 60 seconds each in the same real AudioWorklet/worker
harness and repeated synthetic waveform. Then run the hybrid for 30 minutes.
Short comparisons include a 300 ms worker stall at 20 s and a 350 ms main stall
at 40 s. The soak includes those stalls at 600 s and 1,200 s. Ordinary desktop
load and thermal/power state are uncontrolled and must be disclosed. Only one
agent performance workload runs at a time; pause concurrent tests/training.

Require complete processed-block p99<10 ms (two 10 ms hops), every delivered
frame with certified software render-age<=100 ms, transport maxima<=4 PCM and
4 result batches, one pending lease request and ledger<=8, explicit drops and
segment recovery, completed stop acknowledgments and independently reconciled raw
records. Report maxima and every 20 ms cadence miss even when the p99 gate passes.
Report inference time separately from full processing, cold loading, runtime/
model identity, duration, coverage/drops, finite-state maxima and retained evidence
sizes. Do not equate bounded queues with measured whole-browser heap growth.
The freshness coordinate concerns worklet input observation; physical acquisition
latency, upstream buffering and timer precision remain unestablished.

## Publication

Run relevant final local regression/release checks, including `npm run check`
before commit. Update PR #33, the README and linked reports with exact revision,
source/artifact hashes, passed and failed components and limits. Preserve the
existing workflow and use `[skip ci]` for the local-only request. No corpus audio,
reference trajectories, raw telemetry, private weights or secrets are published.
No production promotion or clinical qualification is implied by passing these
research components.
