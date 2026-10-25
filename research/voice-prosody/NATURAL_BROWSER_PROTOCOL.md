# Finite natural browser execution v2 (timeline receipt repair)

This research route uses a full, finite, authenticated 48 kHz float32 recording in
an `AudioBufferSourceNode`, the actual AudioWorklet, the existing bounded PCM
transport, the existing model worker and main-thread result receiver. Production
remains DSP. The original hybrid exact-checkpoint guard is unchanged. This v1
runner admits the H/H fallback only; it refuses a candidate or external-admission
decision rather than silently substituting H for a required H/R/R/H comparison.

## Frozen workload and source origin

`prepare_natural_browser` requires a retained decision JSON with
`external_assessment_admitted: false` and `candidate_passed_development: false`.
It reads no reference contents. It binds acquisition metadata and selects the
first recording in recording-ID order for F01/F03/F05/F06/M01/M03/M05/M06, under
clean and first-difference conditions. These are already-observed PTDB speakers.
The first difference uses x[-1]=0 before the prefix. Exactly 192 float32 zero
samples precede each complete source. Original provider references remain intact.
The original target is native support end minus 960 minus 192. Common eligibility
starts at original target 3168 samples (66 ms), with the historical 64 ms lower
boundary and 32 ms trailing boundary unchanged. Prefix-overlapping processing
remains in the actual history and receives separate boundary status.

Each source gets a new browser context and model worker. Start is scheduled at a
future AudioContext frame divisible by 1920, which is also divisible by the
480-sample hop and 128-sample render quantum. The worklet arms before the source
connects, ignores earlier render samples, crops the exact final quantum and
returns observed origin, first 16 samples, source endpoint, sequence and partial
sample counts. A synthetic index/impulse fixture must establish this mapping
before any speech execution. Missing initial input never rebinds source zero.
Restarted processed block starts must remain on the exact 480-sample grid. An
off-grid result fails scoring; it is never rounded to a reference.

## Retained v1 failure and v2 repair

The first complete fixed workload remains a failed execution: second-H
M05_si1211 first-difference capture coordinates advanced 128 samples beyond its
render-clock coordinates after one discontinuity. Its start was exact and drains
completed, but post-reset phase was invalid. Callback-level observations were not
retained, so a repeated/overlapping callback is inferred, not proven. The original
source snapshot, raw receipts, trace and failed audit remain under attempt 1.
The first audit also rejected two browser tail outputs beyond the last provider
reference row; that integration error is retained separately.

V2 derives natural-mode capture coordinates directly from scheduled source and
observed render frames. Forward gaps discard partial support and charge skipped
samples through the next original 960-sample block boundary, preserving the
original target grid. A backward/overlapping quantum or an unexpected channel
shape immediately rejects further PCM with an explicit failed receipt; the code
never assumes the ambiguous samples repeat or repairs them by clamping the end.
A 64-event bounded ledger records observed/expected callback coordinates, gaps,
discarded partials and actual grid-restart skips; overflow fails explicitly.
Ordinary microphone transport and default DSP behavior are unchanged.

Every source target remains in the delivery ledger. Outputs without an exact
provider row are counted explicitly as missing reference and excluded only from
reference scoring. All eligible authenticated provider rows remain in the quality
denominator. No prediction or reference time is fitted, rounded or shifted.
Attempt 2 uses unchanged model/input bytes and the same one-pass workload; it
cannot erase attempt 1 or establish that the underlying browser event is resolved.

## Completion and bounds

`onended` marks only the source endpoint. A fence follows PCM on the same
MessagePort. The worker waits for its bounded asynchronous pump to become idle
and all result acknowledgements to arrive, with a frozen 5000 ms timeout. The
main waits at most 6000 ms for all three independently observed receipts: source
origin/end, worker inference/result completion, and the worklet having observed
the final PCM recycle with zero outstanding credits. A worker completion never
stands in for a worklet receipt from another port. Late origin/recycle receipts
remain pending within this bound. The main continues leases, acknowledgements,
telemetry and clock probes throughout draining. Only then does the normal acknowledged
teardown run. The final partial PCM block is discarded and counted, never zero
padded. Timeout, canceled inference and withheld output remain failures/losses.

The existing four PCM credits, four result credits, three queued research blocks
and two-second PCM ring are preserved. Research loss/reset events use a separate
fixed 20,000-entry worker ledger for the bounded 120-second maximum source, with
explicit overflow failure. No per-drop IPC is emitted while the main thread is
blocked; the ledger is returned once with the final fence. Raw telemetry retains
its 200 queued records / 100-record batch limits. These are application buffers,
not a bound on total browser heap. The renderer JS heap is sampled after each
source and is not a peak measurement.

## Commands

Use the existing environment and create-new ignored run directories. The original
prepared `public/.local/voice-mlp/hybrid-config.json` is used without modification.
From `research/voice-prosody`:

```sh
.venv/bin/python -m phenometrix_prosody.prepare_natural_browser \
  --data data/ptdb-tug --decision runs/STUDY/browser-decision.json \
  --output runs/STUDY/natural-inputs
```

From `apps/capture-web`, first verify discovery, then run with one measurement
process and no concurrent fitting or auditing:

```sh
pnpm exec playwright test --config playwright.research.config.ts --list
VOICE_NATURAL_MANIFEST=../../research/voice-prosody/runs/STUDY/natural-inputs/manifest.json \
VOICE_RESEARCH_MODEL_CONFIG=public/.local/voice-mlp/hybrid-config.json \
VOICE_RESEARCH_OUTPUT_DIR=../../research/voice-prosody/runs/STUDY/natural-browser \
pnpm exec playwright test --config playwright.research.config.ts --workers=1 voice-natural-research.spec.ts
```

The runner serves each PCM buffer through a local Playwright response, so corpus
media is not copied to public assets or uploaded. It runs the index fixture,
H/H playlists, then a separate 250 ms worker stall at 800 ms on the first source.
All cases retain raw telemetry, exceptions and execution receipts. It records
source snapshots before running and rejects source changes during measurement.
Warmup is the index fixture; each source still has a fresh worker/model load.
Timing is descriptive local evidence with `timingBenchmark: false`.

After browser execution has ended, from `research/voice-prosody`:

```sh
.venv/bin/python -m phenometrix_prosody.audit_natural_browser \
  --run runs/STUDY/natural-browser --inputs runs/STUDY/natural-inputs/manifest.json \
  --data data/ptdb-tug --config ../../apps/capture-web/public/.local/voice-mlp/hybrid-config.json \
  --export runs/delayed-mlp-export --output runs/STUDY/natural-browser-audit
```

The auditor binds the exact ordered index/H/H/stall case inventory and every
source/model/config/PCM/reference/result receipt before opening content; verifies origin,
phase, sample endpoints, completed/canceled/withheld inventory and terminal
credits; reconstructs features from actual support windows and history resets;
reruns the MLP through independent NumPy equations and raw YIN through librosa;
then opens authenticated reference contents. Feature equations are shared pure
TypeScript, independently driven outside browser transport, not independently
rederived DSP equations. Every intended target receives a delivery and reference
status. Missing and withheld output remains unavailable in full-denominator
quality; conditional delivered accuracy is secondary. The auditor saves parity,
quality, runtime goal and producer execution status separately. A passed audit
never rewrites an earlier failed execution.

Required software checks are capture-web unit/typecheck, repository check,
production build/public-asset exclusion, focused Python corruption tests and the
full research suite. Hosted checks use synthetic fixtures only. No corpus,
per-frame predictions, private absolute paths or browser traces belong in a
public commit. Natural execution and acoustic audits must actually finish before
claiming a browser result.
