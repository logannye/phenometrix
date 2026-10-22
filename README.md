# PhenoMetrix

PhenoMetrix is a nonclinical research prototype for quality-aware face and voice
measurement during telehealth conversation. Its three capabilities are **Ambient
Capture**, **Personal Trajectory**, and **Clinician Evidence Card**.

It is not a medical device. This repository establishes software behavior and
bounded research results, not diagnostic accuracy, disease severity, treatment
response, or clinical readiness. Local demonstrations use synthetic participants.

## Voice streaming and estimation research

The browser voice path now records sample support, clock mapping, output age,
processing time, discontinuities, and dropped input. Fixed credits bound both
AudioWorklet-to-worker PCM transport and worker-to-main result transport. Gaps,
resets, and stale work start a new analysis segment. The default acoustic
estimator remains the versioned DSP baseline.

A locally paced 30-minute Chrome replay accepted 179,958 frames. Complete
processed-block service time was 4.3 ms at p99 and 8.3 ms maximum, with no
20 ms block-cadence misses and explicit recovery from an injected stall.
Mapped audio-to-main arrival medians rose from about 39 to 82 ms across the run;
clock mapping versus upstream scheduling remains unresolved. These results
establish bounded software transport, not physical microphone latency. See the
[streaming report](research/voice-prosody/STREAMING_RESULTS.md).

The accompanying [research report](research/voice-prosody/RESULTS.md) evaluates
compact trained pitch/voicing selectors against PTDB-TUG laryngograph-derived
references and analytic signal fixtures. All computation ran locally on an
Apple M4 Max with 128 GiB unified memory; neural fitting used PyTorch MPS.

| Development method | Joint voicing/F0 error ↓ | Correct within 50 cents among reference-voiced frames ↑ |
| --- | ---: | ---: |
| Existing DSP, evaluated with 20 ms measurement delay | 10.61% | 62.45% |
| Fixed YIN comparator, 20 ms delay | 11.02% | 64.88% |
| Fixed candidate-continuity rule, 20 ms delay | 11.60% | 69.39% |
| Trained GRU, 20 ms delay | 7.95% | 76.38% |
| Trained framewise MLP, 20 ms delay | 8.63% | 75.41% |
| MLP with fixed synthetic augmentation, 20 ms delay | 8.26% | 77.03% |
| Pretrained centered CREPE comparator, 32 ms future support | 9.56% | 66.81% |

These are equal-speaker/equal-condition averages over **four development
speakers, 80 recordings, and four declared conditions**. Training used 240
recordings from 12 different speakers. Development informed the targeted
augmentation; these are not sealed-test results. All conditions and speaker
values appear in the report. The fixed 20 ms support delay is distinct from
processing latency and accounts for part of the improvement over the original
zero-lookahead evaluation.

**No learned estimator qualified for production.** The GRU failed its required
one-percentage-point advantage over the original MLP. Both original models
failed a steady 300 Hz fixture despite having suitable candidates. Augmentation
improved a separately frozen 32-case synthetic challenge from 72.95% to 88.07%
voiced success, but missed the 90% aggregate and 80% per-case gates and retained
the steady-300-Hz failure. Fixed YIN achieved 99.82% on that challenge, while its
natural-speech development error was higher than DSP. This evidence does not
justify silently switching either estimator into the product.

Trained exports and an explicitly unqualified ONNX browser research path support
reproduction. Raw voicing probabilities and candidate scores are **not calibrated
probabilities of a correct F0 measurement**. Calibration, multi-seed confirmation,
and sealed-test evaluation stopped at the failed acceptance gate. Sealed-test
acoustic data remain untouched. Research scripts, public aggregate evidence,
failed decisions, and reproduction commands are in
[research/voice-prosody](research/voice-prosody/README.md). Raw audio, feature
caches, weights, and large run outputs are excluded from Git.

## Implemented encounter path

```text
clinic identity + encounter binding + modality-specific consent
  → authorized branch on the host's patient media tracks
  → local face/voice workers and transient measurement primitives
  → derived observations throughout the encounter
  → scoped API + append-only PostgreSQL history + leased analysis worker
  → immutable history + protocol + treatment-alignment specification
  → descriptive measurements, qualified comparisons, and clinician review
```

`createEmbeddedEncounter()` uses a host-owned stream, video element, and running
audio context. It does not request devices, open another login, or add a capture
screen. The host retains track ownership. `startIntegratedEncounter()` connects
capture to the scoped service client; hosts supply authenticated encounter
integration. No provisioned Zoom installation or connected clinical EHR is
included.

The service implements signed tenant/study/participant authorization, explicit
grants and bindings, idempotent ingestion, append-only corrections, transactional
analysis jobs, worker leases, stale-publication protection, and source-linked
reviews. FHIR R4 normalization preserves administration versus order status,
dose units, revisions, and date precision. EHR credentials, terminology mappings,
patient matching, and scheduling remain external.

A bounded UTC estimate requires a fresh attestation from a deployment clock
monitor for qualified live accuracy. Missing evidence stays unqualified.
Passive voice calibration and the Zoom RTMS reference adapter are engineering
interfaces awaiting acquisition validation; transport tests do not qualify
platform-derived measurements.

## Initial treatment-alignment research protocol

The HFS protocol targets research with adults who already have a clinician
diagnosis of hemifacial spasm and receive botulinum injections. It compares
left/right eye aperture and left/right lid closure completeness. These four
engineering measurements are not spasm counts, severity, or proof of response.

A deterministic core uses a sealed protocol and immutable history to select a
compatible pretreatment baseline, aggregate by encounter, preserve exclusions,
and flag concurrent treatments and possible carryover. Subsequent documented
injections stop index-cycle comparisons. Unknown treatment timing disables
treatment alignment and baseline deltas.

Sparse observations remain sparse. There are no fitted curves, interpolated
observations, causal effects, peak-response or wearing-off estimates, dose
recommendations, or automatic clinical actions. Measurement error and minimum
detectable change are unknown. See the [platform vision](docs/telehealth-platform-vision.md)
and [validation boundaries](docs/validation.md).

## Local demonstration and privacy

`pnpm dev` runs the original facial-movement research demonstration. Consent and
independent camera/microphone permission precede each live capture. After media
disposal, the user can accept a page-memory reference and compare a second
independently consented capture. Reset, visibility loss, or reload discards the
page-local state. This demo has no durable history or authenticated review.

Both demo and encounter paths reuse the `ambient-core` registry of 7 voice and
20 facial metrics. Failed source-quality contracts produce withheld measurements
with reasons. The six-row condition comparison selects facial measurements;
voice remains part of the generic report. Raw audio, images, native landmarks,
transcripts, voiceprints, and embeddings are not observation payloads.

Routine capture retains derived observations only through the governed service.
Optional research-media retention is separately consented and disabled by
default; no production media store is bundled. Evidence access checks current
consent, and corrections create new versions. Acute stroke screening remains
excluded; see [safety.md](docs/safety.md).

## Run and verify

Use Node 22.12+ on the Node 22 line or Node 24+, pnpm 9.12.3, and Chrome on macOS.

```bash
pnpm install --frozen-lockfile
pnpm dev
```

The local demo runs at `http://127.0.0.1:4173`. `pnpm dev:evidence` opens the
synthetic evidence preview at `http://127.0.0.1:4175`. Follow the
[integration runbook](docs/encounter-integration.md) and
[service README](apps/encounter-service/README.md) to configure PostgreSQL and
run `pnpm dev:encounter`. Use synthetic data for development.

```bash
pnpm verify
pnpm test:postgres  # requires TEST_DATABASE_URL for a synthetic test database
PYTHONPATH=research/voice-prosody research/voice-prosody/.venv/bin/python -m pytest research/voice-prosody/tests
git diff --check
```

`pnpm verify` runs structure/digest checks, unit tests, TypeScript, builds,
browser suites, and the isolated voice-sidecar tests. It requires Chrome and
uv/Python 3.11; the research environment has a separate pinned lockfile. The
optional [WavLM service](services/voice-inference/README.md) remains disabled and
disconnected from capture. The [research README](research/voice-prosody/README.md)
separates synthetic smoke, public-corpus experiments, and actual-browser replay.
Browser fixtures and synthetic replay do not establish real-microphone accuracy.

The [local validation record](benchmarks/voice-prosody/local-validation.json)
binds source hashes to 707 unit tests, 18 product browser tests, 77 research Python
tests, four isolated sidecar tests, six PostgreSQL integration tests, and the
focused research browser checks. Both release builds exclude local research
assets and preserve all ten normal public assets byte for byte.

## Repository map

| Path | Responsibility |
| --- | --- |
| `apps/capture-web` | Local demo, embedded capture, DSP, worklet/worker transport |
| `apps/encounter-service` | Scoped API, PostgreSQL, workers, FHIR boundaries |
| `apps/clinician-review` | Embeddable evidence panel and synthetic preview |
| `packages/ambient-core` | Deterministic face and voice extraction |
| `packages/contracts` | Observation and durable encounter contracts |
| `packages/condition-profiles` | Facial demo and HFS research protocol |
| `packages/encounter-capture`, `packages/encounter-client` | Host capture and service integration |
| `packages/trajectory-core`, `packages/evidence-core` | History analysis and traceable evidence |
| `packages/research-governance`, `packages/event-log` | Optional retention controls and session journal |
| `services/voice-inference` | Optional disconnected WavLM service |
| `research/voice-prosody` | Reproducible pitch/voicing experiments and reports |
| `benchmarks/voice-prosody` | Compact public aggregate evidence and source identities |

See [architecture](docs/architecture.md) for component boundaries and
[development baseline](docs/development-baseline.md) for the earlier platform baseline.
