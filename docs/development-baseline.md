# Development baseline — 2026-09-06

The canonical checkout is `/Users/logannye/Projects/phenometrix`; the repository
is [logannye/phenometrix](https://github.com/logannye/phenometrix).
This document distinguishes implemented software from deployment and validation
work. It does not imply that the current working branch has been merged.
See the [integration runbook](encounter-integration.md) for exact setup.

## Implemented starting point

The three product capabilities now have two implementations:

| Capability | Retained local demo | New encounter integration |
| --- | --- | --- |
| Ambient Capture | Explicit device setup, two independently consented captures, 27-outcome ObservationV3 reports | Authorized branch on existing patient tracks; modality-specific consent; rotating bounded derived observations |
| Personal Trajectory | One accepted page-memory reference and six compatible facial comparisons | Immutable multi-visit snapshots, versioned treatment timelines, deterministic baseline/delta and phase coverage |
| Clinician Evidence Card | Page-local report/card and unauthenticated accept/dismiss | Embeddable source-linked evidence panel, scoped service reads, exact-run durable review API |

The new modular TypeScript service and worker use PostgreSQL, append-only source
and evidence history, idempotency, revision checks, transactional jobs, and worker
leases. The service client reuses the host's scoped authentication. FHIR
normalization and configured-source sync preserve source status, knowledge time,
clinical date precision, corrections, and dose units. These are implemented
integration boundaries; no clinical EHR or Zoom installation is provisioned.

The first HFS research protocol selects four existing engineering metrics:
left/right eye aperture and left/right lid closure completeness. It does not
measure spasm burden or establish clinical treatment response. Compatible
pretreatment observations can establish an encounter-weighted baseline. Missing
timing, incompatible sources, sparse phases, concurrent treatments, and prior
cycles remain explicit. Fitted models and causal/clinical claims are disabled.

The optional clip-governance package is separate and disabled by default.
Storage, identity, audit, and retention adapters are required before use.
The optional WavLM service remains isolated and unused by browser capture.

## Versioned identity

| Component | Identity |
| --- | --- |
| Generic measurement pack | `ambient-local-observation` 3.4.0 |
| Face / voice algorithms | 1.1.0 / 1.2.0 |
| Legacy condition profile | `unilateral-facial-movement-research-demo` 1.0.0 |
| Legacy observation | `phenometric.encounter-observation.v3` (unchanged) |
| New durable observation | `phenometric.durable-observation.v1` |
| HFS protocol | `hfs-ambient-treatment-alignment-research` 1.0.0 |
| Descriptive engine | `descriptive-treatment-alignment.1.0.0` |

Canonical digests live in source. Protocol, specification, snapshot, run, and
processor identities control reproducibility and compatibility; none denotes
clinical validation.

## Reproduce verification

Use Node 22.12+ on the Node 22 line (or Node 24+), pnpm 9.12.3, Chrome, and
uv/Python 3.11. From the canonical checkout:

```bash
pnpm install --frozen-lockfile
uv sync --project services/voice-inference --extra dev --locked
pnpm verify
pnpm test:postgres
git diff --check
```

`pnpm verify` includes structure/digest checks, workspace unit tests, TypeScript,
production builds, both browser suites, and isolated Python service tests.
PostgreSQL integration tests require an explicitly configured test database and
run separately; see the [service README](../apps/encounter-service/README.md).
Browser tests use simulated media/worker outputs, and Python tests use a fake
adapter. They do not establish real-device measurements or WavLM performance.

The encounter implementation passed **574 workspace tests, 17 Chrome browser
tests, 4 Python tests, typechecking, production/embedded builds, and
structure/digest checks** on 2026-09-06. Six additional integration tests passed
against actual local PostgreSQL 16.14. The browser suite includes a real
dedicated-worker calculation on synthetic derived voice frames and cancellation
without device access. This validates software behavior, not clinical accuracy.

The earlier local-demo baseline had 420 workspace tests and 12 browser tests.

## Remaining acceptance boundaries

| Work | Required completion evidence |
| --- | --- |
| Host/Zoom integration | Installed and authorized platform integration, exact patient-track binding, actual transport/decode support, permission/lifecycle checks; RTMS metrics remain excluded by the HFS protocol |
| Existing identity and clinical workflow | Trusted signed claims, patient enrollment and matching, consent lifecycle, host review integration, and institutional research authorization |
| EHR connection | Provisioned credentials, source/terminology configuration, date/timezone reconciliation, and operational sync |
| Real-device acceptance | Named Mac/Chrome and devices, source replacement and mute/unmute, background behavior, long-call resource bounds, calibrated timing, and anatomical laterality |
| Measurement validity | Reference-standard agreement, repeatability, minimum detectable change, clinical association, subgroup performance, and prospective workflow utility |
| Treatment-response validity | HFS reference outcomes, measurement sensitivity to symptoms and adverse effects, carryover/concurrent-treatment interpretation, and validation before any fitted curve |
| Research clips | Production encrypted storage, exact observation authorization, separately consented retention/annotation, purge/backup policy, auditing, and governance approval |
| Deployment integrity | Trusted manifests/executed assets, secrets and database operations, authorization configuration, retention, monitoring, and backup/recovery |

Legacy facial-palsy design divergences remain documented in the
[historical design](superpowers/specs/2026-07-24-facial-palsy-protocol-pack-design.md#12-current-generic-pack-divergences).
They do not grant clinical interpretation to the new HFS protocol.

## Earlier repository reconciliation

This implementation branch started at `8506ad5` (PR #30), after the earlier
baseline reconciliation. Before that reconciliation, main was `22caba7` (PR #29). At that earlier review,
GitHub had no open pull requests or issues; the historical visual-foundation,
voice-foundation, and demo-UI feature tips were already ancestors of main.
The two-capture work was checkpointed as `113a14c` before baseline cleanup.
These are historical reconciliation facts, not a current remote-status check.
Guided/v2 examples and plans remain archival.
