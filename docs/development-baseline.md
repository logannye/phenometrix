# Development baseline — 2026-09-06

This is the starting point for the next PhenoMetrix development cycle. It
records implemented software separately from remaining product and validation
work. The canonical checkout is `/Users/logannye/Projects/phenometrix`; the
repository is [logannye/phenometrix](https://github.com/logannye/phenometrix).

## Integrated scope

The browser now connects all three product capabilities in one local research
demonstration:

1. **Ambient Capture:** independently consented camera/microphone setup,
   calibration, local workers, bounded capture, 27 terminal metric outcomes,
   strict ObservationV3 provenance, and a ten-section report after media disposal.
2. **Personal Trajectory:** explicit acceptance of the first observation and
   comparison with a second live observation for the same page-local subject.
   Exactly six facial metrics terminate as measured, withheld, or incompatible.
3. **Clinician Evidence Card:** a deterministic comparison card with exact
   source traces and page-local accept/dismiss. This is a research review action.

The integration includes the previously uncommitted condition profile,
comparison/card contracts, trajectory package, UI wiring, measurement quality
and provenance fixes, independent lane failure/cleanup handling, dependency
lockfile updates, and regression coverage. Voice remains part of each generic
capture but does not influence the condition comparison.

The optional WavLM service is isolated, disabled by default, and unused by the
browser. The active path has no LLM calls, transcript, server, durable storage,
or export. Media, derived frames, and results follow the lifecycle in the
[architecture](architecture.md) and [safety boundary](safety.md).

## Versioned measurement identity

| Component | Baseline identity |
| --- | --- |
| Measurement pack | `ambient-local-observation` 3.4.0 |
| Face algorithms | 1.1.0 |
| Voice algorithms | 1.2.0 |
| Condition profile | `unilateral-facial-movement-research-demo` 1.0.0 |
| Observation contract | `phenometric.encounter-observation.v3` |

Protocol/profile content digests and static-asset hashes are checked by
`pnpm run check`. The canonical digests live in source, not in this document.
These identities describe measurement behavior and compatibility; none denotes
clinical validation.

## Repository reconciliation

The pre-integration main revision was `22caba7` (PR #29). GitHub had no open
pull requests or issues at review. The three remaining feature branch tips
were already ancestors of main:

| Historical branch | Last tip | Merged work |
| --- | --- | --- |
| `codex/visual-foundation-hardening` | `c5c7329` | PR #14 |
| `codex/voice-foundation-hardening` | `87273c9` | PR #15 |
| `codex/demo-ui-simplification` | `c3d6d62` | PRs #16–20 |

Those commits are preserved in main's history. The new two-capture work was
checkpointed as `113a14c` before baseline cleanup. Historical plans, guided/v2
protocols, and examples remain explicitly archival; they are not active inputs.

## Reproduce the automated checks

Use Node 22.12+ on the Node 22 line (or Node 24+), pnpm 9.12.3, Chrome, and uv
with Python 3.11. From the canonical checkout:

```bash
pnpm install --frozen-lockfile
uv sync --project services/voice-inference --extra dev --locked
pnpm verify
```

`pnpm verify` runs the structure and digest checks, workspace unit tests,
TypeScript checks, production build, browser lifecycle suite, and isolated
Python service tests. `pnpm test:voice` runs only the latter. Browser fixtures
substitute media/worker APIs; Python tests use a deterministic fake adapter.
These checks do not exercise real camera/microphone measurement or WavLM weights.

Verification on 2026-09-06 passed: 420 workspace unit tests, 12 browser lifecycle
tests, 4 Python service tests, TypeScript, production build, and structure/
digest/static-asset gates. Both pagehide cases were added after reproducing
retained reference/report state in the previous handler. Frozen pnpm and uv
installs passed, and `pnpm audit --audit-level=high` reported no known
vulnerabilities. The local run used Node 24.3.0, pnpm 9.12.3, uv 0.11.2,
Python 3.11.15, and macOS 26.2. This is an automated test environment record,
not real-device acceptance. GitHub CI verifies the integrated branch separately.

## Remaining work and acceptance boundaries

| Work | Current boundary / completion evidence needed |
| --- | --- |
| Real-device acceptance | Pending: named Mac/Chrome version, repeat sessions, anatomical laterality, adverse capture conditions, accessibility, and five-minute memory/performance checks in the [implementation plan](superpowers/plans/2026-08-21-unilateral-facial-palsy-prototype.md#8-manual-acceptance-and-gono-go). |
| Measurement validity | No reference-standard accuracy, repeatability, measurement error, minimum detectable change, or clinical validation. Raw differences carry no health interpretation. |
| Condition algorithm design | The generic pack's expression segmentation and resting baseline differ from the historical clinical design. Resolve the [recorded divergences](superpowers/specs/2026-07-24-facial-palsy-protocol-pack-design.md#12-current-generic-pack-divergences) with a versioned decision and validation data. |
| Asset execution integrity | Hash checks establish delivery self-consistency; the manifest has no independent trust anchor and processors reload asset URLs. Trusted identity and binding of checked bytes to executed bytes remain deployment work. |
| Camera-only consent/capture | The current generic contract includes camera and microphone. A face-only path needs its own explicit contract and consent design. |
| Durable longitudinal product | Participant identity, retained derived observations, multi-visit history, robust baselines/trends, migrations, authorization, and retention governance are unbuilt. |
| Clinical workflow/integration | Authenticated review, durable audit, narrative, export, EHR/FHIR and telehealth-platform integration are unbuilt. |

The remaining items require deliberate scope or validation decisions. The next
development cycle can choose among them without treating old branch names or
historical design prose as outstanding implementation work.
