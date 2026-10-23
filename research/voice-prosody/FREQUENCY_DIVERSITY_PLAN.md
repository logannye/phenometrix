# Prospective TRAIN-only frequency-diversity coverage study

Base revision: `63e4ed7390ddee5095420e203d0a2c24a49535c1`.
The user authorized one bounded local TRAIN-only coverage experiment after the
prior audited no-go. This is a new protocol, not a revision or continuation of
the expired overnight scientific run. Freeze this plan, exact metadata, producer,
auditor, tests, unchanged dependency closure and supervisor identity before any
scientific generation. No source changes after freeze; retain a failed or partial
attempt and do not silently retry, enlarge or adapt the bank.

## Question and intervention

Can a fixed broad refinement of the previous carrier grid supply enough
added-only correct-candidate TRAIN support under the existing thresholds?
The previous 512-case bank covered four qualifying carriers against a minimum
of six. That result and its preparation failure remain intact.

Refine the same deterministic frequency lattice, without choosing frequencies
from model scores or new outcomes. Define 256 lattice points in integer
microhertz:

`frequency_hz(i) = (53_113_000 + 2_531_250 * i) / 1_000_000`, `i = 0..255`.

Points where `i % 4 == 0` exactly reproduce the 64 previous carrier values.
They remain a separately bound, already-observed baseline. **Generate only the
192 other carrier values**, spanning 55.64425 through 698.58175 Hz. The last
three points extend 7.59375 Hz above the prior maximum while staying inside the
unchanged 50–700 Hz range. No old PCM is regenerated. Exact carrier metadata
must not collide with old TRAIN or the twelve reserved evaluation configurations;
reject a collision before generation rather than remove a frequency afterward.
Frequency refinement is deliberate TRAIN adaptation informed by the previous
support failure, not an independent generalization test.

For every new carrier use the full Cartesian product of:

- phase triples `(0.17, -0.59, 1.23)` and `(1.71, 0.43, -1.11)` radians;
- gains `0.035` and `0.075`;
- post-change harmonic coefficients `(0, 1, 0.5)` and `(0.2, 1, 0.5)`.

Before the change the coefficients are `(1, 0.5, 0.25)`. Each sequence contains
96,000 samples at 48 kHz, with change sample 39,840, no noise and no
normalization. The carrier remains constant. Preserve the old Float32 synthesis,
original v2 feature extraction, expanded seven-candidate acquisition and grouping
unchanged. Every case retains the existing 40 ms trailing support, 10 ms hop,
20 ms measurement delay, 196 acquired frames, and 191 common-eligible targets
(75 before, 4 boundary, 112 after the harmonic change).

The exact new bank is **1,536 configurations**, 301,056 acquired frames and
293,376 common-eligible frames, with 3,072 seconds of newly generated synthetic
TRAIN audio. Iterate carrier, phase, gain, mixture in that order. IDs are
`frequency-diversity-train-f{i}-p{j}-g{k}-m{m}`; case indices are contiguous
0..1535 while frequency indices retain lattice values. The auditor must verify
a bijection between numeric frequency and frequency index; an ID cannot create
artificial carrier diversity.

## Unchanged support decision

Use analytic TRAIN truth and candidate coverage only, at inclusive 50 cents.
Preserve all original raw candidates and source evidence; no candidate admission,
grouping, scoring, voicing or correctness threshold changes are permitted.
No learned output is computed. The primary decision uses the **new 1,536-case
bank alone**, without borrowing qualifying rows from the old bank:

| Minimum per targeted stratum | Value |
| --- | ---: |
| Common-eligible rows | 512 |
| Distinct complete configurations represented | 24 |
| Distinct numeric carrier frequencies | 6 |
| Phase settings | 2 |
| Gains | 2 |

Added-only: no original raw candidate is correct, a distinct added peak is
correct, and the expanded grouped pool retains a correct candidate.
Counterexample: a distinct added peak is wrong, while both original and expanded
grouped pools offer a correct candidate. Broad: every common-eligible TRAIN row,
including uncovered rows. Targeted strata are disjoint; broad overlaps both.
Both targeted strata must meet all five unchanged floors.

Publish every case and all pre/boundary/post counts locally. Report raw/grouped
coverage and loss, per-stratum carriers and cases, the existing fixed frequency
bins `[50,150,250,350,450,550,650,700]`, per-phase/gain counts, adjacent qualifying
carrier gaps, and largest carrier share of qualifying rows. These are descriptive
concentration diagnostics, not additional gates or an outcome-based selection
rule. Closely spaced carriers and shared noiseless waveforms are dependent;
passing a distinct-frequency floor does not establish broad acoustic diversity,
learnability, calibrated confidence or natural-speech generalization.

The prior 512-case result is a frozen comparison. A combined 2,048-case support
summary may be reported descriptively, with old/new contributions explicit;
it cannot rescue a failing primary new-bank decision. Do not regenerate,
rescore model outputs, alter or relabel the original no-go. All 74 previously
consumed analytic cases remain historical regressions without rerunning them.
The twelve reserved fresh configurations remain metadata only and unconsumed.
No natural TRAIN/DEV/test audio or reference content is accessed.

## Terminal behavior and independent verification

This runner terminates after coverage reporting on either support PASS or FAIL.
It has no fitting, sampling-schedule, checkpoint, learned-inference, acoustic-
evidence extraction, natural-evaluation, export or browser-replay phase. A PASS
means this TRAIN bank meets the existing support floor; it does not authorize
training or overturn the prior scientific/model-promotion no-go. All natural,
analytic and timing advancement gates remain unchanged and unexecuted.

Freeze the exact new specification and source snapshots in a new ignored run
folder before generation, including the complete old scientific dependency
closure and new audit/tests/plan. Bind the previous source freeze, completed and
failed manifests, support report, independent receipts and repair receipt by
hash. Every generated PCM, original/expanded feature artifact, provenance object,
frame table and report receives an identity hash. Preserve manifest transitions,
process ownership, exact command, resource observations and any failure.

The independent auditor enumerates the new grid without calling the producer,
reconstructs timing, original/expanded candidate retention and grouping, coverage,
strata and numeric carrier support, checks all reported totals, and verifies
freeze-before-generation chronology and zero dependent scientific outputs.
Independent low-level grouping/coverage validators from the preceding audits
may be reused; the producer's decision/counting functions must not be called.
Do not synthesize waveforms or run learned inference during audit. Keep artifacts
and source bytes unchanged across the run and verify the old evidence again
before publication. Hash chronology is not an operating-system access log.

## Local bounds and publication

One scientific attempt, at most 1,536 new two-second TRAIN sequences, zero fits,
zero evaluation cases. Use the installed local Python/NumPy/PyTorch and Node/tsx
environment, deterministic CPU signal processing and grouping. GPU fitting is
inapplicable to this coverage-only question; no remote computation, installs,
new environment, external data acquisition or native/Rust builds are permitted.

Allow at most 1,800 seconds scientific wall time, with a supervisor watchdog at
1,830 seconds. Each replay or worker request is bounded by 60 seconds and the
remaining scientific budget. Use one orchestrator and at most one active Node
extraction worker at a time. Enforce observed per-process Python RSS <=16 GiB,
Node/launcher RSS <=2 GiB, new run plus supervisor outputs <=8 GiB, and at least
16 GiB free disk. Node heap is limited to 1,536 MiB. Use an exclusive process
lease; do not duplicate an active or completed attempt. The resource limits are
observed bounds, not a guarantee about whole-machine memory.

Before execution, run meaningful fixed-toy tests of metadata, support arithmetic,
retention, timing, source binding, deadline/no-overwrite guards and independent
corruption detection. Do not probe the new scientific grid during preflight.
After the single run, finish independent audits, full research tests,
`npm run check`, source/hash checks, high-level README and detailed results.
Commit only validated source, tests, aggregate receipts and documentation with
`[skip ci]`; update existing PR #33, verify zero hosted Actions runs across all
event types, do not merge, and keep the default DSP estimator unchanged. Raw
media, features, references, arrays and full ledgers remain local and ignored.
