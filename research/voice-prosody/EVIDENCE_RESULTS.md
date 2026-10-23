# Exposure and acoustic-evidence study: audited feasibility no-go

The prospective study stopped before fitting. Its new TRAIN bank supplied enough
qualifying frames and configurations, but too few distinct fundamental
frequencies to meet the frozen support prerequisite. Both independent audits
confirm the no-go. **Zero models were fitted; zero fresh evaluation cases were
generated or consumed.** No gate, candidate policy, voicing rule, timing contract,
or default estimator changed.

The scientific question remains open: do added-only correct pitch candidates
need more training exposure, more candidate-conditioned acoustic evidence, or
both? This result establishes insufficient support for the declared comparison,
not a failure or benefit of the proposed feature.

## Why this comparison

The previous [expanded-pool study](EXPANDED_POOL_RESULTS.md) retained the original
six candidates and added a seventh, yet the selectors missed every post-change
frame at 517.3 and 641.9 Hz despite complete candidate coverage. Its existing
96-case synthetic TRAIN bank contained no cases where only the added candidate
was correct. The new [prospective plan](EVIDENCE_EXPERIMENT_PLAN.md) therefore
crossed two interventions across three fixed seeds:

| Factor | Control | Intervention |
| --- | --- | --- |
| TRAIN exposure | Existing 96-case bank | Fixed 512-case bank with coverage-targeted sampling |
| Acoustic evidence | Zero residual inputs | Candidate-conditioned periodicity evidence |

The predesignated arm was targeted exposure plus evidence. All four arms would
share the expanded candidate pool, natural TRAIN schedule and scaling, original
MLP voicing, and final-epoch checkpoint policy. The maximum was twelve fits;
none began. Allocated parameter count was matched at 3,012, including a
zero-initialized, bias-free three-weight residual. Zero inputs in the control
mean effective information and trainable capacity differ; this is not a claim
of equal effective model capacity.

## Measured TRAIN support

The fixed bank has 64 carrier frequencies, two phase settings, two gains, and
two harmonic mixtures: 512 complete two-second configurations. The completed
attempt acquired 100,352 frames, of which 97,792 met common eligibility. Original
raw and grouped pools covered 94,132 eligible frames; expanded pools covered
all 97,792. Neither policy lost correct candidates during grouping.

“Added-only” means the original raw pool lacks a correct candidate, a distinct
added peak is correct, and the expanded grouped pool retains a correct
candidate. “Counterexample” means the added peak is wrong while both original
and expanded grouped pools offer a correct candidate. Correct means within
50 cents; these are analytic TRAIN labels, not evaluated model outputs.

| Support dimension | Frozen minimum per targeted stratum | Added-only | Counterexample |
| --- | ---: | ---: | ---: |
| Eligible frames | 512 | 3,660 | 26,796 |
| Complete configurations represented | 24 | 32 | 240 |
| Distinct carrier frequencies | 6 | **4 — fail** | 30 |
| Phase settings | 2 | 2 | 2 |
| Gains | 2 | 2 | 2 |

Repeated frames do not substitute for carrier diversity. The failed carrier
floor blocked sampling schedules, natural-data prerequisite scoring, acoustic
feature extraction for the scientific comparison, fits, checkpoints, model
predictions, and fresh evaluation generation. The bank was not enlarged and
no floor was relaxed after observing this outcome.

All 74 previously consumed analytic cases remain historical regressions; they
were not rerun here. Twelve fresh configurations were reserved as metadata
before acquisition and remain ungenerated. All PTDB-TUG speakers have already
contributed observed results. Additional prompts from them would not establish
untouched-speaker validation.

## Engineering completed

The research feature computes normalized correlations from the existing trailing
1,920 Float32 samples at 48 kHz. After Float64 mean removal, it compares the
actual grouped candidate period with half, third, and quarter periods using
linear interpolation and a common anchor. The three model inputs are full-period
correlation, contrast against the strongest subperiod correlation, and explicit
validity. Low-energy and padded candidates produce zero evidence with false
validity. Grouping precedes extraction, preventing evidence from different
emitted frequencies from being merged.

Independent TypeScript and NumPy implementations pass fixture parity, exact
support, validity, interpolation, masking, and causality checks. A fixed
1,000-frame toy Python extraction reached 2,109.30 frames/s with observed peak
RSS 96,387,072 bytes. This was a feasibility screen, not a browser benchmark or
measurement of learned accuracy. The residual scorer preserves base
initialization and passes fixture optimizer parity with evidence disabled.
The planned MPS training path has not been exercised end to end by this study.

The local HTML report displays the saved per-case and regional support counts,
failed prerequisite, audit identity, and repair history. Model ranking, voicing,
and accuracy outcomes remain explicitly unassessed. It embeds no raw media or
natural reference trajectories and requires no network resources. Report
fixtures and static HTML/JavaScript/JSON checks passed. Visual browser inspection
was unavailable because local browser use is disabled in Work settings; no new
browser qualification or learned-model export is claimed.

## Preserved failures and repair chronology

Attempt v1 acquired the same 512 TRAIN configurations, then stopped with
`ValueError('Preparation input bytes changed')`. A narrow adapter repair had
been copied after the source freeze: historical generated predictions needed
to bind to the prior output hash map rather than its input hash map. The
integrity guard rejected the changed bytes before coverage outcomes, fitting,
or fresh generation.

The failed attempt, its source snapshots, logs, and acquisition artifacts remain
intact. An explicit repair receipt records the sole changed source and a new v2
freeze. V2 repeated the same TRAIN acquisition and reached the support no-go.
The terminal audit compared all 512 paired PCM and feature arrays: they are
identical across attempts. Both acquisitions count toward resource accounting;
1,024 TRAIN sequences were generated in total. No completed fit or evaluation
was repeated, and no historical artifact was overwritten or rebound.

During final engineering validation, three report fixture failures exposed a
Python default-argument binding that captured the repository root too early.
The renderer now resolves the default when instantiated; all 23 report tests
pass. This repair is outside the 58-file scientific source closure, whose bytes
remain unchanged. The initial failure record is retained locally. A separate
static-check script initially treated an embedded JSON block as JavaScript;
its corrected check validates the two types separately and preserves that
checker error. Neither repair changed scientific inputs or outcomes.

The overnight window ended before auditing and shipping finished. Its original
incomplete closeout remains intact. The subsequent user-authorized continuation
completed audits, fixture validation, documentation and publication only.

## Independent audits and local validation

| Check | Result | Scope |
| --- | --- | --- |
| Feasibility audit | PASS; 2,229 bound artifacts | Reconstructs eligibility, raw/grouped support, strata and failed floor from saved inputs; verifies no schedules or fresh inputs |
| Terminal study audit | PASS; 3,361 bound artifacts; 43 numeric comparisons, maximum difference 0 | Verifies no-go, exact source closure, preserved repair lineage, repeated TRAIN identity, and zero-fit/zero-fresh terminal state |
| Full research suite | **938 passed in 12.91 s** | Existing environment on macOS arm64; includes feature parity, scoring contracts, audit corruption tests and report fixtures |
| Required repository check | **PASS** | `npm run check`; no installs or native builds |
| TypeScript static check | **PASS** | Strict no-emit check of the periodicity kernel and worker |
| Local report | **23 fixture tests pass; static checks pass** | Support-only rendering; browser visual inspection unavailable |

The full suite emitted three NumPy matmul runtime warnings in an existing
batched-forward fixture; all assertions passed. The same warnings appear in earlier validation logs; their cause remains
undetermined. The complete log is retained.
Audit PASS validates the recorded no-go; it does not satisfy scientific
advancement. Saved source and artifact chronology is not an operating-system
access log. Post-fit scoring and learned-output replay were inapplicable and
were not executed.

All computation used the existing local environment. V1 and v2 outer execution
lasted 114.83 and 160.19 seconds, respectively (275.02 seconds total). Supervisor-sampled
v2 individual-process peaks were 795,344,896 bytes Python RSS and 287,457,280 bytes Node-worker RSS.
The final supervisor sample counted 453,927,653 disposable bytes across both
attempts; that sample is not an absolute storage peak. Fitting time was zero; the v2 runner recorded zero
MPS allocation. No installs, native/Rust rebuilds, remote compute,
new natural-data acquisition, or reserved/test evaluation occurred.

The unchanged scientific advancement gates still require natural joint-error
improvement against both original MLP and hybrid, per-seed and per-condition
constraints, no drop in unconditional voiced success, and at least 95% within
50 cents on every declared analytic gate. They were not reached. The default
DSP estimator remains unchanged, with 40 ms support, 10 ms hop and 20 ms
measurement delay in this research contract. Confidence remains uncalibrated;
no clinical or external-generalization result follows.

## Evidence and inspection

Public aggregate receipts:

- [Study summary](../../benchmarks/voice-prosody/evidence-study-summary.json)
- [Independent feasibility audit](../../benchmarks/voice-prosody/evidence-feasibility-independent-audit.json)
- [Independent terminal audit](../../benchmarks/voice-prosody/evidence-study-independent-audit.json)
- [Local validation](../../benchmarks/voice-prosody/evidence-local-validation.json)

They contain aggregate counts and identity hashes. Full audit ledgers, the
self-contained HTML viewer, source snapshots, logs, raw inputs and arrays remain
in ignored local `runs/` directories. The completed study manifest SHA-256 is
`f586e51633e3d784e13d8065412ba55926fd344bc900d5f76672d7603fe3322d`;
the scientific source-freeze SHA-256 is
`b4a584b204b513c4c7721117fce0e4e3562db0f644df14e5f25f39688881613b`.

With the existing environment and saved artifacts, these commands inspect the
completed result and write new audit/report files without fitting or generating
waveforms. Choose unused output paths; do not overwrite the original receipts.
Run from `research/voice-prosody`:

```sh
.venv/bin/python -m phenometrix_prosody.audit_evidence_feasibility \
  --preparation runs/evidence-factorial-v2/feasibility \
  --output runs/evidence-factorial-v2/feasibility/audit-review.json
.venv/bin/python -m phenometrix_prosody.audit_evidence_scores \
  --run runs/evidence-factorial-v2 \
  --output runs/evidence-factorial-v2/audit-review.json
.venv/bin/python -m phenometrix_prosody.local_evidence_report \
  --preparation runs/evidence-factorial-v2/feasibility \
  --repair-receipt runs/overnight-20261023/repair-v1-to-v2.json \
  --expected-repair-sha256 759dd150be4b7fb2ab8cbcf4fca1cc3a0f3522dfa6dc0291f6381cd5d84c8cb2 \
  --output runs/evidence-factorial-report-review
```

The saved viewer is `runs/evidence-factorial-report-v1/index.html`. A future
scientific attempt would require its own prospective specification and
explicit authorization; the expired overnight runner is not a reproduction
command for this validation-only continuation.
