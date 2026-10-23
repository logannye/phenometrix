# TRAIN-only frequency-diversity result

**The new TRAIN bank passes the unchanged support prerequisite. The learned
estimator remains a no-go for promotion.** This single local coverage experiment
produced no fit, sampling schedule, learned prediction, checkpoint, export,
browser qualification, or evaluation input. The proposed comparison of training
exposure and acoustic evidence remains untested.

## Question and frozen intervention

The [preceding study](EVIDENCE_RESULTS.md) stopped because added-only correct
candidates occurred at four numeric carrier frequencies, below the floor of six.
Thousands of frames from too few carriers could not satisfy that prerequisite.
The new [prospective plan](FREQUENCY_DIVERSITY_PLAN.md) asks whether a fixed
refinement of the same TRAIN lattice supplies adequate support. It changes the
TRAIN bank, without changing extraction, additive candidate admission, grouping,
voicing, timing, the inclusive 50-cent criterion, or any advancement gate.

The plan, exact metadata, 71 scientific source files, and resource supervisor
froze before generation. Of 256 lattice points spaced 2.53125 Hz apart, the 64
previous carriers remain an unchanged, already-observed baseline. Only the other
192 carriers were generated, at two phases, two gains, and two harmonic changes:
**1,536 new two-second synthetic configurations**. Three upper points extend the
old grid while remaining inside 50–700 Hz. The primary decision uses this new
bank alone; combining old and new evidence cannot rescue its decision.

The run acquired 301,056 frames, of which 293,376 are common-eligible under the
existing 40 ms support, 10 ms hop, and 20 ms measurement-delay contract. All data
processing ran on the local Mac in the existing environment. CPU signal
processing is appropriate to this coverage question; there was no GPU fitting,
remote compute, installation, native build, or natural-data acquisition.

## Primary result

Added-only rows have no correct original candidate but retain a correct added
candidate after grouping. Counterexample rows have a wrong distinct added peak
while both original and expanded grouped pools retain a correct candidate.
These targeted strata are disjoint; the broad stratum includes every eligible
row, including uncovered ones.

| Support dimension | Unchanged minimum, each stratum | New added-only | New counterexample |
| --- | ---: | ---: | ---: |
| Eligible rows | 512 | 11,456 | 79,576 |
| Distinct configurations represented | 24 | 104 | 716 |
| Distinct numeric carrier frequencies | 6 | 13 | 90 |
| Phase settings | 2 | 2 | 2 |
| Gains | 2 | 2 | 2 |
| Decision | All five required | PASS | PASS |

The independently reconstructed old bank still fails: 3,660 added-only rows,
32 configurations, and four carriers. Its counterexample stratum has 26,796
rows across 240 configurations and 30 carriers. The descriptive combined bank
contains 2,048 configurations, 15,116 added-only rows at 17 carriers, and 106,372
counterexample rows at 120 carriers. None of these combined counts determine
the primary new-bank decision or revise the historical no-go.

Original candidate coverage on the new bank is **280,076 / 293,376 = 95.4666%**;
expanded coverage is **291,532 / 293,376 = 99.3715%**. Raw and grouped coverage
agree, with zero original-candidate eviction and zero grouping coverage loss.
These are candidate-availability measures, not model pitch accuracy. The bank
was deliberately refined after observing TRAIN support failure and is not a
held-out performance comparison.

## Remaining failures and dependence

The 13 qualifying added-only carriers span **516.33175–693.51925 Hz**. Adjacent
gaps range from 2.53125 to 40.5 Hz, the largest carrier contributes 8.03% of
qualifying rows, and approximately **92.02% of those rows are above 550 Hz**.
There are 11,224 post-change and 232 boundary added-only rows, with none before
the change. Gain counts are exactly balanced; phase counts are 5,732 and 5,724.

Counterexamples are concentrated in a different region: 78,972 of 79,576 rows
(99.24%) are below 350 Hz. Four are at 351.8005 Hz; 600 occur before the change
at 698.58175 Hz. The strata therefore remain strongly associated with frequency
and harmonic regime. A future selector could exploit frequency shortcuts;
this experiment cannot show that it learns candidate-conditioned acoustic
relationships. Dense nearby frequencies and repeated noiseless waveforms also
do not supply independent acoustic observations.

| Eligible region | Rows | Added-only | Counterexample | Expanded uncovered |
| --- | ---: | ---: | ---: | ---: |
| Before change | 115,200 | 0 | 600 | 0 |
| Boundary | 6,144 | 232 | 714 | 52 |
| After change | 172,032 | 11,224 | 78,262 | 1,792 |

All **1,844 uncovered rows** belong to the 16 TRAIN configurations at **696.0505
and 698.58175 Hz**. Every one of these configurations lacks a correct expanded
candidate on all 112 post-change targets. They remain visible in the saved
per-case report; they were neither excluded nor repaired after the result.
The unchanged support prerequisite permits uncovered broad rows. Passing it
does not mean every TRAIN frequency is covered.

Exact carrier metadata does not collide with old TRAIN, historical configured
carriers, or the twelve reserved evaluation configurations. The closest new
carrier to a reserved carrier is nevertheless only **0.34425 Hz** away. Exact
noncollision is not independent frequency support. All 74 previously consumed
analytic cases remain historical regressions without being rerun, and the twelve
reserved configurations remain metadata only. No natural TRAIN/DEV/test audio
or reference content was accessed for this study; all PTDB speakers already
have observed results, so no untouched-speaker claim is available.

## Independent verification and local bounds

The frozen auditor independently enumerates metadata, validates the numeric
frequency/index bijection, and reconstructs timing, retained candidates,
grouping, coverage, strata, support counts, and the complete reported aggregates.
It checks saved new and old TRAIN artifacts, source identity, freeze chronology,
and the absence of dependent scientific outputs. Audit correctness and TRAIN
support are separate fields; neither is a model-promotion result.

The single scientific run completed in **556.13 seconds** under its 1,800-second
limit. The external supervisor records clean exit and no surviving process group;
a separate closeout audit checks its receipt, resource observations, residual
process markers, and released lease. Sampled memory and disk bounds apply to
the observed processes and outputs, not all machine activity or unobserved
sub-poll peaks. Full validation and exact hashes are published in the
[local validation receipt](../../benchmarks/voice-prosody/frequency-diversity-local-validation.json).

The full research suite passes **1,020 tests** before and after the run, including
38 new producer and 44 new independent-audit fixtures. Sixteen supervisor toy
lifecycle tests pass, and `npm run check` passes. Three historical NumPy matrix
multiplication warnings remain recorded; assertions pass and their cause remains
undetermined. Product/browser/database/native build suites were not rerun for
this research-only change; the production estimator is unchanged.

Preflight failures remain recorded locally: a boundary fixture assumed a
rounded 50-cent exponential was exactly on the logarithmic cutoff, and supervisor
toy checks exposed fixture issues and a lease append-mode bug. Corrections and
passing checks occurred before source freeze; the scientific thresholds did not
change. The prior study's failed preparation attempt, explicit repair, completed
no-go, and all 58 historical scientific source files remain intact. There was
exactly one scientific generation attempt in this follow-up and no adaptation
to its outcomes.

## Evidence and inspection

- [Frozen prospective plan](FREQUENCY_DIVERSITY_PLAN.md)
- [Public aggregate result and source bindings](../../benchmarks/voice-prosody/frequency-diversity-summary.json)
- [Independent audit receipt](../../benchmarks/voice-prosody/frequency-diversity-independent-audit.json)
- [Local validation and resource receipt](../../benchmarks/voice-prosody/frequency-diversity-local-validation.json)
- [Preserved prior no-go](EVIDENCE_RESULTS.md)

Full local artifacts are ignored under `runs/frequency-diversity-v1`; raw PCM,
feature arrays, frame tables, and complete artifact ledgers are not distributed.
The saved report contains every configuration and regional outcome. The
following commands inspect existing evidence without generating inputs:

```sh
cd research/voice-prosody
.venv/bin/python -m json.tool runs/frequency-diversity-v1/report.json
.venv/bin/python -m json.tool runs/frequency-diversity-v1/independent-audit.json
.venv/bin/python -m json.tool runs/frequency-diversity-execution-v1/execution.json
```

Hashes establish byte identity and declared chronology, not an operating-system
access log. The independent audit verifies PCM hashes and candidate union/grouping;
it does not resynthesize waveforms or independently replay the full raw
correlation extractor. This result supports designing a future frozen learning
comparison using adequate TRAIN support. It establishes no accuracy improvement,
calibration, external generalization, clinical validity, or promotion readiness.
