# Targeted exposure versus causal acoustic evidence

The saved results support a specific finding: **targeted TRAIN exposure repairs
six retained harmonic-selection failures; this compact acoustic feature alone
does not**. Both targeted arms pass all 78 required analytic cases in every
seed, while both existing-exposure arms pass 72. The predesignated targeted-plus-
evidence model reaches **8.4135%** consumed-development joint error. Its
**0.2214 percentage-point** improvement over the original MLP misses the unchanged
0.5-point gate, so **learned-model promotion remains a no-go**.

Execution status is separately **failed**. All twelve fits, final checkpoints,
frozen predictions and metrics were saved before a terminal stdout serialization
error. The attempt, error and artifacts remain unchanged. A frozen failed-run
audit and a separately declared post hoc audit inspect those saved artifacts;
neither repeats fitting, generates inputs or reclassifies the execution as a
successful run. This report distinguishes that engineering failure from the
scientific decision already supported by saved results.

## Frozen comparison

The [prospective plan](EXPOSURE_STUDY_PLAN.md) crossed existing versus targeted
TRAIN exposure with one causal periodicity feature off versus on, using seeds
20261022, 20261023 and 20261024. The predesignated candidate was `targeted__on`.
Each of the twelve fits performed twelve epochs and 396 optimizer updates;
there was no outcome-based model, seed or checkpoint selection.

The targeted arm consumed only the audited 1,536-configuration bank from the
[frequency-diversity study](FREQUENCY_DIVERSITY_RESULTS.md). Each 512-row auxiliary
batch draws 128 added-only, 128 counterexample and 256 broad rows using fixed
case-first sampling. The existing arm uses the original 96-case bank. This
intervention changes bank composition and sampling together; it does not isolate
an oversampling effect. All natural schedules, initialization, scaling, optimizer
settings and loss weights are matched. Both modes allocate 3,012 parameters,
including three zero-initialized residual weights; off supplies zero channels.
Their effective information and trainable capacity differ.

The feature measures correlation at the candidate period and its half, third
and quarter periods, using only the same 40 ms trailing PCM window. Candidate
pool, actual grouped frequencies, original MLP voicing, 10 ms hop, 20 ms
measurement delay and inclusive 50-cent correctness threshold remain unchanged.

All prerequisites passed: new-bank support, DEV fixed-voicing oracle headroom,
and TRAIN/DEV grouping-loss limits. The bank retains all 1,844 uncovered rows
at 696.0505 and 698.58175 Hz; broad sampling does not filter those failures.
The earlier four-carrier support no-go remains unchanged.

## Natural speech: modest improvement, gate still failed

Percentages below average speaker/condition cells equally. Seed columns are
joint error, including wrong voicing, unavailable output and pitch errors over
50 cents. The dataset contains 320 consumed development records under four
conditions, not an untouched speaker cohort.

| Exposure / acoustic feature | Seed 20261022 | Seed 20261023 | Seed 20261024 | Mean joint error |
| --- | ---: | ---: | ---: | ---: |
| Existing / off | 8.4643% | 8.4751% | 8.4373% | **8.4589%** |
| Existing / on | 8.4577% | 8.4679% | 8.4291% | **8.4516%** |
| Targeted / off | 8.4281% | 8.4039% | 8.4130% | **8.4150%** |
| Targeted / on, predesignated | 8.4275% | 8.3998% | 8.4132% | **8.4135%** |

Original MLP error is **8.6349%**; original hybrid error is **8.9224%** on these
same consumed records. These figures must not be compared directly with the
previous reserved-speaker evaluation in the README. The existing/off refits
reproduce the historical expanded-pool mean and saved controls remain exact.

With evidence off, targeted exposure reduces mean error by **0.043926 points**.
Evidence reduces it by **0.007356 points** under existing exposure and
**0.001504 points** under targeted exposure. The error-scale interaction is
**+0.005853 points**, indicating a smaller incremental feature benefit with
targeted exposure. In seed 20261024, adding evidence to targeted exposure slightly
worsens error. These are paired descriptive differences across three seeds, not
statistical significance or external generalization claims.

| Predesignated candidate gate | Result | Decision |
| --- | ---: | --- |
| Mean error improves by at least 0.5 points over original MLP | 0.221445 points | **FAIL** |
| Mean error improves by at least 0.5 points over original hybrid | 0.508928 points | PASS |
| Every seed improves on original MLP | 3 / 3 | PASS |
| No condition mean worsens by more than 1 point versus MLP | All four condition means improve | PASS |
| Unconditional reference-voiced success not lower than MLP | 76.3453% versus 75.4057% | PASS |
| Every mandatory analytic case reaches 95% in every seed | 78 / 78 for each seed | PASS |
| All advancement gates pass together | Natural MLP margin fails | **NO-GO** |

No control arm replaces the predesignated candidate after observing results.
Calibration, external validity and clinical utility remain unestablished.

## Retained failures and fresh cases

| Exposure / feature | Mandatory gates passed, each of three seeds | Fresh post-change cases passed, each seed |
| --- | ---: | ---: |
| Existing / off | 72 / 78 | 12 / 12 |
| Existing / on | 72 / 78 | 12 / 12 |
| Targeted / off | 78 / 78 | 12 / 12 |
| Targeted / on | 78 / 78 | 12 / 12 |

The six remaining existing-exposure failures comprise two phases at each of
517.3, 633.1 and 641.9 Hz. Existing arms select correctly on **0/112** post-change
frames at 517.3 and 641.9 Hz, and **24/112** or **23/112** at 633.1 Hz. Both targeted
arms reach **112/112** for all six cases in every seed. Earlier 337.1 and 354.5 Hz
cases remain regressions and continue to pass. This comparison supports exposure
as the effective intervention for these cases; it does not show that all forms
of candidate-conditioned acoustic evidence are unnecessary.

All twelve fresh configurations were declared before fitting and generated once
after the complete checkpoint freeze. All arms and seeds reach **112/112**
post-change frames on every fresh case, so these cases do not distinguish the
interventions. There are twelve configurations, not 144 independent experiments.
The exact configurations were fresh, but a TRAIN carrier lies only 0.34425 Hz
from a reserved carrier. Added-only training examples cluster at high frequencies
and counterexamples mostly at low frequencies; frequency shortcuts remain
possible. The noiseless harmonic signals do not establish broad acoustic
robustness. All 86 analytic cases are now consumed regressions.

The unchanged gate covers three stationary cases, three original post-change
cases, sixty consumed harmonic-change cases and twelve fresh post-change cases.
All other whole/pre/boundary/post/step/glide outcomes remain in local metrics;
passing the 78 mandatory cases does not assert perfection on every reported
region. The new bank's upper-frequency candidate-coverage holes remain visible.

## Preserved reporting failure and audit scope

The terminal `json.dumps(decision)` cannot serialize NumPy boolean gate leaves.
The metrics writer had already converted those scalars and safely saved the
metrics. The outer failure handler then recorded the exception and marked the
attempt failed. The full-path preflight fixture missed this because its score
stub returned Python floats. A separate regression fixture now exercises the
real summary arithmetic and expects this archived reporting failure; the frozen
runner is not patched or rerun.

The original frozen auditor checks the failed attempt's artifact inventory.
The first post hoc forensic attempt stopped at a separate inherited admission
bug: fresh scenarios include all eight expected timing/pitch fields plus five
declared waveform fields, while the old auditor expected only eight fields.
That failure is also preserved. A separately versioned adapter requires **all
thirteen fields** to match the frozen specification exactly, including change
sample, harmonic amplitudes, phases and gain; it does not drop fields or alter
score equations, numerical tolerances or advancement gates.

The versioned forensic auditor uses the **actual failed manifest**, frozen
independent model equations and unchanged gate functions to inspect all twelve
checkpoints, saved outputs, targets, schedules, scores and phase bindings. Its
own source and tests freeze before execution. It cannot rewrite source, status,
metrics, checkpoints or predictions. The raw-periodicity audit reconstructs the
predeclared 4,608 windows from saved TRAIN PCM at acquired indices 4, 80 and 194,
with the original 2e-13 correlation and 1e-7 channel tolerances.

The final saved-output audit **passed**: **3,261,936 frame-model comparisons**
across 406 datasets produced zero state, pitch or voicing-gate mismatches.
Raw replay covered 4,608 frames and 17,530 active candidate windows; maximum
correlation difference was **5.55e-16** versus tolerance 2e-13, and maximum
channel difference **7.11e-15** versus tolerance 1e-7. All ten completed audit
stages remain inspectable. This qualification preserves both prior failures
and the unchanged scientific no-go.

Optimizer execution and TRAIN before/after logit-diagnostic summaries are not
independently replayed. Those diagnostic summaries remain hash-bound producer
descriptions. The raw check samples three windows per synthetic TRAIN case,
not every natural/acoustic feature or the original extractor. Hashes and phase
timestamps prove identity and declared chronology, not an operating-system
access log. Post hoc artifact qualification does not erase the failed run or
waive the natural-speech gate.

## Local execution and evidence

The single scientific attempt used **159.27 seconds of MPS fitting** and ended
after **561.59 seconds**; the external supervisor finished cleanup at **562.77
seconds**. Producer-observed peak Python RSS was **7.25 GB**, MPS allocation
**179 MB**, and MPS driver allocation **1.13 GB** descriptively. External sampled
Python RSS peaked at **7.24 GB**. Observed run-plus-execution outputs were about
**243 MB** and free disk remained above **297 GB**. All observed limits passed;
these are process samples, not continuous whole-machine guarantees. The lease
was released and no study processes remained.

The full research suite passed **1,119 tests** before the scientific attempt and
**1,210 tests** after all audit-adapter changes; `npm run check` also passed.
Twenty-three supervisor lifecycle fixtures passed before freeze. All 82 frozen
scientific sources remain byte-identical. Three historical NumPy matrix-product
warnings remain recorded; assertions passed and their cause is unestablished.
Scientific scoring emitted a PyTorch non-writable-array warning, retained in the
log. Independent target/output verification is reported separately from warnings.
Product/browser/database/native build suites were not rerun for this research
change. Preflight coordinate and supervisor fixture failures, the post hoc
declaration path setup error, and both substantive failed attempts are preserved.

All computation used this Mac and the existing environment. There were no
installs, native/Rust builds, remote jobs, new natural-data acquisition, model
exports, browser qualification or default DSP changes. The original 74 analytic
cases and all historical failures are preserved; the twelve new cases are not
available for fresh reuse. All PTDB speakers have observed results.

- [Frozen prospective plan](EXPOSURE_STUDY_PLAN.md)
- [Public aggregate results](../../benchmarks/voice-prosody/exposure-study-summary.json)
- [Independent audit receipts](../../benchmarks/voice-prosody/exposure-study-independent-audit.json)
- [Local validation and resource checks](../../benchmarks/voice-prosody/exposure-study-local-validation.json)
- [Preserved earlier no-go](EVIDENCE_RESULTS.md)

Raw media, features, references, weights, frame predictions and complete ledgers
remain local and ignored. Inspect existing artifacts without rerunning science:

```sh
cd research/voice-prosody
.venv/bin/python -m json.tool runs/exposure-study-v1/metrics.json
.venv/bin/python -m json.tool runs/exposure-study-v1/independent-audit.json
.venv/bin/python -m json.tool runs/exposure-study-v1/forensic-audit.json
.venv/bin/python -m json.tool runs/exposure-study-v1/forensic-audit-v2.json
```

Any future execution first needs a separately versioned runner with corrected
terminal serialization; this archived attempt must not be resumed. The next
research question is natural-speech robustness under a separately frozen, broader
acoustic intervention. This study authorizes neither more fits
nor another fresh evaluation set. Its useful result is a demonstrated repair
of retained selection failures and a narrower diagnosis of the remaining gap.
