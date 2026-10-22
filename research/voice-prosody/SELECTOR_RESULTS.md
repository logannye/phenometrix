# Candidate representation and acceptable-set learning: step 1

**The shared representation repairs the stationary 300 Hz regression across all
three seeds, but the predesignated combined model fails the natural-improvement
gate.** Its development joint error averages 8.4825%, compared with 8.6349% for
the original MLP and 8.9224% for the original hybrid. The improvement of 0.1524
percentage points over MLP is below the predeclared 0.5-point requirement. No
method is promoted, and no temporal model, calibration or reserved/test evaluation
was started. Production remains DSP.

## Hypothesis, preregistration and data

The [plan](SELECTOR_EXPERIMENT_PLAN.md) was written before headroom outcomes or
new fits. The preceding [diagnosis](ALIAS_DIAGNOSIS.md) found position-sensitive
candidate slots and exclusive targets despite multiple acceptable candidates in
over 92% of covered natural training frames. This new experiment tested their
combined representation/objective repair while preserving the 20 ms measurement
delay, 40 ms support, 10 ms hop, 50-cent tolerance and common reference mask.
Earlier failed gates remain failed under their original contracts.

Train: 240 sources from 12 speakers, 960 condition versions, 669,664 valid eligible
frames. Development: 80 sources from four speakers, 320 condition versions,
253,292 valid eligible frames. Speakers, source recordings and selected prompts
are disjoint. Both datasets and all synthetic fixtures were already used in
previous research. The four formerly reserved speakers are excluded from this
stage. Repeated corruption versions and individual frames are not independent
participants. References retain nominal provider timing and validity; no alignment
search, reference repair or outcome-based exclusion is performed.

## Headroom before fitting

The [headroom audit](../../benchmarks/voice-prosody/selector-headroom.json) froze
all training forward predictions and reused development predictions before its
first reference read. It records 4,504 input identities and source bindings.
A separate crosscheck reproduced the existing development baseline counts and
rates. The oracle uses reference outcomes and cannot be deployed.

| Development diagnostic, frozen original voicing | Joint error |
|---|---:|
| Original MLP | 8.6349% |
| Original hybrid | 8.9224% |
| Oracle chooses only MLP's selected pitch or raw YIN | 7.7668% |
| Oracle chooses any original candidate | 7.3205% |
| Oracle chooses original candidates plus raw YIN | 7.2809% |
| Theoretical perfect pitch, original voicing unchanged | 5.9342% |

Original-candidate headroom is **1.3144 percentage points**, exceeding the fixed
1-point prerequisite. Adding YIN increases that headroom by only **0.0396 points**;
most recoverable alternatives already exist in the original candidate pool.
There are 2,888 development voiced frames correct only for MLP and 2,215 correct
only for hybrid. Both miss 8,005 reference-voiced frames through their common
unvoiced decision, a 13.2679% equal-cell false-unvoiced rate. A pitch selector
cannot recover that shared loss. Full training, condition, speaker and F0 strata
remain in the audit; training scores are resubstitution.

Grouping reduces equal-cell candidate coverage from 86.3825% to 86.3135% on train
and 88.1304% to 88.0885% on development. Actual losses are 0.0690 and 0.0418
percentage points, both below the fixed 0.5-point ceiling; no coverage gains are
possible when representatives are original members. The first grouping process
was rejected because a concurrent audit source edit changed its source identity.
That pre-fit failure is retained locally; the completed v2 grouping audit and
prerequisite record bind the finalized source. No fitting preceded the passing
prerequisite record.

## Fixed factorial comparison

All four arms use natural training only, no synthetic supplement and no new
candidate source. Frozen original MLP voicing probabilities and threshold 0.5
are reused exactly for all new outputs. The pitch-only slot control has 3,110
parameters; the shared scorer has 2,689. All arms use identical original-covered
training rows and seed-specific order, train-only scaling, Adam 0.001, 12 fixed
epochs, batch 4096 and gradient norm clip 1. All **132,181** scheduled rows remain;
**107** lose a grouped target and contribute zero pitch loss in the shared arms.
Every fit completed 396 optimizer updates without a skipped batch.

Groups have at most 5 cents total span; per-source evidence is merged by maximum.
The output is the original member with highest maximum evidence, lower Hz on
ties. Group targets depend on that emitted frequency, never a reference-selected
member. Shared scaling, masked-max context and shared scoring accompany grouping,
so their effects cannot be separated in this experiment. Exact duplicate and
permutation tests do not establish invariance to arbitrary near-duplicate changes.

Acceptable-set loss rewards summed probability on all emitted candidates within
50 cents. It supplies no false pitch label to unvoiced, invalid or uncovered
frames. Valid eligible unvoiced and uncovered frames remain in evaluation
denominators; invalid and boundary references remain excluded and counted.
Changing joint voicing/pitch fitting to pitch-only fitting and a new batch schedule
is common to all new arms; improvements over the original MLP alone cannot be
attributed to the representation or objective intervention.

Seeds are 20261022 / 20261023 / 20261024 throughout:

| Representation | Objective | Joint error by seed | Mean joint error | Mean unconditional voiced success | Stationary 300 Hz success by seed |
|---|---|---:|---:|---:|---:|
| Fixed slots | Nearest | 8.3480% / 8.3496% / 8.3484% | 8.3486% | 76.6424% | 0% / 0% / 0% |
| Fixed slots | Acceptable set | 8.2638% / 8.2845% / 8.2589% | 8.2691% | 77.0015% | 0% / 0% / 0% |
| Shared candidates | Nearest | 8.4802% / 8.4460% / 8.4489% | 8.4584% | 76.1679% | 100% / 100% / 100% |
| Shared candidates | Acceptable set | 8.4688% / 8.4969% / 8.4818% | 8.4825% | 76.0734% | 100% / 100% / 100% |

The acceptable-set objective lowers fixed-slot mean error by **0.0796 points**,
but increases shared-representation error by **0.0242 points**. Both shared
arms repair the stationary 300 Hz failure while both fixed-slot arms still fail
all 151 scored frames in every seed. This supports a representation-package
explanation for this particular regression, rather than a loss-only explanation.
The shared combined model has 0.1339 points more error than the new fixed-slot
nearest control. These comparisons do not isolate grouping, scaling or weight
sharing individually and do not establish broad high-pitch robustness.

The predesignated shared/acceptable-set model improves over original MLP in each
seed and in each condition on average, and its unconditional voiced success rises
from 75.4057% to 76.0734%. It nevertheless misses the required mean gain against
both frozen comparators: **0.1524 points versus MLP; 0.4399 versus hybrid**. The
natural gate remains failed. The best observed arm or seed is not substituted.
Three seeds describe optimization variation, not participant uncertainty.

## Full reused analytic regression

Mean success within 50 cents across three seeds, with every scored target in the
denominator. All 14 declared scenarios are retained. These are reused fixtures,
not a fresh challenge or a completed source/channel temporal-trust study.

| Scenario | Slots/nearest | Slots/acceptable | Shared/nearest | Shared/acceptable |
|---|---:|---:|---:|---:|
| constant-100 | 100.000% | 100.000% | 100.000% | 100.000% |
| constant-200 | 100.000% | 100.000% | 100.000% | 100.000% |
| constant-300 | 0.000% | 0.000% | 100.000% | 100.000% |
| glide-down-200 | 88.079% | 88.079% | 99.338% | 99.338% |
| glide-down-100 | 100.000% | 100.000% | 100.000% | 100.000% |
| glide-down-50 | 100.000% | 100.000% | 100.000% | 100.000% |
| glide-up-50 | 100.000% | 100.000% | 100.000% | 100.000% |
| glide-up-100 | 100.000% | 100.000% | 100.000% | 100.000% |
| glide-up-200 | 78.587% | 78.808% | 100.000% | 100.000% |
| step-100-200 | 99.338% | 98.675% | 99.117% | 98.675% |
| step-200-100 | 99.338% | 99.338% | 99.338% | 99.338% |
| observation-change-100 | 100.000% | 100.000% | 100.000% | 100.000% |
| observation-change-200 | 100.000% | 100.000% | 100.000% | 100.000% |
| observation-change-300 | 65.563% | 65.563% | 100.000% | 79.029% |

A material remaining failure is **shared/acceptable seed 20261023** on the 300 Hz
observation-change fixture: it has 37.0861% success overall and **0/95 correct
after the change**, choosing a subharmonic after the fundamental is removed while
harmonics two and three remain. The other two seeds and all three shared/nearest
seeds pass that fixture. Therefore stationary repair is not robust channel-change
qualification. This failure is preserved rather than hidden by the 79.0287%
three-seed average. Per-seed, regional, step-settling and glide metrics are in the
[aggregate summary](../../benchmarks/voice-prosody/selector-step1-summary.json).

## Execution and evidence boundary

All twelve final checkpoints were frozen before new development predictions;
all new predictions were then frozen before this fitting/scoring process opened
development references. The separate prerequisite audits had already read those
references. This chronology does not restore independence to reused development.
Raw original voicing remains uncalibrated and no correctness probability is claimed.

All computation ran on this Mac. Fitting used the existing PyTorch MPS environment
and took **39.40 seconds** for all twelve fits. Peak
observed MPS allocation was **66.98 MiB**;
this is allocator telemetry, not whole-process RAM or device utilization. The new
fit/prediction artifact set was approximately **10.53 MiB**
at completion. Raw arrays, references and weights stay ignored and local. There
was no package installation, native build, remote fit, new browser qualification
or product-model switch. Browser performance measured for earlier models does
not transfer to these selectors.

The result supports retaining the shared representation as a research hypothesis
and rejecting the assumption that combining both changes necessarily improves it.
The predeclared decision remains **no-go for advancing the combined candidate**.
Further work needs a separately declared experiment and fresh eventual evaluation;
no additional fit or threshold revision was performed after these outcomes.

## Local validation

The complete research suite passes **193 tests** and the repository's required
`npm run check` structure/static-asset check passes. The suite emits three NumPy
matmul warnings (divide by zero, overflow, invalid) in the headroom batch-parity
test; these are retained, not suppressed. Its finite-output checks and batch/
single-frame assertions pass. A separate [Torch CPU parity audit](../../benchmarks/voice-prosody/selector-torch-parity.json)
checks all 960 frozen training prediction files and 676,384 acquired frames:
all probabilities are finite, maximum probability difference is 2.04e-7, and
voicing, state, emitted candidate and emitted frequency have zero mismatches.
All 1,928 input files remain byte-identical. Ungated candidate indices on unvoiced
frames were not retained and are outside that comparison. This audit reads no
references and performs no fitting. It verifies outputs, but does not establish
the warnings' cause. The new selector fits and inference use Torch; frozen natural
baseline arrays are reused exactly. No nonfinite prediction is admitted by either
audit. The [local validation receipt](../../benchmarks/voice-prosody/selector-local-validation.json)
binds the checked sources and aggregate evidence.

The [independent output audit](../../benchmarks/voice-prosody/selector-factorial-audit.json)
reconstructs 86,360 numeric comparisons with zero differences using the existing
object-based scorer. It binds 2,025 artifacts, all 12 checkpoints and all 14
natural methods across 16 cells, verifies unchanged voicing on all 255,532
acquired development rows per arm, and reconstructs all 14 synthetic cases.
The audit confirms the natural gate failure and stationary gate pass. It checks
saved outputs and tensor identities; it does not independently rerun every new
checkpoint or establish an OS-level reference-access log.
