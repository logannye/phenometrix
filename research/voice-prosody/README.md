# Causal voice estimation research

**Production remains DSP.** The original learned candidate selectors and the
single synthetic-augmentation follow-up retain their failed known-truth gates.
The subsequent no-fit original-MLP-voicing/raw-YIN hybrid passes its declared
synthetic checks and now the frozen natural quality component on four previously
reserved speakers: joint error 9.6747%, versus DSP 11.8482%. Its own candidate
MLP remains better on natural speech at 9.1158%, so the pitch repair has a
measurable tradeoff. A separately versioned browser numerical repair passes the
unchanged bounded parity checks. The complete local runtime series and 30 min
hybrid soak also pass: complete-block p99 6.1 ms, zero 20 ms cadence misses and
maximum certified software render age 54.4 ms in the soak.
See [HYBRID_VALIDATION_RESULTS.md](HYBRID_VALIDATION_RESULTS.md) for the preceding
hybrid stage and [HYBRID_RESULTS.md](HYBRID_RESULTS.md) for the earlier reused-development
follow-up. Neither stage establishes calibrated correctness or clinical use.

Read [RESULTS.md](RESULTS.md) for the final comparison, [PROTOCOL.md](PROTOCOL.md)
for the declared experiment, and [NOTEBOOK.md](NOTEBOOK.md) for decisions and
failure records. [DATA_AUDIT.md](DATA_AUDIT.md), [ALIGNMENT_AUDIT.md](ALIGNMENT_AUDIT.md),
[SUPPORT_DIAGNOSTIC.md](SUPPORT_DIAGNOSTIC.md), and
[COMPARATOR_AUDIT.md](COMPARATOR_AUDIT.md) describe the reference and timing limits.
The [timing follow-up](TIMING_FOLLOWUP.md) documents the independently probed
clock behavior and same-main-clock freshness gate.
This public-corpus research path is separate from encounter media and the optional
WavLM service. Software, measurement, and clinical validation are distinct.

## Exposure and acoustic-evidence feasibility

The latest [bounded study](EVIDENCE_RESULTS.md) stopped at its TRAIN-support
prerequisite, before fitting. The fixed 512-case bank supplied 3,660 added-only
correct-candidate rows across 32 configurations, but only **4 distinct carriers
against a floor of 6**. The counterexample stratum passed. All gates remain
unchanged; training exposure versus candidate-conditioned acoustic evidence is
still an untested comparison.

The shipped research harness includes a trailing-window periodicity feature in
TypeScript and NumPy, a zero-initialized residual scorer, frozen study/sampling
contracts, independent support and terminal audits, and a local HTML failure
report. These components passed the full 938-test research suite; the required
repository structure check passed. Neither model promotion nor new browser
qualification follows from these engineering checks.

The [fixed plan](EVIDENCE_EXPERIMENT_PLAN.md),
[aggregate summary](../../benchmarks/voice-prosody/evidence-study-summary.json),
[feasibility audit](../../benchmarks/voice-prosody/evidence-feasibility-independent-audit.json),
[terminal audit](../../benchmarks/voice-prosody/evidence-study-independent-audit.json),
and [local validation](../../benchmarks/voice-prosody/evidence-local-validation.json)
retain the no-go and repair lineage. All 74 consumed cases remain historical
regressions, without rerunning them; the twelve reserved fresh configurations
remain ungenerated. The overnight authorization has ended. See the results for
commands that inspect saved artifacts without launching an experiment.

## Additive candidate pool and matched retraining

The [six-fit study](EXPANDED_POOL_RESULTS.md) preserves every original raw
candidate and adds at most one already-admitted full-window peak. The same
3,009-parameter shared scorer is retrained on six- and seven-candidate pools
across three seeds, with matched schedules and original-six TRAIN scaling.
Historical weights evaluated on both pools separate acquisition from retraining.

Raw development coverage improves from 88.1304% to 88.2399% without eviction,
but expanded-pool joint error is 8.4589%, versus 8.4020% for the matched control.
517.3/641.9 Hz remain at 0/112 correct post-change targets despite full coverage.
Fresh 633.1 Hz phases score 24/112 and 23/112 for every selector; the other ten
new configurations pass. All 62 prior cases remain, and all 74 cases are reported.
Unchanged natural and per-case gates reject advancement.

The [fixed plan](EXPANDED_POOL_PLAN.md) explicitly permits consumed development
references for prerequisite pass/stop checks before fitting; final scores reload
the same references after predictions freeze. It does not claim globally blinded
or independent development evaluation. Six final checkpoints precede generation
of the twelve prospective configurations. The unchanged synthetic TRAIN bank
already has complete original-pool coverage, limiting what the experiment teaches
about selecting cases where only the added candidate is correct.

With the existing pinned environment and historical local artifacts, an authorized
reproduction uses a NEW ignored path:

```sh
.venv/bin/python -m phenometrix_prosody.expanded_pool_experiment --output runs/expanded-pool-reproduction-v1
```

The runner refuses overwrites, caps the comparison at six fits and preserves
source, target, schedule, checkpoint and prediction identities. The recorded run
completed locally in 158.90 seconds, including 61.69 seconds of MPS fitting.
See the [aggregate evidence](../../benchmarks/voice-prosody/expanded-pool-summary.json),
[independent score/training audit](../../benchmarks/voice-prosody/expanded-pool-independent-audit.json),
[target-free equation replay](../../benchmarks/voice-prosody/expanded-pool-inference-audit.json)
and [local validation receipt](../../benchmarks/voice-prosody/expanded-pool-local-validation.json).
No additional experiment, production promotion or new browser qualification
follows from this no-go. These fresh cases are now consumed regressions.

## Candidate-coverage follow-up

The [bounded no-fit study](CANDIDATE_COVERAGE_RESULTS.md) restores correct raw and
grouped candidates at 517.3/641.9 Hz from 0/112 to 112/112 in both phases. The
same three frozen selectors still choose incorrectly on all those post-change
targets. Natural development error rises from 8.4018% to 8.4571%; coverage losses
outweigh gains. All twelve fresh configurations pass under both policies, and
all fifty old cases remain recorded. The unchanged gates reject advancement.

The [prospective plan](CANDIDATE_COVERAGE_PLAN.md) fixes one replacement policy:
retain the shortest already-admitted full-window autocorrelation peak in the
existing six-candidate budget. No thresholds, voicing, weights, scaling, timing
or grouping rules change. The whole experiment runs locally on CPU with zero
fits; previous browser timings do not qualify this offline path. See the
[aggregate evidence](../../benchmarks/voice-prosody/candidate-coverage-summary.json),
[independent scoring](../../benchmarks/voice-prosody/candidate-coverage-independent-audit.json),
[checkpoint replay](../../benchmarks/voice-prosody/candidate-coverage-inference-audit.json)
and [local validation](../../benchmarks/voice-prosody/candidate-coverage-local-validation.json).

With the pinned environment and historical local artifacts present, a separately
authorized reproduction from this directory uses a new ignored output path:

```sh
.venv/bin/python -m phenometrix_prosody.candidate_coverage_experiment --output runs/candidate-coverage-reproduction-v1
```

No dependencies are installed, old outputs overwritten or reserved speech read.
Fresh configurations from this completed run are now consumed evidence; a new
study needs its own prospective specification. The portable consumed-input
`diagnose_candidate_coverage.ts` accepts `--output` in an existing ignored run
directory and is run with `node --import tsx` from `apps/capture-web`.

## Harmonic-balance training augmentation

The [fixed plan](AUGMENTATION_EXPERIMENT_PLAN.md) compares natural-only training,
stationary synthetic augmentation and predesignated harmonic-balance augmentation
across three seeds. The model, natural-only scaling, original voicing, timing,
natural minibatches and 396 optimizer updates stay fixed. Each augmented update
adds 0.25 times a 512-row synthetic loss. Ninety-six paired cases per arm cover
24 frequencies, two phases and two gains; exact 337.1 and 354.5 Hz evaluation
cases are excluded from training. All old gates remain required, with twelve
new frequency/phase cases generated only after every checkpoint freezes.

From this directory, with the existing pinned environment and prerequisite
artifacts present, a new authorized reproduction uses a NEW ignored output path:

```sh
.venv/bin/python -m phenometrix_prosody.augmentation_experiment --output runs/harmonic-augmentation-reproduction-v1
```

The runner refuses overwrites and binds source, input, schedule, target,
checkpoint and prediction identities. It does not install dependencies or use
remote computation. The [results](AUGMENTATION_RESULTS.md),
[aggregate evidence](../../benchmarks/voice-prosody/augmentation-summary.json)
and [local validation receipt](../../benchmarks/voice-prosody/augmentation-local-validation.json)
record the measured outcome and independent audits. Both augmentation arms repair
337.1/354.5 Hz at 112/112 post-change targets in both phases and every seed, but
the candidate remains a no-go: its natural gain over original MLP is too small,
and fresh 517.3/641.9 Hz cases have no correct post-change candidates. The local
suite passes 371 research tests and `npm run check`; three historical NumPy
warnings remain documented. Synthetic-family performance
and consumed natural development remain distinct from external speech and
browser qualification.

## Subharmonic-ranking objective follow-up

The [nine-fit objective study](SUBHARMONIC_RESULTS.md) partially repairs the
337.1 Hz seed-specific failure to 56/112 and 58/112 correct post-change targets,
but both remain below 95%. Alias-margin raises development error to 8.5126%
from the exactly reproduced control's 8.4492%. New 354.5 Hz phase cases fail
for every new/prior selector despite full candidate coverage and voiced output.
The designated candidate remains a no-go under the unchanged advancement criteria.

The [frozen plan](SUBHARMONIC_EXPERIMENT_PLAN.md) retains architecture, natural
training, scaling, voiced decisions, timing, targets and update count. It adds
a symmetric TRAIN-only harmonic-negative margin without inference-time rules.
All nine checkpoints precede new challenge generation and all predictions
precede development scoring. The old twelve challenge cases remain required
consumed regressions. No extra fit or outcome-based arm selection followed.

With the existing caches, the recorded experiment command was:

```sh
.venv/bin/python -m phenometrix_prosody.subharmonic_experiment --output runs/subharmonic-objective-v1
```

This run completed normally. Existing outputs are never overwritten; any
reproduction requires a fresh ignored run path. Raw media, references, features,
weights and natural frame predictions remain local. The source-bound
[independent score audit](../../benchmarks/voice-prosody/subharmonic-independent-audit.json),
[checkpoint replay](../../benchmarks/voice-prosody/subharmonic-inference-audit.json),
[coverage/state diagnostic](../../benchmarks/voice-prosody/subharmonic-challenge-diagnostic.json),
[refitted-control comparison](../../benchmarks/voice-prosody/subharmonic-refitted-control-audit.json),
and [local validation](../../benchmarks/voice-prosody/subharmonic-local-validation.json)
verify the reported outcome without changing the no-go.

The final local suite passes **306 research tests** and `npm run check`; three
historical NumPy warnings remain explicit. These checks verify software behavior
and do not override the failed advancement gates.

## Harmonic-context follow-up

The [bounded harmonic-context experiment](HARMONIC_RESULTS.md) is complete.
Relative candidate context repairs the old 300 Hz post-change failure in every
seed, but the predesignated model's 8.4492% development error misses the unchanged
natural gate. A prospective 12-case frequency/phase challenge exposes another
threefold-period choice at 337.1 Hz: seed 20261023 gets 0/112 correct in both
post-change phase cases. The candidate remains a no-go.

The [plan](HARMONIC_EXPERIMENT_PLAN.md) fixes twelve local MPS fits, identical
allocated parameter counts, natural-only training, unchanged voicing and support,
and stricter per-seed/per-case harmonic gates. All final checkpoints precede new
challenge generation; all predictions precede development-reference scoring.
No new fit or scientific setting was tried after the result.

The original run retains a NumPy-boolean JSON serialization failure after every
prediction was frozen. Its sources are kept exact for provenance. The separate
recovery command reads those predictions, applies unchanged scoring, and converts
NumPy scalars only for JSON output. It never fits, infers or generates waveforms.
With the existing local caches, the recorded sequence was:

```sh
.venv/bin/python -m phenometrix_prosody.harmonic_selector_experiment --output runs/harmonic-context-v1
# The preserved runner exits at the documented post-freeze serialization error.
.venv/bin/python -m phenometrix_prosody.recover_harmonic_scores --run runs/harmonic-context-v1 --output runs/harmonic-context-score-recovery-v1
```

Existing run paths are never overwritten. Reproductions require fresh run paths;
the diagnostic/evaluation data used here must not be relabeled independent.
See [aggregate results](../../benchmarks/voice-prosody/harmonic-context-summary.json),
[old-failure diagnosis](../../benchmarks/voice-prosody/harmonic-ranking-diagnosis.json),
[challenge bottlenecks](../../benchmarks/voice-prosody/harmonic-challenge-diagnostic.json),
[independent scoring](../../benchmarks/voice-prosody/harmonic-independent-audit.json),
[checkpoint replay](../../benchmarks/voice-prosody/harmonic-inference-audit.json),
and the [local validation receipt](../../benchmarks/voice-prosody/harmonic-local-validation.json).
The final local suite passes 244 research tests and `npm run check`; software
validation does not change the model's failed advancement gates.

## Candidate representation and loss: step 1

The [bounded factorial study](SELECTOR_RESULTS.md) is now complete. Six-candidate
headroom is 1.3144 points on reused development. Shared scoring/grouping repairs
the stationary 300 Hz regression in every seed, but the predesignated combined
model's 8.4825% mean joint error improves only 0.1524 points over the original MLP,
below the fixed 0.5-point requirement. A channel-change failure in one seed also
remains. No arm is promoted or substituted after observing outcomes.

The [plan](SELECTOR_EXPERIMENT_PLAN.md) fixes four arms, three seeds, natural-only
training, frozen original voicing and exact support/target semantics. All fitting
ran locally on MPS. The commands below reproduce this completed diagnostic and
refuse to overwrite prior evidence. They do not authorize another adaptive search.

```sh
.venv/bin/python -m phenometrix_prosody.selector_headroom --output runs/selector-headroom-v1 --public-output ../../benchmarks/voice-prosody/selector-headroom.json
.venv/bin/python -m phenometrix_prosody.selector_experiment grouping --output runs/selector-grouping-v2
.venv/bin/python -m phenometrix_prosody.selector_experiment prerequisites --headroom ../../benchmarks/voice-prosody/selector-headroom.json --grouping runs/selector-grouping-v2/grouping.json --output runs/selector-prerequisites-v1
.venv/bin/python -m phenometrix_prosody.selector_experiment run --prerequisites runs/selector-prerequisites-v1/prerequisites.json --output runs/selector-factorial-v1
.venv/bin/python -m phenometrix_prosody.audit_selector_experiment --run runs/selector-factorial-v1 --prerequisites runs/selector-prerequisites-v1/prerequisites.json --grouping-failure runs/selector-grouping-v1/failure.json --output ../../benchmarks/voice-prosody/selector-factorial-audit.json
```

The public [headroom](../../benchmarks/voice-prosody/selector-headroom.json) and
[factorial summary](../../benchmarks/voice-prosody/selector-step1-summary.json)
contain counts, gates, every seed and input/source identities. Raw predictions,
reference arrays, training caches and weights remain ignored. The rejected first
grouping attempt was repeated against stable sources before any fit; its failure
record is retained; the audit command binds that historical local record. Development
and old synthetic fixtures remain consumed. The [independent score audit](../../benchmarks/voice-prosody/selector-factorial-audit.json)
reconstructs 86,360 numeric comparisons exactly. The [Torch parity receipt](../../benchmarks/voice-prosody/selector-torch-parity.json)
checks every frozen training prediction; its [runnable source](../../benchmarks/voice-prosody/source-snapshots/selector-full-torch-parity-v1.py)
requires the existing local caches and a fresh receipt output path. Full test
counts, retained numerical warnings and checked identities appear in the
[local validation record](../../benchmarks/voice-prosody/selector-local-validation.json).

## Environment and artifact handling

Use the locked Python 3.11 environment and prebuilt wheels; `--no-build` refuses source builds. The recorded fits use
local PyTorch MPS; `train` refuses to silently fall back to another device. The
browser uses the repository's pinned pnpm dependencies. No Rust build is needed
for these research commands.

From the repository root:

```sh
uv sync --project research/voice-prosody --locked --no-build
cd research/voice-prosody
.venv/bin/python -m pytest -q -p no:cacheprovider tests
```

Unless stated otherwise, subsequent commands run from `research/voice-prosody`.
They document the recorded workflow, not an instruction to repeat fitting after
the final no-go. Outputs require fresh directories and refuse to overwrite runs.
Preserve existing evidence; use a fresh workspace or new output names for a new
reproduction. Run manifests record input, source, dependency, checkpoint and
output identities. MPS reruns are not promised to produce byte-identical weights.

Corpus audio, reference trajectories, generated caches, checkpoints and run
artifacts remain in ignored local folders. Dataset ODbL/DbCL notices are independent
of this repository's MIT license. Raw prediction JSON is losslessly compressed
with deterministic gzip; `evaluate.read_predictions(path)` reads it. Raw voicing
probability is never presented as calibrated pitch-correctness probability.

## Acquisition and original failed endpoint screen

Inventory reads public metadata. Acoustic test content remained sealed during
this historical experiment; the separately frozen validation stage below later
accesses its declared reserved subset. The first 24-record training pilot precedes
the larger acquisition:

```sh
.venv/bin/python -m phenometrix_prosody.data inventory
.venv/bin/python -m phenometrix_prosody.data acquire --split train --per-speaker 2 --roles mic,ref
.venv/bin/python -m phenometrix_prosody.pilot --data-root data/ptdb-tug --feature-version v1 --output runs/train-pilot
.venv/bin/python -m phenometrix_prosody.pilot --data-root data/ptdb-tug --feature-version v2 --selection-manifest runs/train-pilot/manifest.json --output runs/train-pilot-v2
.venv/bin/python -m phenometrix_prosody.natural_error_stratification --pilot runs/train-pilot-v2 --data-root data/ptdb-tug
```

The selection manifest freezes the original 24 microphone/reference identities;
a later acquisition must not silently expand this diagnostic. V1 candidate
coverage was 52.46%; the single authorized K6 revision reached 64.57%, below the
75% prerequisite. These endpoint failures remain failures.

The original analytic source-support experiment uses eleven fixed signals:

```sh
(cd ../../apps/capture-web && node --import tsx ../../research/voice-prosody/support_diagnostic.ts --output ../../research/voice-prosody/runs/support-diagnostic-v1)
.venv/bin/python -m phenometrix_prosody.support_diagnostic --run runs/support-diagnostic-v1
```

The synthetic experiment establishes support-induced lag on exact known truth.
Its midpoint/window-mean analyses do not retrospectively change endpoint scores.
Reference-derived stratification is scorer-only and never enters a feature.

## Separate declared 20 ms measurement-delay experiment

A feature supported on `[t-20 ms, t+20 ms)` reports target `t` and cannot become
available before `t+20 ms`. The feature cache retains acquisition endpoint `e`;
training and evaluation explicitly map it to `t=e-960` at 48 kHz. This is streaming
with 20 ms future support relative to the reported target, not zero lookahead.
All methods use the common scoring mask `t>=64 ms` and `t+32 ms<=N`, exact nominal
`16+10*k ms` reference grid, and explicit missing-output accounting.

```sh
.venv/bin/python -m phenometrix_prosody.data acquire --split train --per-speaker 20 --roles mic,ref --max-bytes 2147483648
.venv/bin/python -m phenometrix_prosody.data acquire --split development --per-speaker 20 --roles mic,ref --max-bytes 2147483648
.venv/bin/python -m phenometrix_prosody.features --data data/ptdb-tug --output runs/delayed-train-features --split train
.venv/bin/python -m phenometrix_prosody.features --data data/ptdb-tug --output runs/delayed-development-features --split development
.venv/bin/python -m phenometrix_prosody.train --features runs/delayed-train-features --data data/ptdb-tug --output runs/delayed-gru-seed20261022 --kind causal-gru
.venv/bin/python -m phenometrix_prosody.train --features runs/delayed-train-features --data data/ptdb-tug --output runs/delayed-mlp-seed20261022 --kind framewise-mlp
.venv/bin/python -m phenometrix_prosody.evaluate --features runs/delayed-development-features --training-features runs/delayed-train-features --data data/ptdb-tug --gru runs/delayed-gru-seed20261022 --mlp runs/delayed-mlp-seed20261022 --output runs/delayed-development-evaluation
.venv/bin/python -m phenometrix_prosody.evaluate_synthetic --support-run runs/support-diagnostic-v1 --training-features runs/delayed-train-features --gru runs/delayed-gru-seed20261022 --mlp runs/delayed-mlp-seed20261022 --output runs/delayed-known-truth
```

Caches contain input-only features and baseline predictions, across clean,
10 dB noise, first-difference and amplitude-clipping conditions. They are finished
before references are loaded. Model scaling uses training features only. Both
models use the fixed final epoch, explicit state/reset semantics and the same
57-feature/K6 candidate contract. Evaluators use independent NumPy equations and
freeze every prediction before loading scorer references.

`selection-decision.json` records the pre-development selection clarification.
The GRU missed the required one-percentage-point improvement over the MLP. The
MLP passed the natural-development quality components but failed constant 300 Hz
and harmonic-balance cases. A quality component is never a final keep decision.

## Pretrained comparison with a different lookahead

The pinned tiny CREPE comparator has centered 64 ms support and **32 ms future
support**, compared with 20 ms for the candidate experiment. Its scores are
reported separately and do not establish equivalent-latency performance.

```sh
.venv/bin/python -m phenometrix_prosody.comparator_cache --features runs/delayed-development-features --data data/ptdb-tug --output runs/delayed-development-comparator --device mps
.venv/bin/python -m phenometrix_prosody.score_comparator --features runs/delayed-development-features --predictions runs/delayed-development-comparator --data data/ptdb-tug --output runs/delayed-development-comparator-scores
```

For the earlier 24-record diagnostic, the recorded centered and endpoint-anchored
comparisons and descriptive delay analysis are reproducible with:

```sh
.venv/bin/python -m phenometrix_prosody.comparator_pilot --data data/ptdb-tug --selection runs/train-pilot/manifest.json --output runs/comparator-pilot --anchor centered
.venv/bin/python -m phenometrix_prosody.comparator_pilot --data data/ptdb-tug --selection runs/train-pilot/manifest.json --output runs/comparator-end-anchored-pilot --anchor end_anchored
.venv/bin/python -m phenometrix_prosody.delay_diagnostic --data data/ptdb-tug --original runs/train-pilot --revised runs/train-pilot-v2 --centered runs/comparator-pilot --end-anchored runs/comparator-end-anchored-pilot --output runs/delay-diagnostic
```

## Single synthetic-augmentation follow-up

Both train and challenge parameter plans were frozen before generation. The
curriculum has 128 distinct 2.5-second training sequences: stationary harmonics,
steps, glides, harmonic-balance changes and aperiodic noise. Training repeats each
sequence four times to set its weight, not to invent independent observations.
The 32-case challenge uses separate seeds, complete parameters, trajectories,
stationary frequencies and harmonic mixtures. A continuous training glide can
cross a challenge frequency; disjointness concerns full source configurations.

The new decision requires every original stationary and postchange observation
case to reach 95% success, challenge equal-scenario voiced success of at least
90%, every voiced case at least 80%, and false-positive rate no greater than 5%
in **each** unvoiced case as well as pooled. Natural-development and transition
requirements remain in force. Challenge generation checks the completed augmented
checkpoint and its frozen training-curriculum identity.

```sh
.venv/bin/python -m phenometrix_prosody.synthetic_curriculum freeze --output runs/synthetic-curriculum-plan
.venv/bin/python -m phenometrix_prosody.synthetic_curriculum generate --plan runs/synthetic-curriculum-plan --split train --output runs/synthetic-train-curriculum
.venv/bin/python -m phenometrix_prosody.train --features runs/delayed-train-features --data data/ptdb-tug --output runs/delayed-augmented-mlp-seed20261022 --kind framewise-mlp --synthetic-curriculum runs/synthetic-train-curriculum
.venv/bin/python -m phenometrix_prosody.evaluate --features runs/delayed-development-features --training-features runs/delayed-train-features --data data/ptdb-tug --gru runs/delayed-gru-seed20261022 --mlp runs/delayed-augmented-mlp-seed20261022 --output runs/delayed-augmented-development-evaluation
.venv/bin/python -m phenometrix_prosody.evaluate_synthetic --support-run runs/support-diagnostic-v1 --training-features runs/delayed-train-features --gru runs/delayed-gru-seed20261022 --mlp runs/delayed-augmented-mlp-seed20261022 --output runs/delayed-augmented-known-truth
.venv/bin/python -m phenometrix_prosody.synthetic_curriculum generate --plan runs/synthetic-curriculum-plan --split challenge --output runs/synthetic-challenge-curriculum --checkpoint-manifest runs/delayed-augmented-mlp-seed20261022/manifest.json
.venv/bin/python -m phenometrix_prosody.evaluate_curriculum --curriculum runs/synthetic-challenge-curriculum --training-features runs/delayed-train-features --augmented runs/delayed-augmented-mlp-seed20261022 --original runs/delayed-mlp-seed20261022 --output runs/delayed-augmented-challenge
```

Before the last command, preserve the recorded pre-score
`runs/synthetic-curriculum-plan/challenge-evaluation-decision.json`, which declares
the per-case false-positive gate. Its source is shown in the notebook. A fresh
reproduction must create that declaration before scoring, not copy a later result.
The challenge scorer reconstructs classical baselines from frozen parameters,
verifies PCM digests and bit-identical cached features, then freezes all five
methods before reading analytic references. No synthetic record is represented
as a PTDB participant or recording.

The augmented MLP improved challenge voiced success from 72.95% to 88.07% and
fixed the original 300 Hz harmonic-balance case. It still failed the stationary 300 Hz
three-harmonic case and four challenge cases. The fixed causal YIN comparator reached 99.82%
on this synthetic challenge, while showing weaker natural-development joint
error than the learned models. Neither result alone establishes deployment or
clinical validity. The accepted stopping rule is **no learned promotion**.

## No-fit subharmonic diagnosis and hybrid follow-up

The original failures remain recorded. The bounded [diagnostic](SUBHARMONIC_SWEEP.md)
uses frozen checkpoints, a physical frequency sweep and declared counterfactuals.
The separately frozen [hybrid plan](HYBRID_FOLLOWUP_PLAN.md) combines each existing
MLP voicing head with raw YIN pitch, with original primary and augmented secondary.
It changes neither weights nor v2 features. Both variants pass the declared fresh
synthetic, transition and reused natural-development quality components, while
both worsen natural error relative to their own candidate MLP. The predeclared
preference retains the original as an offline research candidate; the production
path stays DSP. Read [HYBRID_RESULTS.md](HYBRID_RESULTS.md) for all tradeoffs and
remaining qualification limits.

These commands require the exact existing model, feature and evaluation artifacts
from the preceding experiments. They do not run training. Freeze the complete
plan before either new evaluation. Use fresh output folders; the recorded names
below already exist and must not be overwritten:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_alias_diagnostic.py tests/test_hybrid_followup.py tests/test_hybrid_natural.py
.venv/bin/python -m phenometrix_prosody.alias_diagnostic --output runs/subharmonic-diagnostic-v1
.venv/bin/python -m phenometrix_prosody.hybrid_followup freeze --output runs/hybrid-followup-plan-v1
.venv/bin/python -m phenometrix_prosody.hybrid_followup synthetic --plan runs/hybrid-followup-plan-v1 --output runs/hybrid-synthetic-v1
.venv/bin/python -m phenometrix_prosody.hybrid_natural --plan runs/hybrid-followup-plan-v1 --features runs/delayed-development-features --original-evaluation runs/delayed-development-evaluation --augmented-evaluation runs/delayed-augmented-development-evaluation --data data/ptdb-tug --output runs/hybrid-natural-v1
.venv/bin/python -m phenometrix_prosody.audit_hybrid_synthetic --run runs/hybrid-synthetic-v1 --plan runs/hybrid-followup-plan-v1 --output runs/hybrid-synthetic-v1/independent-audit.json
```

`read_frozen_plan` verifies the 19 source bindings, checkpoint weights/manifests,
natural-cache identities and exact synthetic definitions. Synthetic and natural
runners freeze their respective complete input-only predictions before loading
analytic or corpus references. The natural runner can separate these phases with
`--phase predict` and `--phase score`; references are needed only for scoring.
The independent synthetic audit reconstructs counts and signed delays from frozen
predictions without rerunning inference. All fresh synthetic case results remain in the ignored metrics artifact. Compact
synthetic aggregates, source identities and all 16 natural speaker/condition cells
appear in the [public summary](../../benchmarks/voice-prosody/hybrid-summary.json). Reused natural
development is exploratory; the sealed test remained closed during that stage. Neither raw voicing
probability nor passing synthetic cases establish calibrated correctness.

## Independent reserved-speaker and browser validation

The [fixed validation plan](HYBRID_VALIDATION_PLAN.md) admits only the original
hybrid and the exact first 40 metadata-ordered prompts for each of F05/M05/F06/M06.
This is 160 sources and 640 dependent condition versions, with no new fitting,
threshold selection or secondary-model switch. The pre-access audit checks prompt
separation and historical acquisition metadata. Actual source, model, configuration,
inventory and protocol bytes are frozen before MIC-only acquisition. Every
prediction is completed and hashed before REF acquisition or reference audit.
Generic historical training/development runners still reject the sealed split.

The commands below reproduce the phase order in a **fresh reproduction workspace
with the recorded trained artifacts and an unacquired reserved split**. The local
recorded folders already exist; the pre-access audit must refuse to relabel them
as unseen. Do not delete evidence or rerun fitting to satisfy that guard. Commands
are shown from `research/voice-prosody` unless a directory change is explicit:

```sh
.venv/bin/python -m phenometrix_prosody.sealed_metadata_audit --output runs/sealed-preaccess-metadata-audit-v1.json
.venv/bin/python -m phenometrix_prosody.hybrid_validation freeze --output runs/hybrid-validation-freeze-v1 --preaccess-audit runs/sealed-preaccess-metadata-audit-v1.json
.venv/bin/python -m phenometrix_prosody.hybrid_validation acquire-mic --freeze runs/hybrid-validation-freeze-v1/freeze.json --output runs/hybrid-validation-v1
cd ../..
mkdir -p .local
cp research/voice-prosody/run_frozen_hybrid_stage.py .local/run-hybrid-natural-stage.py
research/voice-prosody/.venv/bin/python .local/run-hybrid-natural-stage.py predict
cd research/voice-prosody
.venv/bin/python -m phenometrix_prosody.hybrid_parity --output runs/hybrid-browser-parity-v2
.venv/bin/python -m phenometrix_prosody.hybrid_reserved_parity --freeze runs/hybrid-validation-freeze-v1/freeze.json --output runs/hybrid-reserved-browser-parity-v1
```

The published phase wrapper is an exact copy of the recorded wrapper. It uses
fixed phase paths, records command start/end, exit status and log hashes, and
refuses to overwrite execution evidence. The parity helpers never open corpus
references. They use 14 analytic sources, one full source per development speaker,
and one preselected full source per reserved speaker under all four conditions.
They enforce a 20 s source bound, verify PCM hashes and check all frames, including
unvoiced raw-YIN estimates. They do not select windows by label or model outcome.

Prepare the original frozen ONNX assets with the explicit hybrid option, then
run the current browser parity implementation against the unchanged fixtures:

```sh
cd ../..
node apps/capture-web/scripts/prepare-voice-mlp-research.mjs --hybrid ../../research/voice-prosody/runs/delayed-mlp-export
cd apps/capture-web
VOICE_RESEARCH_MODEL_CONFIG="$PWD/public/.local/voice-mlp/hybrid-config.json" \
VOICE_HYBRID_PARITY_MANIFEST="$PWD/../../research/voice-prosody/runs/hybrid-browser-parity-v2/manifest.json" \
VOICE_RESEARCH_OUTPUT_DIR="$PWD/../../research/voice-prosody/runs/hybrid-browser-reproduction-development" \
pnpm exec playwright test --config playwright.research.config.ts --grep 'frozen hybrid' --workers=1
VOICE_RESEARCH_MODEL_CONFIG="$PWD/public/.local/voice-mlp/hybrid-config.json" \
VOICE_HYBRID_PARITY_MANIFEST="$PWD/../../research/voice-prosody/runs/hybrid-reserved-browser-parity-v1/manifest.json" \
VOICE_RESEARCH_OUTPUT_DIR="$PWD/../../research/voice-prosody/runs/hybrid-browser-reproduction-reserved" \
pnpm exec playwright test --config playwright.research.config.ts --grep 'bounded PCM hook' --workers=1
cd ../..
```

The first mixed-precision browser implementation failed one reserved-input raw-Hz
comparison. The [single numerical-repair plan](YIN_NUMERICAL_REPAIR_PLAN.md) and
failed evidence are preserved. Consistent float64 internal arithmetic passes
4,420 analytic/development and 10,812 reserved-input frames under the original
tolerance; the repaired reserved check is a regression after a diagnosed input
failure, not an untouched implementation holdout. Python natural predictions,
model weights and scorer targets remain unchanged. No tie epsilon, unvoiced
exemption or tolerance change was used.

Only after complete prediction freeze and the declared browser check, launch
REF acquisition and scoring serially from the repository root:

```sh
research/voice-prosody/.venv/bin/python .local/run-hybrid-natural-stage.py acquire-ref
research/voice-prosody/.venv/bin/python .local/run-hybrid-natural-stage.py score
cd research/voice-prosody
.venv/bin/python -m phenometrix_prosody.audit_hybrid_validation \
  --run runs/hybrid-validation-v1 --freeze runs/hybrid-validation-freeze-v1/freeze.json \
  --data data/ptdb-tug --execution runs/hybrid-validation-execution-v1 \
  --freeze-sha256 c5c4288cc04c0d7fb46154977627158336e2b1a5b8b496f8700c7a96dd25c67d \
  --allow-sealed-reference-audit --output runs/hybrid-validation-v1/independent-audit.json
```

The shown hash identifies the recorded freeze. A fresh reproduction must record
its own freeze hash **before acquisition** and supply that identity to its audit;
it cannot inherit the historical hash merely by reusing filenames. The independent
audit reconstructs 3,200 recording/method count comparisons and verifies exact
macros, pair effects and paired-prompt bootstrap intervals. Intervals condition on
four speakers and two pairs, not a participant population. Full count/stratum
results and limits are in [HYBRID_VALIDATION_RESULTS.md](HYBRID_VALIDATION_RESULTS.md)
and the [public summary](../../benchmarks/voice-prosody/hybrid-validation-summary.json).

The sequential runtime driver freezes measured source/configuration identities,
runs DSP/hybrid/hybrid/DSP for 60 s each and a 30 min hybrid soak, and preserves
failures. It does not run training or hosted computation. From this directory:

```sh
.venv/bin/python run_hybrid_browser.py --output runs/hybrid-runtime-reproduction \
  --config ../../apps/capture-web/public/.local/voice-mlp/hybrid-config.json
```

Do not run other compute workloads during that series. After completion, run
`phenometrix_prosody.audit_browser` and `phenometrix_prosody.audit_clock` on each
run, and `phenometrix_prosody.audit_hybrid_browser` on each hybrid run with the
same pinned configuration hash. The public summary requires matching raw/result
identities and all applicable audits. All five recorded runs and their applicable audits passed. The 30 min hybrid
soak has one block above 10 ms (maximum 10.5 ms), zero 20 ms cadence misses,
eight completed frames withheld from delivery and 859 unavailable legacy
cross-clock frame timings. The report preserves these details and the limits
of the same-clock software freshness certificate. Inner replay artifacts still declare
`timingBenchmark: false` and `runPurpose: "sustained-functional-soak"`; the outer
sequence supplies the frozen component evidence, not a controlled hardware
benchmark. The final sample endpoint and extra boundary-bin frame remain in the
audits rather than being truncated to 30 min.

## Export and browser engineering checks

These commands establish arithmetic and interface compatibility, not accuracy:

```sh
.venv/bin/python -m phenometrix_prosody.export --spike --feature-version v2 --output runs/spike-v2 --check-mps
.venv/bin/python -m phenometrix_prosody.export --checkpoint runs/delayed-gru-seed20261022 --output runs/delayed-gru-export
.venv/bin/python -m phenometrix_prosody.export --checkpoint runs/delayed-mlp-seed20261022 --output runs/delayed-mlp-export
cd ../..
node apps/capture-web/scripts/prepare-voice-research.mjs ../../research/voice-prosody/runs/spike-v2
node apps/capture-web/scripts/prepare-voice-mlp-research.mjs ../../research/voice-prosody/runs/delayed-mlp-export
pnpm --filter @phenometrix/capture-web exec playwright test --config playwright.research.config.ts
```

The research page uses a synthetic AudioBufferSource, real worklet/worker transport
and controlled overload; it does not request a microphone. Trained hooks remain
explicit research opt-ins. Graph parity alone is distinct from full-stream parity
or a complete-worker service percentile. Report production replay validation
separately from these offline Python scores.

The preserved DSP numerical identity is `voice-analysis-1.1`. Transport/worker v2
adds exact sample endpoints, separate clock origins and a bounded credit queue.
Legacy `tMs` remains window start. Inconsistent clocks retain raw evidence and
make derived latency unavailable; negative times are not clamped into success.
These historical export/replay checks alone establish no calibration, reserved-speech
accuracy or clinical claim. The separately frozen validation stage above has its
own natural and browser evidence.

## Historical candidate-MLP runtime reproduction

After preparing both research asset sets above, run each replay in a fresh output
folder. From `apps/capture-web`, the counterbalanced order is DSP, trained MLP,
trained MLP, DSP, sequentially. Use the same command with the appropriate method:

```sh
export VOICE_RESEARCH_OUTPUT_DIR="$(mktemp -d "$PWD/../../research/voice-prosody/runs/browser-runtime-XXXXXX")"
VOICE_RESEARCH_METHOD=dsp VOICE_RESEARCH_DURATION_MS=60000 \
VOICE_RESEARCH_OVERLOAD_AT_MS=30000 VOICE_RESEARCH_OVERLOAD_DURATION_MS=300 \
pnpm exec playwright test --config playwright.research.config.ts \
  --grep 'real AudioWorklet/worker replay' --workers=1 --reporter=json \
  > "$VOICE_RESEARCH_OUTPUT_DIR/playwright-report.json"
```

For the sustained DSP replay, choose a fresh output folder and set duration to
`1800000` and overload time to `900000`, keeping the 300 ms stall. The fixture is
a repeated known signal, not additional speech data. Do not run fits, other tests,
or simultaneous replay methods during a measured sequence. Record the source,
artifact, runtime, machine and background-load limits as the final report does.

From the repository root, independently audit any completed raw replay:

```sh
PYTHONPATH=research/voice-prosody research/voice-prosody/.venv/bin/python \
  -m phenometrix_prosody.audit_browser --run /absolute/path/to/completed-run \
  --output /absolute/path/to/completed-run/independent-audit.json
```

Production builds exclude `public/.local` research assets. After building both
capture outputs, verify that exclusion and byte identity of every normal public
asset with `node apps/capture-web/scripts/verify-production-public-assets.mjs`.
The final measurements and their limits are in [STREAMING_RESULTS.md](STREAMING_RESULTS.md).
