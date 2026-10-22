# Engineering notebook

## Scope and final decision

This notebook records the completed 2026-10-22 research sequence. The starting
repository revision was `7e10705a562688f8582ec99c94495f9ba0b645ad`; run manifests
and source snapshots identify each subsequent experiment. No learned estimator
passed all frozen qualification gates. Classical DSP remains the default, and
there is no further model tuning or promotion in this experiment.

The investigation distinguishes transport correctness, estimator arithmetic,
reference-derived accuracy, known-truth diagnostics, browser execution and
clinical validation. All fitted models are research artifacts. Public corpus
and generated source data are not encounter recordings. The test split remains
sealed, and pitch-correctness calibration was not performed.

## Stream and endpoint evidence

The initial source audit found mixed main/worker clock coordinates, endpoint
timestamps tied to render-quantum completion, unbounded PCM pressure and retained
state across gaps. These were source findings, not fabricated live measurements.
The engineering work preserves the classical arithmetic while making timing,
queue capacity, gaps, drops and recovery explicit. Browser engineering results
are reported separately in the final results document.

The first clean training-only pilot contained 24 utterances from 12 speakers,
172.97 seconds total. Classical DSP speaker-macro joint error was 15.729% and
reference-voiced success was 44.582%; the disclosed causal YIN baseline had
17.163% joint error and 48.449% voiced success. K3 candidate coverage was 52.457%.
One predeclared K6 candidate revision reached 64.571%, still below the 75%
prerequisite. Both original endpoint screens failed; no fitting used that failed
endpoint configuration as an accepted result.

Known-truth tones, steps and phase-integrated ramps independently demonstrated
approximately 20 ms of support-induced lag. Moderate ramp coverage alone did not
explain all natural errors. The natural-error strata also identified low-frequency,
rapid-change and boundary failures. Diagnostic midpoint and slope analyses were
scorer-only and did not relabel the original endpoint scores.

## Separately declared 20 ms task

A new measurement contract reports `t=e-20 ms`, with input support ending at `e`
and earliest availability at `e`. The target grid remains `16+10*k ms` and the
common mask is `t>=64 ms`, `t+32 ms<=N`. This uses future support relative to the
reported target; it is not zero-lookahead estimation or a fitted reference shift.

The expanded train cache has 240 source recordings and four conditions, 960
sequences. Unique clean source duration is 1,700.56 seconds; condition-expanded
duration is 6,802.24 seconds. Natural development uses four separate speakers
and the same four conditions. Scores pool recordings within each speaker-condition
cell and then weight all 16 cells equally.

The fixed GRU32 and framewise MLP32 share the same 57 inputs/K6 candidates,
projection32, train-only scaling and seed 20261022. Fits use Adam 0.001, 12 final
epochs, recording batch64, contiguous 128-frame chunks, gradient norm clip1 and
zero state at recording boundaries. Candidate loss applies only to covered
voiced frames; uncovered voiced frames retain positive voicing supervision.
Prediction uses no reference history. Pure NumPy inference independently checks
the trained equations. Raw predictions are frozen before reference files open.

| Method | Development joint error | Reference-voiced success |
|---|---:|---:|
| Delayed classical DSP | 10.607% | 62.453% |
| Delayed causal YIN | 11.021% | 64.880% |
| Fixed continuity | 11.598% | 69.393% |
| GRU32 | 7.953% | 76.379% |
| MLP32 | 8.635% | 75.406% |
| Single augmented MLP32 follow-up | 8.262% | 77.034% |

The initial GRU gained only 0.682 percentage points over the initial MLP, below
the frozen one-point margin. The MLP passed the independent natural-development
quality components. It nevertheless selected approximately 150 Hz for all 151
scored constant-300-Hz frames, despite a candidate near 300 Hz being present.
The GRU mostly selected approximately 100 Hz. Both passed the step/glide lag
component, demonstrating why lag checks alone cannot establish pitch validity.
The centered pretrained comparator has a different 32 ms lookahead and is kept
separate in the final results.

## One augmented configuration, frozen before fitting

A final same-family MLP configuration added 128 unique 2.5-second analytic training
sources, 320 seconds total: 48 stationary, 24 glide, 24 step, 24 harmonic-balance
and 8 aperiodic cases. Each was repeated four times in scaling and training,
approximately 15.8% of duration-weighted input after repetition. Repetition did
not create independent observations. The architecture, candidates, hyperparameters
and seed remained fixed. The development set was reused for this approved
follow-up and must not be described as an untouched final test.

The 32-case challenge, 80 seconds, was frozen before generation and fitting. It
uses disjoint identities, seeds, full configurations and trajectories. Its 28
voiced cases include unseen stationary frequencies and six-harmonic mixtures;
its four aperiodic cases test false voicing. Challenge generation was blocked
until the completed augmented checkpoint bound the training-plan identity.

Frozen records:

| Artifact | SHA-256 |
|---|---|
| Training plan | `7da7463291ee318636b29b109dc2e9ec9515765915e37af95062166e49c5bb17` |
| Challenge plan | `007b63dfcb30d5b08c5dc98d8134ce818d75bb3e9272e2761b47e54b9b5cd6b1` |
| Follow-up selection declaration | `b100de6ca13038014568c6e1a3e3850f3646a3427c94cde224e31f6df2aa5fe1` |
| Augmented checkpoint | `9c18d9aca2829071dfa869a99db6c53e891216b6cf17d7f1cf59f0166d33aee2` |
| Challenge metrics | `2c9b2692b1e90d29be3d057127b77c9c14b0116f0ac2f4f657721cc7239eb1dc` |

The training generator retained about 3.85 MiB and the challenge about 1.13 MiB
of file payloads. Temporary raw PCM and full replay JSON were removed after
compact features and hashes were preserved. No native build or model download
was needed for the curriculum. The original failures and runs were retained.

The pre-score clarification requires each unvoiced challenge case, as well as
pooled false-positive rate, to be at most 5%. To create the declaration in a fresh
reproduction **before challenge evaluation**, run from `research/voice-prosody`:

```sh
.venv/bin/python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
path = Path('runs/synthetic-curriculum-plan/challenge-evaluation-decision.json')
value = {
    'schema': 'phenometrix-synthetic-challenge-evaluation-clarification-v1',
    'declared_at': datetime.now(timezone.utc).isoformat(),
    'timing': 'before any challenge model predictions or scored outcomes',
    'unvoiced_false_positive_gate': 'each of4 unvoiced cases AND pooled false-positive rate<=0.05',
    'baselines': 'input-only reconstruction from frozen parameters; verify PCM and feature identities',
}
with path.open('x') as stream:
    json.dump(value, stream, indent=2)
    stream.write('\n')
PY
```

This fresh declaration has its own timestamp/hash; it does not replace the
recorded declaration or claim historical byte identity. Full acquisition,
feature, training, scoring and export commands are in [README.md](README.md).

## Held-out challenge and final no-go

All methods were predicted before analytic references were read. Reconstructed
classical baselines matched the frozen PCM hashes and feature/candidate arrays
exactly. All 32 challenge cases were scored and reported.

| Method | Equal-scenario voiced success | Worst voiced case | Unvoiced false positives |
|---|---:|---:|---:|
| Original MLP | 72.955% | 0.000% | 0 / 964 |
| Augmented MLP | 88.071% | 2.905% | 0 / 964 |
| Classical DSP | 75.993% | 0.000% | 0 / 964 |
| Fixed continuity | 62.967% | 0.000% | 0 / 964 |
| Causal YIN | 99.822% | 98.755% | 0 / 964 |

The augmented MLP gained 15.116 percentage points on disjoint synthetic cases,
fixed the original 300 Hz harmonic-balance case, and reached 100% on the original
576 glide-interior frames. It still scored 0% on the original plain 300 Hz tone,
and four new cases remained below the required 80% success:

| Challenge source | Augmented success | Candidate oracle |
|---|---:|---:|
| 353.3 Hz stationary, 14 dB noise | 2.905% | 100.000% |
| 479.7 Hz stationary, clean | 50.207% | 100.000% |
| 137 to 509 Hz step, clean | 33.610% | 100.000% |
| 593 Hz harmonic-balance change, 14 dB noise | 45.228% | 46.058% |

Three failures are selection errors despite available accurate candidates. The
fourth is largely a candidate-coverage limit. Overall challenge oracle coverage
is 97.718%; causal YIN succeeds on all four failing cases. The augmented model
misses the 90% equal-scenario gate, the per-case 80% gate and the original 95%
stationary gate. Every noise case individually has zero false positives, and
the transition-lag components pass. Passing components do not erase failed ones.

**Decision: no learned promotion, no additional fitting or architecture search.**
No calibration or sealed-test claim is made. A defensible future proposal would
study YIN-based pitch evidence and voicing/quality failures using fresh prospective
gates, rather than assume that another fit of the current selector will solve
its subharmonic failures. That proposal is not an implemented result.

## Validation record

The full research suite completed with **77 passed in 1.25 seconds** using
`.venv/bin/python -m pytest -q -p no:cacheprovider tests`. Tests cover alignment,
input/reference separation, state/reset semantics, provenance, model arithmetic,
fixed synthetic generation, per-case gates and serialization. Actual 14-case and
32-case scored runs supply behavioral evidence beyond unit tests. Browser and
repository-wide validation are tracked separately; Python success does not certify
a browser service budget, microphone behavior or clinical utility.
