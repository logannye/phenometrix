# Diagnosing the stationary 300 Hz subharmonic failure

The subsequent no-fit hybrid experiment is complete; see
[HYBRID_RESULTS.md](HYBRID_RESULTS.md) for its separately frozen evaluation.
This report preserves the preceding diagnostic evidence.


The frozen learned models receive candidates near the correct 300 Hz fundamental
but select a subharmonic. The evidence points to periodicity ambiguity, candidate
ordering and sparse training support as testable causes. It does not establish a
single cause, justify a new model fit, or reverse the failed gates in
[RESULTS.md](RESULTS.md). This audit used existing training references, analytic
regression features and frozen checkpoints. The sealed natural test remains untouched.

## Correct signal and observed decision

The original `constant-300` fixture is a stationary **three-harmonic** waveform,
`0.075 × [sin(φ) + 0.5 sin(2φ) + 0.25 sin(3φ)]`, with fundamental 300 Hz and
analytic phase at each sample. It is not a pure sinusoid. The generator is
[`support_diagnostic.ts`](support_diagnostic.ts); the delayed evaluator reused its
exact PCM bytes. The source-description correction in RESULTS changes no metric.

At frozen feature row 20, the five valid candidates are:

| Slot | Frequency, Hz | Retained source evidence |
|---|---:|---|
| 0 | 100.00685 | Refined final-20-ms correlation 0.999939 |
| 1 | 100.00943 | Refined full-40-ms correlation 0.999354 |
| 2 | 150.01959 | Refined final-20-ms correlation 0.994032 |
| 3 | 296.29630 | Legacy autocorrelation 0.993248 and AMDF 0.924197 |
| 4 | 300.03305 | Zero-crossing regularity 0.999853 |
| 5 | unavailable | Padding |

Both slots 3 and 4 satisfy the 50-cent criterion. The original and augmented MLP
choose slot 2, approximately 150 Hz; the GRU chooses approximately 100 Hz. Their
voicing probabilities at this row are 0.936663, 0.984362 and 0.950555 respectively.
Thus this failure is candidate selection, not a rejected-voicing decision.
The preserved DSP reports 296.296 Hz and YIN 300.493 Hz. The complete original
stationary case has 151 scored frames: DSP and YIN pass all, while all three
learned checkpoints fail all. These probabilities are uncalibrated voicing
outputs, not probabilities of correct pitch.

## Mechanisms supported by the source

The refined estimator ranks local autocorrelation maxima by their score at an
integer lag, then retains only one full-window and two suffix-window peaks.
Parabolic refinement changes the reported frequency but does not recompute that
ranking score. At 8 kHz, 300 Hz has period 26.67 samples, whereas three periods
are exactly 80 samples. A period multiple can therefore score above the shorter
fundamental period. The observed near-100 and near-150 candidates demonstrate
this ambiguity. This concerns multiples of a waveform's period, not Nyquist
frequency aliasing.

The feature builder merges only exactly equal frequencies, sorts them by Hz,
and assigns separate parameters to each of six slots. The two nearly equal
100 Hz estimates above remain different slots with separated source evidence.
Changes in which period multiple wins, or whether two estimates happen to be
exactly equal, can move a correct candidate between input slots and output
classes. This design is not invariant to equivalent candidate reorderings or
duplicate representations. An intervention that changes slots is a mechanism
probe; it is not a valid deployment change with the old weights.

The supervised loss chooses the single nearest candidate within 50 cents.
Cross entropy then discourages probability assigned to other candidates, even
when they also meet the evaluation tolerance. This is a mismatch between an
exclusive nearest-candidate training target and the broader accepted-output set,
not a discovered label-timestamp bug. All voiced frames retain voicing loss even
when no candidate is within tolerance. Voicing uses unweighted binary cross
entropy; candidate loss is normalized over covered frames separately.

## Exact training coverage

The independent audit verified each opened feature and reference hash and
reconstructed the fixed 20 ms target mapping and common-support mask. Counts
match the frozen training manifests. The [aggregate audit](../../benchmarks/voice-prosody/training-semantics-audit.json)
contains full histograms and source identities; its
[reproduction script](../../benchmarks/voice-prosody/source-snapshots/training-label-audit-v1.py)
uses NumPy and reads only the training split.

| Quantity | Natural training, four conditions | Unique synthetic supplement |
|---|---:|---:|
| Feature frames | 676,384 | 31,488 |
| Valid supervised frames | 669,664 | 30,848 |
| Reference-voiced frames | 152,772 | 28,920 |
| Reference-unvoiced frames | 516,892 | 1,928 |
| Candidate-covered voiced frames | 132,181 | 28,333 |
| Covered frames with multiple candidates within 50 cents | 121,893 | 25,739 |
| Additional acceptable candidates not chosen as target | 222,611 | 43,832 |

Natural training is 77.19% reference-unvoiced among valid frames. This can inform
voicing priors, but cannot explain the observed voiced 300 Hz subharmonic choice.
Among its voiced frames, only 564 (0.369%) are at least 300 Hz, representing 141
reference positions repeated across four dependent conditions. The maximum
training reference is 361.072 Hz; no voiced training reference reaches 400 Hz.
The high-frequency regime is therefore poorly represented by this bounded
natural subset, not by a claim about the full corpus.

| Reference frequency range, Hz | Natural voiced frames | Unique synthetic voiced frames |
|---|---:|---:|
| Below 100 | 33,532 | 6,660 |
| 100–200 | 88,436 | 7,609 |
| 200–250 | 24,932 | 2,600 |
| 250–300 | 5,308 | 2,545 |
| 300–400 | 564 | 4,827 |
| 400–500 | 0 | 2,419 |
| 500–701 | 0 | 2,260 |

Ranges include their lower bound and exclude their upper bound. Synthetic
sequences were repeated four times for scaling and training, not generated as
four independent sources. Augmented totals are 802,336 feature frames, 793,056
valid frames and 245,513 candidate-covered voiced frames.

| Target slot | Natural | Unique synthetic | Combined after fourfold synthetic repetition |
|---|---:|---:|---:|
| 0 | 38,988 | 4,943 | 58,760 |
| 1 | 40,992 | 5,666 | 63,656 |
| 2 | 34,750 | 6,175 | 59,450 |
| 3 | 15,756 | 7,215 | 44,616 |
| 4 | 1,670 | 4,323 | 18,962 |
| 5 | 25 | 11 | 69 |

Over 92% of naturally covered frames have multiple acceptable candidates; only
1.28% have an exclusive training target in slot 4 or 5. The synthetic supplement
adds high-frequency examples and changes this distribution, but uses five
randomized harmonics and phases. Its stationary frequencies are a fixed
logarithmic grid, with 320.202 Hz nearest the failed 300 Hz fixture. Absence of
that exact waveform alone does not explain failure; generalization beyond exact
training examples was a requirement.

## Scaling and saturation are insufficient explanations

Independent CPU evaluation of the saved float32 weights reproduces the observed
MLP decisions. For row 20 of each stationary fixture:

| Fundamental | Maximum absolute standardized feature, original / augmented | Projection units with absolute preactivation >3, original / augmented |
|---|---|---|
| 100 Hz, passes | 17.89 / 9.78 | 10 / 4 |
| 200 Hz, passes | 14.20 / 9.61 | 8 / 7 |
| 300 Hz, fails | 7.29 / 5.89 | 5 / 1 |

Saturation and outlying standardized features occur, but are stronger on some
passing examples. They are not a discriminating explanation of this failure.
The training code verifies feature-name order before scaling and fitting; this
audit found no order or scaling-contract discrepancy. These comparisons do not
exclude interactions between scaling, sparse slots and learned priors.

The separately frozen [subharmonic sweep](SUBHARMONIC_SWEEP.md) examines nearby frequencies, phases and all predeclared candidate interventions. Its new cases are diagnostic evidence, not an independent qualification of a subsequently designed estimator.

## What another experiment may establish

The original 14 cases, the previously evaluated 32-case challenge, and reused
natural development are now diagnostic/regression material for any new proposal.
Their prior held-out status applies only to the earlier frozen attempt. A new
deterministic rule also requires prospective evaluation; absence of model fitting
does not make adaptive rule design independent of the observed failures.

A bounded next comparison should freeze its source, support/target mapping,
voicing policy, period-selection rule and failure accounting before generating
or scoring a fresh challenge. Test nearby frequencies and phases around integer
lag coincidences, weak/missing fundamentals, noise and gain, octave steps,
glides and aperiodic cases. Report all declared cases. A failed frame or withheld
pitch remains a failure in the existing success denominator. Preserve at least
the prior stationary/post-change 95%, challenge macro 90% and per-case 80%,
unvoiced false-positive 5%, and transition-lag requirements; these are engineering
gates, not population guarantees. Natural development retains its original
comparisons and remains exploratory. Sealed pairs 5 and 6 remain unopened.

For any future learned revision, a multi-positive loss over the fixed acceptable
candidate set or a representation insensitive to duplicate candidates is a
falsifiable alternative. Neither is implemented or fitted by this audit. Do not
apply frozen weights to a reordered or deduplicated feature schema. A hybrid of
frozen voicing with a different deterministic pitch estimator is a new estimator
and must freeze a new prospective evaluation contract.

## Cached YIN evidence available to a future input-only diagnostic

Natural feature caches under `runs/delayed-train-features/predictions/` and
`runs/delayed-development-features/predictions/` retain `yin_evidence[]` with
`sample`, `raw_f0_hz`, `window_rms` and `normalized_lag_correlation`. Raw F0 is
retained when the separately gated `yin[]` output is unvoiced and has null F0.
`sample` is the immutable support endpoint; the declared delayed target is
`sample - 960` at 48 kHz. The original diagnostic voicing gate requires RMS
at least 0.003 and normalized correlation at least 0.55.

The known-truth `runs/delayed-known-truth/predictions/*.json.gz` and augmented
counterpart retain only gated `predictions.yin[]`. The challenge evaluator also
discards returned raw YIN evidence. Rejected-frame raw pitches cannot be recovered
from those gated files. Any reconstruction must use frozen synthetic PCM or
parameters input-only, complete and hash all predictions before scoring, and
must never substitute analytic target values.

## Identities

Local run artifacts are ignored; no audio or reference trajectories are included
in this report. The aggregate audit binds every listed file below. Full natural
and synthetic feature/reference file identities are in their acquisition and
feature manifests. All prior predictions and checkpoint files remain unchanged.

| Artifact | SHA-256 |
|---|---|
| `delayed-train-features/manifest.json` | `35e92d701dcd52891c256fa99b9567c5db4d80e14f4ef71858822b470edf5a87` |
| `delayed-train-features/acquisition-manifest.json` | `50d25236779d5cd95d3eeef58826c5f30825cb733af10d70d664108d8df66f28` |
| `synthetic-train-curriculum/manifest.json` | `8dd2e18a5ae705c6ce21edfab6dbd3acfdb86593a0d14ccffc96df2e0b708c90` |
| `synthetic-curriculum-plan/train-plan.json` | `7da7463291ee318636b29b109dc2e9ec9515765915e37af95062166e49c5bb17` |
| `delayed-mlp-seed20261022/manifest.json` | `73533700a756e295876a3145017b3d263c6af0b73d962ec2bf9b7cdd9ac2d61f` |
| `delayed-mlp-seed20261022/weights.pt` | `06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d` |
| `delayed-augmented-mlp-seed20261022/manifest.json` | `ed81dbab0e63a4578e3684ed27655fce5464335e90e7108410e9ade45e021c62` |
| `delayed-augmented-mlp-seed20261022/weights.pt` | `9c18d9aca2829071dfa869a99db6c53e891216b6cf17d7f1cf59f0166d33aee2` |
| `delayed-known-truth/features/constant-300.npz` | `cb41b3a9d922184255368a1903e0138dfb84a7b13d34d1787d1d69ec4248284a` |
| `delayed-known-truth/predictions/constant-300.json.gz` | `cbd142d1b4834556ef93b6537c71485a3f220520faae01905e0f8119affa8b43` |
| `delayed-augmented-known-truth/predictions/constant-300.json.gz` | `52e1e59903aac6e9173b7cdd35e99bf153d3caf77e193debf3a5b8e0a4c036f9` |

Source identities for the generator, model, labels, alignment and candidate code
are in the aggregate audit. To reproduce counts after other timing work finishes,
run from the repository root:

```sh
research/voice-prosody/.venv/bin/python benchmarks/voice-prosody/source-snapshots/training-label-audit-v1.py
```

The public script differs from the executed audit only in repository-root
discovery; the executed script hash is retained in the aggregate record. This
report does not claim a new fit, new natural holdout, calibrated uncertainty, or
production readiness.
