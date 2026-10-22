# Frozen voicing / YIN pitch follow-up

The no-fit hybrid repairs the fixed 300 Hz failure and passes the declared
synthetic, transition and reused natural-development quality components. The
original MLP voicing head plus raw YIN pitch is the preferred research candidate
under the rule frozen before either hybrid was scored. **Production remains DSP.**
The hybrid has no browser implementation, calibrated correctness probability or
independent natural-test result, so this is an offline component result rather
than a deployment qualification.

The repair has a measurable tradeoff. Natural-development joint error is 8.9224%
for the original hybrid, better than DSP's 10.6065%, but worse than its own
candidate-selecting MLP's 8.6349%. The augmented hybrid also worsens its own MLP.
The experiment establishes a useful decomposition across these declared cases;
it does not establish universal improvement.

## What failed and what changed

The original 300 Hz input contains three harmonics with amplitudes 1/.5/.25.
It is not a pure sinusoid. Correct approximately 296 and 300 Hz candidates were
available, but the original MLP chose approximately 150 Hz, the GRU 100 Hz, and
the augmented MLP 150 Hz. Independent NumPy and Torch equations reproduce those
decisions. The [bounded sweep](SUBHARMONIC_SWEEP.md) and
[training-label audit](ALIAS_DIAGNOSIS.md) identify sensitivity to candidate rank,
source evidence and global context, alongside sparse high-frequency natural
training and single-nearest-candidate labels. They do not prove one sufficient
cause across all inputs. No weights, scaling, features or candidate definitions
were changed in this follow-up.

The new contract is `phenometrix.mlp-voicing-yin-pitch.v1`. A frozen MLP supplies
raw voicing probability. Below 0.5, the output is unvoiced. Otherwise, the unchanged
raw causal YIN estimate supplies pitch if finite and within 50–700 Hz; an invalid
estimate is unavailable. The MLP's candidate choice and YIN's old RMS/correlation
voicing gates are unused. This is an explicitly different estimator, not a
reinterpretation of old scores. Both raw voicing probabilities and the absence
of pitch-correctness calibration remain visible in the contract.

The [follow-up plan](HYBRID_FOLLOWUP_PLAN.md) declared the original MLP primary
and the already-fitted augmented MLP secondary. Both face the same gates; prefer
the original if both pass. The plan froze both checkpoints, 19 source identities,
all scoring rules and the complete fresh challenge before new outcomes. There
was no new fit, threshold search, candidate revision or sealed-test access.

## Fixed synthetic evidence

All methods retain the declared 20 ms measurement delay: a 40 ms input window
ending at `e` predicts `t=e−20 ms`, available at `e`. The common support mask is
`t>=64 ms` and `t+32 ms<=N` on the nominal `16+10*k ms` target grid. Predictions
are frozen before analytic truth is used for scoring.

The original 14 cases and previous 32-case challenge are now regression evidence.
The fresh 32-case challenge has 28 voiced and four aperiodic cases, each 2.2 s:
stationary pitches, glides, steps, harmonic-balance changes and noise. Its complete
parameters were declared before generation, including off-grid frequencies,
independent harmonic phases, amplitudes, 10/20 dB noise, weak or absent fundamental
with harmonics 2 and 3 retained, and aperiodic RMS levels. These fixed sources
supply exact analytic truth, not a population generalization estimate.

| Method | Fresh voiced success, equal-case mean | Worst voiced case |
|---|---:|---:|
| DSP | 75.5247% | 0.0000% |
| Existing gated YIN | 99.7969% | 98.5782% |
| Fixed continuity | 61.7637% | 0.0000% |
| Original candidate MLP | 66.8585% | 0.0000% |
| Augmented candidate MLP | 88.3209% | 0.0000% |
| Original hybrid, primary | **99.5091%** | **93.8389%** |
| Augmented hybrid, secondary | **99.8646%** | **98.5782%** |

Success requires a voiced output within 50 cents of truth. Both hybrids have zero
false voiced outputs on all 844 eligible aperiodic frames, with zero in each of
the four cases. They pass the frozen requirements of at least 90% equal-case
success, at least 80% in every voiced case, and at most 5% false voicing in every
unvoiced case. The original hybrid's worst case is `hybrid-fresh-012`, a low-amplitude 331.7–557.9 Hz
glide with 13 false unvoiced outputs among 211 scored frames. All cases, classical
baselines and candidate-oracle coverage are retained in the
[public summary](../../benchmarks/voice-prosody/hybrid-summary.json).

Both hybrids reach 100% success on each original stationary 100, 200 and 300 Hz
case, and on the postchange portions of each original 100, 200 and 300 Hz
harmonic-balance case. Both reach 99.8370% mean and 98.7552% minimum voiced success
on the previous challenge, with zero false voicing on its 964 aperiodic frames.
These successes do not erase the original models' recorded failures.

All eight original transition delays remain defined. Extra median signed target
delay relative to delayed DSP is +0.0979 ms for the original hybrid and +0.0892 ms
for the augmented hybrid, for both the combined and glide-only summaries. The
two-step summary adds 0 ms for either. Each passes the separately declared
combined, step-only and glide-only limit of 10 ms. These are analytic signal
response delays; the 20 ms earliest availability delay remains separate, and
negative target lag is not clamped. They are not wall-clock runtime measurements.

## Reused natural development and its residual errors

The natural comparison reuses 80 previously scored recordings from four speakers,
with clean, 10 dB noise, first-difference and clipped versions: 320 conditions,
253,292 eligible frames, of which 60,760 are voiced. These conditions are dependent
versions of the same sources. Counts are pooled within each speaker/condition;
the primary score weights all 16 cells equally. Because development outcomes
were already observed when this follow-up was designed, this is exploratory
reuse, not an independent natural holdout.

| Method | Joint error, lower is better | Reference-voiced success |
|---|---:|---:|
| DSP | 10.6065% | 62.4534% |
| Existing gated YIN | 11.0210% | 64.8801% |
| Fixed continuity | 11.5983% | 69.3928% |
| Original candidate MLP | 8.6349% | 75.4057% |
| Augmented candidate MLP | 8.2620% | 77.0342% |
| Original GRU, historical failed gate | 7.9527% | 76.3785% |
| Original hybrid, primary | **8.9224%** | **74.1933%** |
| Augmented hybrid, secondary | **8.8142%** | **74.6849%** |

Joint error counts incorrect voicing, unavailable outputs and voiced pitch errors
outside 50 cents. The original hybrid reduces joint error by 1.6841, 2.0985 and
2.6759 percentage points against DSP, gated YIN and continuity, respectively.
The secondary reduces it by 1.7924, 2.2068 and 2.7841 points. Both pass the declared
one-point minimum versus each classical comparator, two-point maximum
voiced-success loss versus the best classical comparator, and one-point maximum
clean-condition excess. Their clean joint error is actually 0.4963 and 0.5180
points lower than the best classical comparator.

Relative to their own MLPs, joint error increases by **0.2875 points** for the
original hybrid and **0.5522 points** for the augmented hybrid. Their voiced
success also falls from 75.4057% to 74.1933% and from 77.0342% to 74.6849%.
This comparison was required reporting, not a newly introduced selection gate.
The better secondary aggregate does not change the predeclared preference for
the original hybrid when both pass.

Substantial errors remain. Under first difference, original-hybrid voiced success
is 54.4417% and false-negative rate 37.7044%; the secondary reaches 56.4677% and
35.1296%. For speaker M03 in that condition, original success is only 40.1080%
and false-negative rate 55.0319%, versus 45.2872% and 48.0609% for the secondary.
M01 first-difference success is 41.0043% and 42.6973%. All 16 cells are published;
the aggregate must not conceal these failures. Pooled across all conditions, the
original hybrid has 8,005 false negatives, six unavailable voiced frames, 7,439
voiced pitch errors and 6,896 false positives. The secondary has 7,627, six,
7,501 and 6,935, respectively. These pooled counts are not the equal-cell macro.

## Integrity checks and disposition

Eleven focused hybrid tests passed before the plan was frozen. The independent
synthetic audit verified all 19 bound sources, 78 PCM/feature/prediction records,
16,968 exact acquisition/target alignments and 560 independently reconstructed
case-method count comparisons. Its analytic truth formulas separately reproduce
the fresh success/false-positive gates and signed transition summaries. Natural
identity checks reproduce 3,200 recording/method comparisons and 40 aggregate
comparisons exactly against prior frozen outputs. Neither audit refits a model.

| Artifact | SHA-256 |
|---|---|
| Frozen plan | `71adf6745c1d28e4ec7595445678f8eb970a3c591fd171d7c2609aab58f9ca6a` |
| Synthetic metrics | `695b878a09b2ac3e706a6fbd22d782d64d11fc20b1fe744e75e1b446b9bee441` |
| Synthetic prediction freeze | `a4ff117e3ef280a2d0b0c6617985900f6aae1a418a7a2ab36f3c50d0efae30c3` |
| Independent synthetic audit | `1191ce3b899ae154daa86ff916f345ef9b6ab59f54796df654358045e6f20d6d` |
| Natural metrics | `225d63547591ee210ff65431dfb3b71e0c7ba239a57365ffee1d68f168fe64c9` |
| Natural prediction freeze | `3a7be525cb503f48d146e1af2f6bcc8feff62825038c9e8d5ed273e75eea6bb0` |
| Natural baseline identity audit | `db66b6678b3f2dd70d3766b4b081b573483c8f27e603a848485c076f2cfaf5ae` |

Raw corpus audio, reference trajectories, feature caches, weights and frame-level
predictions remain ignored. The public summary contains aggregate metrics,
synthetic case definitions and identities. Reproduction commands are in the
[research README](README.md#no-fit-subharmonic-diagnosis-and-hybrid-follow-up).

The original hybrid is retained as an offline research candidate under the frozen
preference rule. All prior failed gates remain failed; this new contract passes
only its declared component checks. A browser implementation and full-stream
qualification of this exact estimator, calibration on separately assigned data,
and independent natural confirmation remain unestablished. No clinical claim,
production promotion or sealed-test result follows from this experiment.
