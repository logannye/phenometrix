# Frozen-model subharmonic sweep

The subsequent no-fit hybrid experiment is complete; see
[HYBRID_RESULTS.md](HYBRID_RESULTS.md) for its separately frozen evaluation.
This report preserves the preceding diagnostic evidence.


The 300 Hz failure is a learned pitch-selection error, not a missing correct
candidate or a Torch/NumPy discrepancy. The original MLP declares the original
three-harmonic input voiced with probability about 0.937, then chooses about
150 Hz. The augmented MLP declares it voiced with probability about 0.984 and
makes the same selection. Those raw probabilities are not calibrated correctness.
The original case is a harmonic signal, not a pure sinusoid.

The new diagnostic was explicitly authorized after the original no-go. Its
[plan](ALIAS_FOLLOWUP_PLAN.md) was frozen before new inputs or scores. It changes
no weights, features, candidates, reference convention, production behavior or
old gate. The sealed test remains unopened. Training-distribution and label
semantics are examined separately in [ALIAS_DIAGNOSIS.md](ALIAS_DIAGNOSIS.md).

## Bounded physical sweep

The fixed sweep covers 28 frequencies from 75 to 600 Hz, with extra points around
300 Hz, three harmonic profiles and two initial phases. Each case lasts 0.4 s,
with an independent history/state reset. The profiles are a pure sinusoid,
amplitudes 1/.5/.25, and amplitudes 0/1/.5; the last retains harmonics 2 and 3.
There is no noise. Predictions are frozen before the scorer uses frequency.
The declared target is e−20 ms, available at e; the unchanged common-support mask
leaves 31 scored frames per case, 5,208 total from 6,048 acquired frames.

| Method | Equal-case success within 50 cents | Cases with zero success |
|---|---:|---:|
| Original MLP | 63.96% | 57 |
| Original GRU | 60.08% | 54 |
| Augmented MLP | 77.67% | 37 |
| Preserved DSP | 79.76% | 34 |
| Fixed YIN | 100.00% | 0 |

The v2 candidate oracle is 98.81%. Its two failing cases are missing-fundamental
550 Hz at the two phases. Both MLPs output voiced on every scored frame; their
sweep failures are entirely pitch errors. YIN succeeds on every declared case,
including both oracle failures, so its independent pitch output is useful for a
bounded component-replacement test. This densely sampled deterministic sweep is
a diagnosis, not a population accuracy estimate or independent qualification.

All three learned models succeed on the harmonic 299 and 305 Hz cases, but fail
all scored frames at 299.99, 299.999, 300, 300.001, 300.01 and 301 Hz, at both
phases. The correct candidates remain available. Around 300 Hz, integer-lag
peak ranking gives strong refined evidence to subharmonics before parabolic
refinement. At 300 Hz the five candidates are approximately
100.00685, 100.00943, 150.01959, 296.29630 and 300.03305 Hz. The two near-100-Hz
entries carry different source evidence and remain separate because only exact
frequency duplicates are merged. Correct candidates occupy slots 3 and 4.
At 299 Hz correct candidates occupy slots 2, 3 and 4. This change in the candidate
representation accompanies a sharp change in learned decisions.

## Artificial interventions and inference checks

The diagnostic reports all predeclared interventions on the original 300 Hz
regression, not only successful ones. Removing the near-duplicate full-40-ms
100 Hz candidate and repacking sorted slots changes the original MLP from
0/151 to 151/151 correct frames, without adding any pitch information. It does
not repair the augmented MLP. For the augmented model, replacing the nine
global features by their training means repairs all 151 frames; individual
neutralizations of several source/rank coordinates or estimator agreement also
change the winner. Adjacent slot permutations change decisions, but violate the
model's declared sorted-slot input contract and are not candidate fixes.

Clamping standardized inputs to ±3 does not repair any model. Resetting state
every frame does not repair the GRU. No individual feature neutralization repairs
the original MLP. These results localize dependence on combinations of candidate
rank, source evidence and global context; they do not identify one universally
sufficient cause or justify deleting a candidate in production.

Torch and independent NumPy candidate logits agree over all 156 original acquired
frames within maximum absolute errors 1.19e−7 for MLP, 5.96e−7 for GRU and 2.98e−7
for augmented MLP. Three focused diagnostic tests pass. Masks and raw training
scaling are unchanged. The model therefore reproduces its bad learned selection;
this is not repaired by changing inference runtimes.

## Evidence and next test

The output is `runs/subharmonic-diagnostic-v1/`, with a frozen plan, source
snapshots, per-case features, compressed predictions, complete counterfactuals
and scores. Plan SHA-256 is
`ab7997e371b9bbd2dcaba3521f64eb2ce83fb183a823dac853ea352776a47604`;
results SHA-256 is
`2dc68a3704b39a018ac406ca1be7c4dc55a60123fcc2519f4064ded5fcfa6981`.
Raw artifacts remain ignored. Run with a fresh output directory:

```sh
cd research/voice-prosody
.venv/bin/python -m phenometrix_prosody.alias_diagnostic --output runs/subharmonic-diagnostic-new
```

The next authorized experiment is a fixed decomposition: retain the original
MLP's voicing output and replace its candidate selection with the existing raw
YIN frequency. This changes the estimator contract and must be evaluated under
a separately frozen plan; no improved natural score or deployment claim is made
here. Another fitting run is unnecessary to test that hypothesis. Old failures
remain failed, and existing challenge/sweep outcomes are now diagnostic evidence.
