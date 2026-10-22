# Frozen no-fit MLP-voicing / YIN-pitch follow-up

This experiment is explicitly authorized after the original no-go and the bounded
subharmonic diagnosis. It preserves all prior failures and does not fit weights,
tune thresholds, modify v2 features, or open the sealed test.

The new estimator, `phenometrix.mlp-voicing-yin-pitch.v1`, uses the original MLP
checkpoint as its primary voicing source. The augmented MLP is a predeclared
secondary voicing source. Their weight identities are:

| Variant | Weights SHA-256 |
|---|---|
| Original, primary | `06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d` |
| Augmented, secondary | `9c18d9aca2829071dfa869a99db6c53e891216b6cf17d7f1cf59f0166d33aee2` |

Both independently face exactly the same gates. Prefer the original if both pass;
use the secondary only if the original fails and the secondary passes. This avoids
an adaptive switch after seeing a new outcome.

Raw p(voiced) below 0.5 returns unvoiced. Otherwise the unchanged raw causal YIN
estimate in the inclusive 50–700 Hz range supplies pitch. A missing, nonfinite or
out-of-range estimate returns unavailable. The candidate head is ignored. No
additional RMS, correlation, continuity or confidence gate is applied. This changes
the estimator contract; it does not correct old results. Raw voicing probability
remains uncalibrated, and F0 correctness probability remains unavailable.

YIN retains librosa 0.11.0, a 1,920-sample frame and 480-sample hop at 48 kHz,
`center=False`, `fmin=50`, `fmax=700`, and the existing frame phase. Support ends
at e and predicts target t=e−960 samples, available at e. The common mask requires
t≥64 ms and t+32 ms≤N. Reference labels, future true F0 and ground-truth history
never enter inference.

Hash-verify cached raw YIN evidence and both frozen MLP prediction streams on all
320 development conditions. Freeze both new prediction streams before reading
reference content. Report every recording, speaker and condition, plus the equal
speaker/condition macro. Compare DSP, originally gated YIN, continuity, both
original candidate-selecting MLPs and both hybrids. Natural component gates remain
at least 1 percentage point of joint-error reduction against EACH of DSP, YIN and
continuity, no more than 2 points of voiced-success loss versus the best of those
three, and clean joint-error excess no more than 1 point versus the best of those
three. Differences against each hybrid's own MLP are reported explicitly.

Reuse all 14 original analytic regressions and report the previous 32-case
challenge as regression context. Every original stationary and post-change
observation case must reach 95% voiced success. All eight original transitions
must have defined delays; combined, step-only and glide-only extra median target
delay may not exceed 10 ms relative to delayed DSP. Negative lag remains signed,
and earliest availability delay is reported separately.

Freeze one fresh 32-case analytic challenge before generating or scoring it:
8 stationary, 8 glides, 8 steps, 4 harmonic-balance changes and 4 aperiodic sources.
Each lasts 2.2 seconds at 48 kHz, totaling 70.4 seconds. New off-grid frequencies,
seeds, independent harmonic phases and complete trajectories are declared in
`plan.json`. Gains cycle through .006, .012, .03 and .075; voiced cases cycle
through no noise, 20 dB, no noise and 10 dB. Harmonic profiles include a pure
sinusoid, dominant, weak and absent fundamental; an absent fundamental retains
harmonics 2 and 3. White and first-difference noise have fixed RMS values .009,
.021, .043 and .065. No parameter changes follow outcomes.

Fresh challenge gates require at least 90% equal-scenario voiced success, at least
80% in EACH voiced case, and at most 5% false voicing in EACH unvoiced case. Report
all cases and candidate-oracle coverage. These deterministic sources do not
establish natural population generalization. Existing development outcomes were
already observed; no independent natural test is claimed. Passing these components
cannot promote a model. Browser parity, complete-stream timing/drop accounting,
calibration, confirmation and a separate final freeze remain later work. The
sealed test stays closed during this experiment.
