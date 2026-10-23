# What the pitch errors say about causal support

This report records the original endpoint diagnostic before model fitting. Its
next-check recommendation was subsequently carried out in the
[alignment audit](ALIGNMENT_AUDIT.md). The separate delayed experiment, trained
models, and final no-go are reported in [RESULTS.md](RESULTS.md); they do not
change the endpoint scores preserved here.

The existing 40 ms estimator responds to a changing source approximately 20 ms behind the exclusive window endpoint, even when the signal's instantaneous frequency is known exactly. The revised candidate set supplies some evidence closer to the endpoint, with approximately 10 ms of lag on the tested glides. This explains a real part of the error mechanism. It does not explain every natural-speech error, and it does not rescue the failed training prerequisite: v2 candidate coverage was **64.57% speaker-macro**, below the predeclared **75%** threshold. No model was fitted and no third candidate revision was introduced.

These findings come from two separate analyses: a fixed analytic synthetic experiment, and descriptive strata of the original 24 training recordings. Neither is held-out model performance. An oracle chooses the closest candidate using the reference target, so its result is a ceiling for a selector, not an executable estimator's accuracy.

## Known-truth experiment

The fixed synthetic set contains constant 100, 200 and 300 Hz signals; six one-second linear glides centered on 200 Hz at slopes −200, −100, −50, +50, +100 and +200 Hz/s; and steps from 100 to 200 Hz and from 200 to 100 Hz. Glides have 250 ms of initial plateau and 350 ms of final plateau. Steps occur at 600 ms. Each signal is 1.6 seconds at 48 kHz. The waveform is a phase-continuous harmonic sum with relative amplitudes 1, 0.5 and 0.25, without noise. Phase is computed from the analytic integral of F0 at each sample, rather than accumulating an approximate numerical frequency update.

The existing DSP, v2 evidence adapter and causal YIN all use the original `16 + 10*k` ms target grid and complete trailing 40 ms support. DSP and YIN keep the pilot's existing/disclosed voicing gates. The same boundary mask also requires the centered comparator's future 32 ms support to exist. All acquisition predictions are generated before their scorer opens analytic target arrays. Midpoint and window-mean F0 are additional diagnostic targets; the primary endpoint target and its availability remain unchanged.

On the 576 frames whose complete support lies inside a glide:

| Method | Within 50 cents of endpoint, all eligible frames | Median absolute endpoint error | p95 absolute endpoint error | Median equivalent slope delay |
| --- | ---: | ---: | ---: | ---: |
| Preserved DSP | 91.15% | 19.97 cents | 56.04 cents | 19.94 ms |
| Causal YIN | 93.92% | 17.36 cents | 52.22 cents | 20.11 ms |
| v2 candidate oracle | 100% | 7.77 cents | 26.60 cents | 9.79 ms |

All three methods have 100% pitch coverage on these frames. The equivalent delay is `(target F0 - estimate F0) / slope`, evaluated only inside a linear glide. It describes contour response, not worker service time or measured browser latency. The oracle's equivalent delay describes its reference-aware choice among candidates.

The signed error reverses with source direction and increases with slope:

| Glide slope, Hz/s | DSP mean signed cents | YIN mean signed cents | v2 oracle mean signed cents |
| ---: | ---: | ---: | ---: |
| −200 | +38.08 | +41.92 | +18.12 |
| −100 | +18.45 | +21.43 | +7.30 |
| −50 | +8.59 | +11.58 | +3.53 |
| +50 | −7.95 | −5.47 | −3.48 |
| +100 | −17.38 | −14.26 | −6.82 |
| +200 | −36.65 | −33.43 | −17.49 |

A lagged estimate of a rising contour is low; a lagged estimate of a falling contour is high. For sufficiently small changes, signed cents error is approximately `−1200 * lag_seconds * slope_Hz_per_second / (ln(2) * F0_Hz)`. The observed signs and scaling support this mechanism independently of PTDB reference labels.

When the scorer instead asks how well the same predictions describe the midpoint or mean frequency of their observed window, all three methods are within 50 cents on every glide-interior frame. Median absolute midpoint errors are 10.17 cents for DSP, 2.75 cents for YIN and 0.48 cents for the v2 oracle. The remaining DSP midpoint error is compatible with its discrete lag resolution. This comparison does not authorize relabeling endpoint predictions after seeing errors.

All 1,051 complete-support plateau frames are within 50 cents for all three methods. For the upward 100→200 Hz step, the first of three consecutive correct endpoint estimates begins 36 ms after the change for DSP and YIN, versus 16 ms for the v2 oracle. For the downward 200→100 Hz step, all three begin at 16 ms. Those values retain the 10 ms output-grid phase. The asymmetry is itself informative; a single universal response-delay correction would not capture every transition.

## What remains different in natural speech

The original 24-recording pilot contains 4,014 eligible, valid, reference-voiced frames from 12 training speakers. Pooled v2 oracle coverage is **65.72%**, while the equal-speaker macro result is **64.57%**. There are 85 frames with no valid candidate. Candidate absence is counted as failure in every coverage denominator.

The following tables are descriptive scorer-only strata. Central reference slope is `[F0(t+10 ms) − F0(t−10 ms)] / 20 ms`. Future reference values are used only to describe an already-frozen error; they are not inference features. Frames with an absent, invalid or unvoiced neighboring reference are shown separately. F0 and slope tables exclude that boundary group.

| Reference group | Frames | Pooled oracle coverage | No candidate | Median nearest error when a candidate exists |
| --- | ---: | ---: | ---: | ---: |
| F0 below 100 Hz | 907 | 54.13% | 16 | 43.75 cents |
| F0 100–200 Hz inclusive | 2,057 | 73.75% | 22 | 23.67 cents |
| F0 above 200 Hz | 668 | 72.90% | 5 | 25.28 cents |
| Adjacent unvoiced/invalid/absent reference | 382 | 37.43% | 42 | 71.61 cents |

| Absolute central reference slope, Hz/s | Frames | Pooled oracle coverage | No candidate | Median nearest error when a candidate exists |
| --- | ---: | ---: | ---: | ---: |
| 0–200 inclusive | 2,490 | 79.20% | 33 | 18.55 cents |
| Greater than 200 through 500 | 845 | 52.31% | 6 | 47.45 cents |
| Greater than 500 through 1,000 | 224 | 29.46% | 3 | 76.90 cents |
| Greater than 1,000 | 73 | 20.55% | 1 | 99.96 cents |

The original synthetic glides ended at 200 Hz/s, while 1,142 non-boundary natural frames exceed that magnitude. Those faster frames account for 619 of the 1,376 oracle failures. The adjacent-reference-boundary group contributes another 239 failures. These groups are disjoint. The remaining 518 failures occur in the slow-slope group, so fast contours and boundaries are not a complete explanation.

Low F0 is also important. Even within the ≤200 Hz/s group, oracle coverage is 62.40% below 100 Hz, 85.22% at 100–200 Hz, and 91.32% above 200 Hz. A fixed error in Hz occupies more cents at lower F0. Around 80 Hz, a roughly 115 Hz/s slope over a 20 ms lag can already exceed 50 cents. The v2 short-support candidates intentionally start at 100 Hz; the low-F0 frames retain the longer-support evidence needed for multiple periods. The current results expose that tradeoff without resolving it.

Per-speaker counts and coverage, cross-strata, median failed-frame errors, candidate absence and deterministic example traces are retained in the machine-readable artifacts. The apparent precision of thousands of frames is not a sample size of thousands of independent speakers. The reference is algorithmically extracted laryngograph F0 and can itself be uncertain near transitions or irregular phonation.

## Decision and next discriminating check

The synthetic experiment establishes physical support lag without relying on corpus labels. The natural strata show that stronger local dynamics, low F0 and boundaries accompany reduced candidate coverage. They do not establish whether a residual microphone/reference alignment offset, vocal-waveform complexity or reference-estimator behavior explains the remaining errors. In particular, the 20 ms synthetic lag is not a license to shift PTDB predictions by 20 ms and report an improved primary score.

The next useful check is a bounded training-only review of paired microphone and raw laryngograph traces with one fixed timestamp convention, emphasizing low-F0 slow-slope errors and voiced boundaries. An independent cycle-period or otherwise explicit local laryngograph estimate can test whether the provider reference, microphone periodicity and claimed target time agree. Any offset claim needs a shared physical or processing cause; per-recording shifts selected to maximize scores remain excluded. This investigation should precede another trained-estimator commitment. The existing failed v2 prerequisite stays failed.

## Reproduction and evidence

From `apps/capture-web`, create a new output directory:

```sh
node --import tsx ../../research/voice-prosody/support_diagnostic.ts \
  --output ../../research/voice-prosody/runs/support-diagnostic-new
```

From `research/voice-prosody`, score the analytic experiment and separately stratify an existing frozen pilot:

```sh
.venv/bin/python -m phenometrix_prosody.support_diagnostic \
  --run runs/support-diagnostic-new
.venv/bin/python -m phenometrix_prosody.natural_error_stratification \
  --pilot runs/train-pilot-v2
```

The second command refuses to overwrite existing strata; the retained run already has these results. Reproduction should use a copied fresh run directory or remove only regenerated strata files after preserving the original evidence.

- Synthetic definitions and predictions: `runs/support-diagnostic-v1/manifest.json`, `prediction-freeze.json`, `predictions/`, `analytic-targets/`.
- Independent YIN freeze and scored traces: `runs/support-diagnostic-v1/yin-prediction-freeze.json`, `scores.json`, `scored-traces.json`, `evaluation-manifest.json`.
- Natural descriptive results: `runs/train-pilot-v2/natural-strata.json` and `natural-strata-rows.json`.
- Original candidate decision: `runs/train-pilot-v2/gate-decision.json`; the predeclared 75% prerequisite was not met.

The causal YIN call retains the exact 1,920-sample, 40 ms support. Librosa 0.11.0 warns that its strict two-period check at 50 Hz would require 1,921 samples. The warning is preserved; the experiment did not silently alter support to remove it. This is an offline accuracy/support investigation, not a browser timing result, calibration claim, clinical validation or trained-model improvement.
