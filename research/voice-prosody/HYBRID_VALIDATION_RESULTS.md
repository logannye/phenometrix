# Independent natural-speech and browser hybrid validation

The frozen original MLP-voicing/raw-YIN hybrid **passes the declared natural
quality component on four previously reserved speakers**. Its equal-speaker,
equal-condition joint error is 9.6747%, compared with DSP's 11.8482%, gated YIN's
11.3622% and continuity's 11.2488%. The original candidate MLP remains better on
these natural recordings at 9.1158%. The hybrid therefore retains the same
tradeoff seen in development: it repairs the fixed synthetic pitch failures,
while sacrificing some natural pitch-selection accuracy. **Production remains DSP.**

The browser's first numerical implementation failed one reserved-input raw-YIN
check. A single declared arithmetic repair passes the unchanged analytic,
development and reserved-input checks. The original failure remains failed.
The sustained browser runtime component also passes its frozen local gates,
including the 30 min soak. These component passes establish neither calibrated
correctness nor clinical use.

## Frozen method and independent sample

The [validation plan](HYBRID_VALIDATION_PLAN.md) fixed the original checkpoint,
all 24 natural-pipeline source files, configuration, inventory, sample selection
and decision rules before reserved microphone or reference acquisition. The
checkpoint is the existing 32-unit framewise MLP with v2 57-feature inputs; no
model was fitted, selected, augmented or calibrated in this stage. The hybrid
contract remains `phenometrix.mlp-voicing-yin-pitch.v1`: raw MLP p(voiced)<0.5
returns unvoiced; otherwise raw causal librosa 0.11 YIN supplies pitch if finite
and within inclusive 50–700 Hz, with unavailable returned outside that range.
Candidate logits are diagnostic. The gated-YIN comparator retains its historical
RMS/correlation voicing gate; the hybrid does not use that gate.

Exactly 40 prompts per speaker were selected by the existing metadata-only SHA
order for F05, M05, F06 and M06, excluding sa1/sa2. There were no substitutions,
post-outcome exclusions or sample expansion. The pre-access metadata audit found
no prior reserved acquisitions in the checked ledger, corpus filenames or 47
historical manifests, and no training/development overlap in prompt IDs or
normalized texts. This is unseen-speaker evidence **within PTDB-TUG**, with the
same corpus recording protocol and reference extraction procedure. It is not an
external population or device cohort, and the metadata audit is not an operating
system access log.

The 160 microphone sources contain 57,611,808 samples at 48 kHz: 1,200.246 s,
about 20.0 min. Clean, 10 dB noise, first-difference and clipping conditions yield
640 dependent versions and 4,800.984 s of transformed input. Those versions and
their frames do not constitute additional independent participants. Each
recording resets feature history and estimator state. Predictions use 40 ms
support ending at `e` for target `t=e−20 ms`, available at `e`; this is 20 ms
future support relative to the reported target, not zero lookahead.

All 640 prediction sets were frozen at **20:31:01.694897 UTC on 22 October 2026**.
Only afterward was the reference-acquisition command launched, at 20:36:08.060115
UTC; its first completed reference download was 20:36:13.010636 UTC. The scorer
retains the provider's nominal `16+10*k ms` targets, common mask `t>=64 ms` and
`t+32 ms<=N`, and fixed validity rules. It neither fits alignment nor repairs the
reference. The laryngograph-derived target is an independent sensor reference,
not manually established physiological truth; its filtering history and exact
physiological alignment remain uncertain.

## Natural accuracy, unconditional denominators and the keep rule

Joint error includes incorrect voicing, unavailable outputs and voiced pitch
errors greater than 50 cents. Voiced success counts a voiced prediction within
50 cents among **all reference-voiced frames**, including false negatives and
unavailable outputs in its denominator. False-positive rate uses reference
unvoiced frames. False-negative-or-unavailable and octave-error rates use all
reference-voiced frames. Counts are pooled within each speaker/condition, then
all 16 cells receive equal weight.

| Method | Joint error | Voiced success | False positive | False negative or unavailable | Octave error |
|---|---:|---:|---:|---:|---:|
| DSP | 11.8482% | 60.1734% | 1.1837% | 34.1754% | 1.4050% |
| Existing gated YIN | 11.3622% | 64.9050% | 2.3250% | 24.1287% | 2.0339% |
| Fixed continuity | 11.2488% | 71.3013% | 4.6043% | 15.0521% | 3.1324% |
| Original candidate MLP | 9.1158% | 74.0813% | 2.7177% | 13.6715% | 2.8798% |
| Original MLP-voicing/raw-YIN hybrid | 9.6747% | 72.0178% | 2.7094% | 13.7772% | 2.8984% |

The hybrid reduces joint error by **2.1735, 1.6875 and 1.5741 percentage points**
against DSP, gated YIN and continuity, exceeding the frozen one-point minimum
against each. Its voiced success is 0.7165 points higher than the best classical
comparator, within the allowed two-point loss. Clean joint error is 0.1936 points
lower than the best classical comparator, within the one-point excess limit.
All required cell denominators are defined. These are the original point-estimate
gates; no confidence-interval criterion was added after scoring.

Relative to its own candidate MLP, the hybrid's joint error is **0.5590 points
worse**, and voiced success is 2.0636 points lower. This required comparison is a
reported tradeoff, not a posthoc model-selection rule. No augmented-head switch
or additional candidate configuration was evaluated on these reserved references.

There are 473,056 eligible valid reference frames across conditions, comprising
131,304 voiced and 341,752 unvoiced frames. Invalid references and valid F0 outside
50–700 Hz both number zero; 3,200 reference entries are excluded by the fixed
boundary mask. Each method also has 1,920 predictions outside the supplied
reference extent, reported separately rather than assigned invented labels.
The hybrid has 45,894 pooled joint errors: 18,011 false negatives, 137 unavailable
voiced outputs, 18,513 voiced pitch errors, 9,204 false positives and 29 unavailable
unvoiced outputs. All 166 unavailable outputs count as errors. The 3,748 octave
errors are a subset of voiced pitch errors. Pooled counts are not the equal-cell
macro shown above; hybrid macro measurement coverage is 99.9658%.

## Condition and speaker effects

Every method's condition-level joint error is shown below, with equal speaker
weight. First difference is the hardest condition for every method, and the
hybrid does not eliminate that failure mode.

| Condition | DSP | Gated YIN | Continuity | Candidate MLP | Hybrid |
|---|---:|---:|---:|---:|---:|
| Clean | 9.5197% | 8.6446% | 9.6414% | 7.9738% | 8.4510% |
| 10 dB noise | 9.3486% | 8.2453% | 8.2836% | 7.5489% | 8.1563% |
| First difference | 19.1702% | 19.9256% | 17.4217% | 12.9158% | 13.6707% |
| Clip at 0.1 | 9.3545% | 8.6332% | 9.6486% | 8.0245% | 8.4209% |

All 16 hybrid speaker/condition cells follow. Coverage means a voiced or unvoiced
output, not correctness. The full public summary includes all five methods'
per-record counts, all speaker/condition cells, pair effects, coverage and error
components; raw audio, reference trajectories and frame predictions stay ignored.

| Speaker | Condition | Eligible frames | Voiced frames | Joint error | Voiced success | False positive | False negative or unavailable | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F05 | Clean | 30802 | 8865 | 7.8047% | 82.5268% | 3.8656% | 4.7152% | 99.9675% |
| F05 | 10 dB noise | 30802 | 8865 | 7.4670% | 80.9137% | 2.7442% | 5.3920% | 99.9773% |
| F05 | First difference | 30802 | 8865 | 12.8596% | 57.2138% | 0.7476% | 34.2470% | 99.9740% |
| F05 | Clip at 0.1 | 30802 | 8865 | 7.7235% | 82.8652% | 3.8884% | 4.6475% | 99.9675% |
| F06 | Clean | 27041 | 7457 | 8.1432% | 81.4268% | 4.1667% | 3.0575% | 99.9963% |
| F06 | 10 dB noise | 27041 | 7457 | 7.6255% | 80.2602% | 3.0127% | 3.7683% | 100.0000% |
| F06 | First difference | 27041 | 7457 | 10.6949% | 62.9878% | 0.6740% | 24.7687% | 99.9963% |
| F06 | Clip at 0.1 | 27041 | 7457 | 8.0618% | 81.7219% | 4.1667% | 2.9502% | 99.9926% |
| M05 | Clean | 30333 | 8045 | 9.9759% | 68.2536% | 2.1132% | 9.3971% | 99.9044% |
| M05 | 10 dB noise | 30333 | 8045 | 10.0913% | 65.7676% | 1.3729% | 12.0075% | 99.8978% |
| M05 | First difference | 30333 | 8045 | 16.7936% | 37.5016% | 0.2961% | 54.3816% | 99.9637% |
| M05 | Clip at 0.1 | 30333 | 8045 | 10.0946% | 67.8310% | 2.1222% | 9.0864% | 99.8253% |
| M06 | Clean | 30088 | 8459 | 7.8802% | 84.4308% | 4.8731% | 3.5583% | 100.0000% |
| M06 | 10 dB noise | 30088 | 8459 | 7.4415% | 83.6624% | 3.9623% | 3.7593% | 100.0000% |
| M06 | First difference | 30088 | 8459 | 14.3346% | 49.9468% | 0.3653% | 41.4115% | 99.9900% |
| M06 | Clip at 0.1 | 30088 | 8459 | 7.8038% | 84.9746% | 4.9794% | 3.2864% | 100.0000% |

M05 under first difference reaches only **37.5016% voiced success**, with
54.3816% false-negative-or-unavailable rate and 16.7936% joint error. M06 under
first difference reaches 49.9468% voiced success. F06's octave-error rate in that
condition is 5.9139% of reference-voiced frames. These are material residual
limitations despite the aggregate component pass. Even clean M05 voiced success
is only 68.2536%, so the result does not imply uniformly reliable speech tracking.

## Pair effects and conditional prompt uncertainty

The two pair effects are the primary indication of between-group variation in
this small sample. The table reports comparator joint error minus hybrid joint
error, in percentage points; positive values favor the hybrid. Both pairs improve
on every classical comparator, while both worsen relative to the original MLP.

| Comparator | Pair 5 advantage | Pair 6 advantage | Conditional 95% prompt-bootstrap interval |
|---|---:|---:|---:|
| DSP | +2.6193 | +1.7277 | [+1.9858, +2.3635] |
| Existing gated YIN | +1.7208 | +1.6541 | [+1.5257, +1.8550] |
| Fixed continuity | +1.0213 | +2.1269 | [+1.3579, +1.7840] |
| Original candidate MLP | -0.8478 | -0.2702 | [-0.6536, -0.4743] |

Pair ranges are 1.7277–2.6193 points versus DSP, 1.6541–1.7208 versus gated YIN,
and 1.0213–2.1269 versus continuity. The supplemental intervals use 2,000 draws
with seed 20261022, resampling prompts within each fixed pair and retaining both
sexes and all conditions together. They condition on **these four speakers and
two pairs**. They are not participant-population confidence intervals, do not
justify a four-speaker generalization claim, and were not used to select a model.
Individual frames were not bootstrapped as independent samples.

## Browser arithmetic: preserved failure and one bounded repair

The original browser port used a double-precision FFT followed by float32 ACF,
energy and cumulative-difference intermediates. It passed all 4,420 analytic and
full-development frames, then failed **one of 10,812 reserved-input raw-Hz checks**.
In `F06_sx264__first_difference`, frame 666, librosa returns 50 Hz while that port
returns approximately 53.3834 Hz. The waveform window has RMS 0.000119705 and
p(voiced)=0.00392948, so both final outputs are unvoiced. That state agreement
does not excuse failure of the declared all-frame raw-Hz gate.

Independent numerical reconstruction identifies nearly tied global CMND minima
at lags 899 and 960: their order flips under different FFT rounding, with a CMND
margin of about 1.6e-7 and maximum ACF difference of about 1.8e-12. No single
indexing error was found. The [bounded repair plan](YIN_NUMERICAL_REPAIR_PLAN.md)
authorized exactly one revision: consistent float64 ACF, energy, difference and
normalization, preserving input float32 samples, equations, bounds, trough
threshold, interpolation and the hybrid rule. It forbids a tie epsilon, preferred
lag, unvoiced-frame exclusion, reference replacement or tolerance relaxation.

The revised implementation,
`phenometrix.raw-yin-librosa011-semantics.float64.ts.v2`, passes all **4,420**
analytic/full-development frames and **10,812** fixed reserved-input frames,
including raw YIN on every unvoiced frame. Maximum raw-Hz error against the
unchanged float32 Python references is 0.000611765 Hz and 0.000773748 Hz,
respectively. Both runs have zero parity failures and zero boundary exceptions.
The frozen tolerance remains `max(0.01 Hz, 1e-4*abs(Python Hz))`; model/features
and output-state checks also pass. Development sources are one full recording
per speaker; reserved fixtures are the first preselected source per reserved
speaker under all four conditions. These bounded fixtures are arithmetic checks,
not a second natural-accuracy sample or proof of equality on arbitrary inputs.

The original failed run remains separately identified. Because its input exposed
the numerical failure, the repaired reserved-input check is a regression after
a declared implementation repair, not an untouched implementation holdout.
Offline natural results above use the unchanged frozen Python estimator; the browser is a separately
versioned numerical implementation, not a recomputation or correction of those
natural scores. A real-worklet 300 Hz regression and both-stall functional replay
also pass. Functional replay is separate from sustained performance measurement.

## Runtime component

**All declared runtime components pass on this machine and synthetic replay.**
The complete local series ran from 20:43:15 to 21:17:27 UTC on 22 October 2026,
with 97 measured source/configuration bindings unchanged before and after every
run. The order was DSP, hybrid, hybrid, DSP for 60 s each, then a 30 min hybrid
soak. Hardware was Apple M4 Max with 128 GiB unified memory; the recorded browser
was HeadlessChrome 155.0.0.0, using single-threaded ONNX Runtime Web 1.23.2 WASM
and the original frozen checkpoint. No other agent computation ran concurrently.
Ordinary desktop load, power/thermal conditions and filesystem caches were
uncontrolled, so these are descriptive measurements rather than a controlled
hardware comparison. The inner replay results retain `timingBenchmark: false`
and `runPurpose: "sustained-functional-soak"`. The outer sequential experiment
adds exact source/configuration binding and the named component checks; it does
not reinterpret those functional results as a controlled hardware benchmark.

A real AudioBufferSource repeated the fixed two-second 200/400 Hz signal through
the AudioWorklet and worker. No microphone or speech recording was used for
runtime. Requested source duration and observed sample endpoints remain
separate. The soak's final retained frame ends at sample 86,400,448, or
1,800,009.333 ms: one frame is preserved in five-minute bin 6, beyond the nominal
30 min boundary. It is not truncated or retimed to make the requested duration
look exact. Every short run injected a 300 ms worker stall at 20 s and a 350 ms main
stall at 40 s; the soak used 600 s and 1,200 s. The stall handlers block their
respective event loops separately from a processed block's service timer. Their
transport gaps, stale results and freshness withholding remain counted below;
the service quantile is not total elapsed stall-inclusive latency.

Complete processed-block service includes shared DSP/features, raw YIN when
selected, model completion and posting result data. It excludes posting the final
block-completion message. The primary denominator is **every processed block**,
including blocks whose completed results are later discarded, rather than only
delivered frames. Freshness-withheld blocks have their own distributions. All
values below are milliseconds except counts and source duration.

| Run | Requested source s | Processed blocks | Complete p99 | Complete max | Blocks >10 ms | Blocks >20 ms | Certified age p99 / max | Legacy invalid frame timings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01-dsp | 60 | 2,970 | 3.8 | 6.8 | 0 | 0 | 49.2 / 50.7 | 0 |
| 02-hybrid | 60 | 2,970 | 4.9 | 10.4 | 1 | 0 | 49.1 / 52.1 | 43 |
| 03-hybrid | 60 | 2,969 | 5.1 | 10.8 | 1 | 0 | 47.3 / 50.1 | 0 |
| 04-dsp | 60 | 2,969 | 4.0 | 7.9 | 0 | 0 | 48.4 / 50.7 | 0 |
| 05-hybrid-soak | 1800 | 89,970 | 6.1 | 10.5 | 1 | 0 | 50.3 / 54.4 | 859 |

The two 60 s hybrid p99 values are 4.9 and 5.1 ms, versus DSP's 3.8 and 4.0 ms.
The soak's p99 is **6.1 ms**, below the frozen 10 ms threshold. Each hybrid run has
one block above the 10 ms service budget; the largest is 10.8 ms in the second
short hybrid run. All five runs have **zero 20 ms complete-block cadence misses**.
The soak's successive five-minute complete-block p99 values are 5.9, 5.9, 5.8,
6.3, 6.1 and 6.3 ms. These five-minute summaries include all admitted blocks,
including freshness-withheld records, while the primary table uses processed
blocks. The pass does not mean every block stayed below 10 ms.

Model initialization and isolated component timings are reported separately.
The inference/YIN columns below include **all completed predictions**, including
eight subsequently discarded results per run. Fresh worker initialization is
measured, but cold operating-system caches are not established.

| Hybrid run | Model/runtime load ms | Total setup ms | Completed predictions | Inference p50 / p99 / max ms | Raw YIN p50 / p99 / max ms |
|---|---:|---:|---:|---:|---:|
| 02-hybrid | 212.5 | 301.7 | 5,931 | 0.1 / 0.3 / 3.7 | 0.2 / 0.5 / 1.1 |
| 03-hybrid | 207.6 | 290.2 | 5,929 | 0.1 / 0.3 / 3.7 | 0.2 / 0.5 / 1.1 |
| 05-hybrid-soak | 213.4 | 292.9 | 179,931 | 0.0 / 0.3 / 3.8 | 0.2 / 0.7 / 3.6 |

A reported zero median inference interval in the soak reflects the browser's
available timer resolution, not zero computation cost. No timer-precision bound
was established. Model/runtime loading is excluded from warmed block service.
The complete path still computes the shared DSP baseline, so these measurements
do not estimate a hypothetical hybrid-only optimized worker.

### Frame accounting, drops and bounded state

The hybrid audit reconstructs each frame's identity, support, model/configuration
binding and voicing/raw-YIN combination from logged evidence. Prepared frames
reconcile exactly to delivered plus discarded plus cancelled frames:

| Hybrid run | Prepared/completed | Delivered | Discarded | Cancelled | Maximum queued async blocks |
|---|---:|---:|---:|---:|---:|
| 02-hybrid | 5,931 | 5,923 | 8 | 0 | 0 |
| 03-hybrid | 5,929 | 5,921 | 8 | 0 | 0 |
| 05-hybrid-soak | 179,931 | 179,923 | 8 | 0 | 0 |

Every delivered hybrid frame is voiced and within the declared 2 Hz diagnostic
tolerance of the known 200 Hz source. This uniform synthetic check does not
represent natural-speech accuracy. Eight completed frames per run are rejected
by the host because their freshness lease expired; they are retained in raw
telemetry and excluded from delivery. No research queue drops or cancellations
occurred in the measured series. Separate functional reset/cancellation tests
exercise those paths; absence of cancellations here is not proof of recovery
under every scheduling pattern.

| Run | Transport-dropped blocks | Result-backpressure blocks | Freshness-withheld blocks | Recorded lost blocks | Final transport-dropped samples | Partial samples at disposal | In-flight PCM / result snapshot |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01-dsp | 11 | 14 | 5 | 30 | 10,560 | 0 | 1 / 1 |
| 02-hybrid | 11 | 13 | 5 | 29 | 12,288 | 320 | 0 / 0 |
| 03-hybrid | 11 | 14 | 5 | 30 | 12,800 | 576 | 0 / 0 |
| 04-dsp | 11 | 14 | 5 | 30 | 11,264 | 704 | 0 / 0 |
| 05-hybrid-soak | 11 | 13 | 5 | 29 | 12,032 | 64 | 1 / 1 |

Loss counters describe different stages and are not interchangeable frame
counts. The final transport sample count includes the separately reported partial
buffer at disposal; the earlier worker snapshot may therefore contain fewer
dropped samples. All runs recover into three segments. All stop handshakes and
transport terminal acknowledgments complete. The displayed in-flight counts are
literal saved disposal/diagnostic snapshots: a value of one is not rewritten as
zero or interpreted as a failed handshake, and completion is not claimed to prove
that every earlier queue snapshot was empty.

Across every run, observed maxima are four in-flight PCM buffers, four result
batches, one pending freshness-lease request and eight lease-ledger entries.
The hybrid async queue maximum is zero, within its bound of three. The sample
ring has a fixed 96,000-sample capacity. Retained frame/block detail is capped at
120/80, numeric statistics at 200,000 per metric, and raw telemetry at 100 pending
records observed against a 200-record bound. These are bounded structures; a
whole-browser heap-growth measurement was not performed. Streamed telemetry is
stored separately: the soak retains 271,707 records in 875,600,614 bytes; all five
raw files total 964,201,369 bytes. Those files remain ignored and are not
public speech or reference data.

### Freshness and clock limits

Every delivered frame has a required same-main-clock freshness certificate. The
soak certifies all **179,923** delivered frames; its age-bound p99 is 50.3 ms and
maximum **54.4 ms**, within the 100 ms requirement. The certificate measures a
conservative interval from a main-clock lease issued before the block's first
observed input to main-clock result receipt. It concerns software-rendered input,
not physical microphone acquisition, upstream buffering or display latency.
It does not require a fitted audio-to-main clock mapping.

Legacy cross-clock diagnostics remain imperfect: 43 frames in the first short
hybrid run and 859 in the soak have unavailable derived arrival timing because
ready-to-main subtraction is slightly negative. The retained raw values are
approximately −0.0000978 ms and −0.0001955 ms, respectively; they are not clamped
into valid latency. Their same-clock freshness certificates remain independently
valid. The independent clock audit also finds **no single affine clock relation
consistent with every observed callback bracket in any of the five runs**.
Scheduling/phase variation is permitted by that result; removing outliers or
fitting a corrective mapping would not justify a physical latency claim.

The soak records 1,801 clock probes, with 1,800 render observations and one
stopped/unavailable observation. Descriptive residual slopes are about +0.5521,
+0.5544 and +0.5502 ms/minute for context reads, render-bracket midpoints and output
device timestamps. They are not calibrated clock-rate estimates or confidence
intervals. None is applied to the measurements. The output-device timestamp
coordinate does not establish capture availability.

Independent raw-summary and clock audits pass for all five runs; the hybrid
identity/decision/accounting audit also passes for all three hybrid runs. Each
audit is bound to the exact saved result/raw hashes, and the public publisher
checks the fixed run order, durations, stalls, source stability and these bindings
before reporting a component pass. This qualifies the named research runtime
component on this local replay, not arbitrary hardware, speech, clinical use or
production promotion.

## Independent audit and identities

The separate audit verifies all 24 frozen source bindings, 160 source identities,
640 condition records and 477,536 acquisition/target alignments. It independently
reconstructs **3,200 recording/method count comparisons**, all aggregates, both
pair effects, the quality gate and the paired-prompt intervals exactly. Serialized
outer command starts and saved logs corroborate that prediction freeze precedes
reference acquisition. They do not constitute an operating-system access log or
network first-byte trace. No fitting or new inference is performed by this audit.

| Artifact | SHA-256 |
|---|---|
| Pre-access metadata audit | `812b6097d9eafb60b8104a09dc87f01bdebe0dce14bddb68641aab8c7892a2bf` |
| Frozen natural protocol/config/source identity | `c5c4288cc04c0d7fb46154977627158336e2b1a5b8b496f8700c7a96dd25c67d` |
| Complete natural prediction freeze | `002a05e1d6c8404d925efd33bd9f40e150e07aa3aa51447c700423487cd7f917` |
| Natural metrics | `cfdf88d02977ee6702696726380b337f6ac1ca762bc7193cbe08040ff4fd731d` |
| Independent natural audit | `c0369dc79539e4bb02089230fa1f17ad4b05424b100f451bb5d918a58d8f3970` |
| Repaired analytic/development browser parity | `96d9f602b5be8f5f57c305bb16cff52b7eb2f3c1da42fc85b122edd8b3f35527` |
| Repaired reserved-input browser parity | `f7caa94d726cb0b209099abc3ae7d5e763090b44a2f4304dbecf88daccb86748` |
| Complete five-run runtime manifest | `d7404dd13a786243d158693befd670cdf9604d17ad82a540944e7b555b4db486` |
| Hybrid soak result | `d9e78ca35ca76e94bbde30f9d32ba763c3ce49b42ef3c2008a68ae32ad1a97fa` |
| Hybrid soak raw telemetry audit | `1540bbbc0033a41a6c1cb4377cebfcb26ef0669cbc9c31358952ebedd830966c` |
| Hybrid soak clock audit | `5d4eed465eb37c32079aa11b86852e7a385c31f5431f98c0e90ff48ac96efcc0` |
| Hybrid soak identity/accounting audit | `d18c478aee5c97ca9fa382491ae2074f920bdd4b11427b4e859c027ed16663ad` |

The frozen base commit is `a0de18b520eb02f8eee787f36a826a5c577539dc`; the freeze
explicitly identifies the base commit plus exact uncommitted source hashes and
snapshots, rather than falsely claiming the new runner was already committed.
The [public evidence summary](../../benchmarks/voice-prosody/hybrid-validation-summary.json)
contains aggregate/count evidence and source identities. Reproduction commands
are in the [research README](README.md#independent-reserved-speaker-and-browser-validation).
No raw corpus media, reference trajectories, private weights or frame-level
predictions are published. All earlier failed gates remain failed, raw voicing
probability remains uncalibrated, and no production or clinical promotion is made.
