# Prespecified voicing robustness experiment v1

Status: frozen prospective design; no new fit or outcome yet. The user supplied
the design on October 24, 2026. All PTDB results in this study are development
and regression evidence. Production remains DSP. Original failed experiments,
source snapshots and artifacts remain immutable. The entire provided fixed
mechanism, exposure, diagnostics, metric and gate contract follows below.

Execution uses the existing Python 3.11 environment and installed libraries,
MPS only for the six fits, one fitting/measurement owner, no native builds or
environment sync. New run folders are ignored, create-new, and retain exceptions.
Raw audio, references, features, weights, predictions and traces stay local.

The bank namespace is `phenometrix-voicing-robustness-v1-20261025`. Draw seeds
use the first eight bytes of SHA-256 over UTF-8 `namespace:source_id:slot`,
little endian, passed to NumPy PCG64. Parameters and noise use separate domains.
Draw order is gain, family (combined only), filter coefficient, noise family and
SNR (combined only). Events use a separate `:events` namespace.

The explicit phases are `inventory`, `freeze`, `prepare`, and `fit` in
`phenometrix_prosody.voicing_robustness`. Preparation binds all anchor views,
transformed PCM/features, TRAIN-only references/labels, matched slot indices,
and every epoch's batch/chunk/skip inventory before training. Each seed starts
from random parameters paired by tensor-content hash. Scaling buffers alone
come from the authenticated original checkpoint. The unchanged exporter can
read the genuine training schema, with augmentation provenance retained in
the new training manifest; export remains contingent on passed development.

Exact update counts are derived from 1,920 matched sequence slots and actual
valid masks, never borrowed from earlier studies. Scientific fits use 12 epochs,
64 sequences/batch, 128-frame chunks, Adam 0.001, gradient clipping 1.

Resource bounds: 4 GiB new study artifacts and at least 15 GiB free disk;
only the existing environment is reused. Source/recipe freeze precedes bank
construction; a second final training/audit source freeze precedes the fits.
Historical sources authenticate their historical artifacts; live browser edits
receive new identities and cannot inherit old runtime qualifications.

## 5. First-principles hypothesis and fixed inference contract

Voicing is evidence of periodic excitation in speech. Multiplying a waveform by a gain or changing its spectral envelope can alter absolute energy and feature magnitudes while preserving useful periodic structure. Noise can instead obscure that structure. A useful model must distinguish nuisance changes from genuine loss of evidence; it cannot recover information that is absent.

The hypothesis is that **continuous, label-independent acoustic variation during TRAIN exposure improves the learned voicing rule's robustness**, without simply making it call more frames voiced. The false-positive and transition constraints are central to this hypothesis.

Use the existing hybrid rule unchanged:

```text
p(voiced) < 0.5  → unvoiced, F0=null
otherwise       → raw YIN pitch if finite and within inclusive 50–700 Hz
otherwise       → unavailable, F0=null
```

Keep 57 v2 features, the six-candidate feature representation, 32-unit projection/hidden layers, the existing MLP export interface, raw YIN equations, 0.1 trough threshold, and the fixed voicing threshold. Candidate logits remain diagnostic and do not select the delivered pitch. Voicing scores remain uncalibrated; they are not probabilities that a reported F0 is correct.

At 48 kHz, keep 1,920-sample support, 480-sample output hop, and 960-sample declared measurement delay. A window ending at `e` describes target `t=e−960`; it is available no earlier than `e`. Thus inference is causal relative to arriving PCM but uses **20 ms of future support relative to its reported target**. Do not call it zero-lookahead or add that delay ambiguously to an unrelated clock metric.

Before fitting, quantify the existing errors using consumed PTDB references: strict voiced→unvoiced decisions, voiced-but-invalid YIN, voiced pitch errors, correct outputs, and unvoiced false positives. Compute an explicitly labeled oracle upper bound with unchanged raw YIN and reference voicing, solely to show available headroom. If fixed-YIN coverage makes the intended improvement impossible, stop this intervention rather than silently changing pitch estimation.

For each design choice record the mechanism, predicted effect, smallest falsifying test, result, and keep/reject decision. Technical insight means explaining why an intervention helped or failed, not adding more components.

## 6. Data roles and the matched training experiment

### Exposure and partitions

All 20 PTDB speakers have now been observed. Retain the historical TRAIN membership and do not add development/formerly reserved speakers to fitting. New outcomes on PTDB are development/regression evidence. Repeated utterances, corruptions, crops, and seeds do not create new speakers.

The existing TRAIN cache has **240 sources from 12 speakers**, four conditions each, totaling 960 views and 676,384 feature frames. The conditions already include clean speech, 10 dB noise, first difference, and clipping. The original development cache has 320 views from 80 sources/four speakers. The later reserved evaluation has 640 views from 160 sources/four speakers. Preserve those exact selected sources and their acquisition records.

Use the eight non-TRAIN speakers for the new development/regression table: pool recordings within speaker/condition, then weight speakers and conditions equally, so the different prompt counts do not dominate. Keep the original four-speaker development and later four-speaker results separately visible. A new adapter may explicitly admit the now-exposed reserved records; do not weaken an old sealed-split loader or pretend the old access chronology changed.

### Six fits, no search

Use optimization seeds **20261025, 20261026, 20261027**, each with control C and robustness treatment R. There are at most six completed scientific fits. Designate **R/20261025** as the only candidate for deployment before any new outcomes. No best-seed, best-epoch, or ensemble substitution.

| Property | Control C | Treatment R |
|---|---|---|
| Architecture and inference | Existing framewise MLP; fixed hybrid output | Identical |
| Per-source exposure | Original four conditions, each twice | Original four conditions plus four continuous acoustic views |
| Total view slots | Eight per source, 1,920 total | Same |
| Initialization within a seed | Same initial parameter tensors | Same |
| Scaling | Frozen original TRAIN feature mean/scale | Identical buffers |
| Optimization and labels | Same schedule and objective | Same schedule and objective |

Copy the verified original `feature_mean` and `feature_scale` buffers and set `scaling_fitted=True` with their authenticated provenance; new models initialize that flag false. These buffers were fitted on all four original TRAIN conditions, not clean-only speech. Do not call `fit_scaling()` or refit on treatment/development/external recordings. Both new arms start from matched random initializations rather than the old learned weights; the old hybrid remains a separate frozen comparator.

Retain the existing objective: unweighted voicing BCE on all valid reference frames plus candidate cross-entropy only where a correct candidate exists within 50 cents. The auxiliary candidate loss can affect the shared representation even though its output is not deployed. Report changed covered-candidate counts; do not claim this isolates a voicing-head-only weight change. Holding the entire objective fixed makes C→R the comparison of acoustic exposure.

Use 12 epochs, Adam learning rate 0.001, batch size 64, chunk length 128, gradient norm clipping 1, final epoch only. Pair source/view slots, lengths, permutations, initialization RNG, and update inventory across arms. Precompute and hash schedules; preserve padding/validity masks and skip rules. **Do not borrow 396 updates from the unrelated exposure-selector study.** Derive the exact update count from this matched inventory and assert equality. Log all elapsed time, failures, and complete/incomplete fits.

### Fixed continuous TRAIN bank

Use one immutable augmentation bank shared across all three seed pairs, with randomness independent of optimization. Derive each draw from SHA-256 of a frozen study namespace, source ID, and view slot; record the seed conversion convention. Freeze the bank description before preparation, then hash all transformed PCM/features before fitting.

Four new views per source:

1. Gain only: gain in dB sampled uniformly from −18 to 0.
2. Gain plus causal high-frequency emphasis: the same gain distribution, with `y[n]=(x[n]−a*x[n−1])/(1+a)`, `a` uniform in [0.2, 1.0].
3. Gain plus causal low-pass: the same gain distribution; cutoff log-uniform from 1.5 to 10 kHz; `b=exp(−2*pi*fc/48000)`, `y[n]=(1−b)*x[n]+b*y[n−1]`.
4. Combined view: gain, one of the two filter families with fixed 50/50 selection, and additive white or causal AR(1) noise with fixed 50/50 selection; SNR uniform from 5 to 25 dB. AR coefficient is fixed at 0.8.

Apply channel filter, then gain, then noise. Reset filter/noise state at each source boundary; retain complete startup behavior and the original common scoring mask. Never filter separate overlapping windows independently or use zero-phase/future-dependent filters in the estimator.

Generate white noise from independent standard-normal samples. For AR noise use `z[n]=0.8*z[n−1]+sqrt(1−0.8^2)*epsilon[n]`, starting at `z[−1]=0`. Scale the complete realized noise sequence to the requested SNR using power of the post-filter/post-gain signal **before** noise addition, without any reference/voicing mask. If signal power is exactly zero, add zero noise and record achieved SNR as undefined. This whole-recording scaling is offline dataset construction, not causal runtime normalization.

Compute construction arithmetic in Float64 and cast final PCM to Float32. Add no implicit saturation or peak normalization; clipping remains its explicit historical anchor condition. Reject nonfinite construction, retain values outside ±1 and count them, and bind the float-PCM convention. Store requested/achieved SNR, coefficients, sample count, extrema, and PCM digest. Prefix/chunk invariance tests hold the realized noise and offline scale fixed; they must not recompute whole-recording RMS from each prefix. Parameters must not depend on speaker sex, pitch, voiced labels, model outputs, or difficulty.

Generate input-derived features from the transformed PCM using the actual TypeScript extractor. Do not perturb cached feature vectors as a substitute. Reuse identical anchor views rather than duplicating large caches. This single compound treatment tests robust training exposure; it does not isolate which augmentation component caused a change.

## 7. Development metrics, gates, and stop rules

The primary shifted conditions are **the existing noise10db, first_difference, and clip01 conditions**, evaluated across all eight exposed non-TRAIN speakers. Clean speech is a separate guardrail. These definitions cannot change after fitting. The extreme M05 stratum is a required regression row, not a selected success sample.

Also freeze exactly six standalone diagnostic views before fitting: gain −12 dB; gain −3 dB; normalized emphasis with coefficient 0.5; normalized emphasis with coefficient 0.95; white noise at 8 dB SNR; AR(1) noise at 18 dB SNR. Apply each to the original clean source, without another transformation. Use independent source-seeded noise. Report these as diagnostic generalization across acoustic settings, not independent speakers and not a substitute for the primary gate.

For each seed, speaker, condition, and comparator report counts and:

- Joint error: wrong voicing, unavailable output, or voiced pitch error above 50 cents, over all eligible valid references.
- Strict voicing false negatives, unavailable-on-voiced rate, and their sum over all reference-voiced frames.
- False positives over all reference-unvoiced frames.
- Voiced success within 50 cents over all reference-voiced frames, including missed/unavailable outputs.
- Octave error, coverage, and conditional pitch error, with their different denominators identified.

Pool counts within each speaker/condition and then macro-average. Average seed scores, not predictions. Publish every seed, worst speaker/condition, negative result, undefined denominator, and out-of-range reference count. Do not omit reference F0 outside 50–700 Hz from the primary denominator.

**New study gates**, fixed before fits and separate from every historical gate:

| Gate | Requirement |
|---|---|
| Shifted voicing robustness | R's three-seed mean false-negative-or-unavailable rate is at most 0.8 times both C's three-seed mean and the frozen hybrid's rate; each R seed must also avoid regression against its paired C and the frozen hybrid |
| Exported artifact's robustness | Predetermined R/20261025 independently meets the same 20% relative reduction against C/20261025 and the frozen hybrid |
| False-positive safety | Each R seed increases macro unvoiced false positives by at most 1 percentage point versus both comparators; no speaker/condition increase above 2 points |
| Clean speech | Each R seed's clean joint error rises by at most 0.5 points and clean voiced success falls by at most 1 point versus both comparators |
| Overall utility | Three-seed mean unconditional joint error is strictly lower than both matched control and frozen hybrid; R/20261025 must also improve individually against C/20261025 and the frozen hybrid |
| Diagnosed stratum | No R seed worsens M05 first-difference false-negative-or-unavailable rate by more than 2 points versus either comparator |
| Analytic robustness | Every R seed passes the unchanged applicable hybrid analytic and transition requirements |

Relative improvement is undefined when the baseline rate is zero; do not manufacture a ratio. An undefined required gate is unassessable, not a pass. Report absolute percentage-point changes alongside relative percentages. These are practical point-estimate gates, not statistical evidence of a population-wide effect.

Reuse the original hybrid analytic contracts: each required stationary/postchange case reaches 95% voiced success; challenge macro voiced success at least 90%, every voiced challenge case at least 80%, and false voicing at most 5% in each unvoiced case. Preserve all original pitch-transition definitions and the combined/step/glide extra-median target-delay limit of 10 ms versus delayed DSP. Existing cases are regression fixtures now, not fresh scientific samples.

The old transition tests cover pitch steps/glides, not voicing onset/offset. Add a small fixed **new** voicing-event fixture set before fits: 3-second signals with a harmonic segment only on [1,2) seconds; fundamental 90/180/360 Hz, common phase 0/0.7 radians, and either silence or fixed white-noise background (12 cases). Use harmonic amplitudes 0.1/0.05/0.025; noise standard deviation 0.003 and source-seeded realizations independent of training. Keep these cases out of training. Search within [event−20 ms, event+200 ms] for the first of three consecutive target-grid predictions in the required state: `voiced` for onset, `unvoiced` for offset. `Unavailable` does not count as an offset. A missing event is a failure, never dropped from the delay summary.

Every R seed must detect both events in each case. On cases with defined baseline events, its median signed onset delay and median signed offset delay may each exceed the original hybrid by at most 10 ms; report every per-case delay. If the original hybrid misses an event, show that separately without manufacturing a relative delay. Retain the search-window bound as the absolute detection limit. Report false voicing in the fixed pre/post-event plateaux [0.2,0.8] and [2.2,2.8] seconds, and missed voicing on [1.2,1.8]; apply the 5% unvoiced and 95% voiced-success limits there. This is a new prespecified regression requirement, not a historical gate. Keep signed target lag and earliest-availability delay separate.

Compare R/C/frozen hybrid with the original candidate MLP, DSP, gated YIN, and continuity in the result table. The candidate MLP's lower natural joint error must remain visible even though its pitch robustness failed. No voicing-threshold sweep, post hoc frequency exception, data expansion, or second augmentation bank follows failure.

**If development fails, stop fitting and leave KEELE outcome data unused.** Complete the natural-browser work with the original hybrid and publish the failed controlled experiment. The project already has a positive application result; this sprint does not require a new positive research result.

## 8. External corpus intake: KEELE, conditionally

KEELE is the proposed external corpus; no KEELE adapter or authenticated local acquisition was found in this checkout. Spend at most approximately half a day establishing admissible provenance, rights, reference interpretation, and timing. Do not let acquisition become a second project.

The [versioned Zenodo archive](https://zenodo.org/records/3921794) offers `KEELE.zip` separately from `KEELE_mod.zip`; the former is listed as 22.8 MB with MD5 `f5a87014bad14744660b90187de7d43f`. Use the unmodified-named archive and inspect retained release documentation; the redistribution is a converted package, not an assumption of byte identity with the original files. The record describes noncommercial use. Retain applicable terms, attribution, download receipt, publisher checksum, local SHA-256, extraction inventory, and converter identity. Download only the needed archive, not the 15.8 GB collection. Do not execute bundled code or install its legacy PDA package.

The [original paper](https://www.isca-archive.org/eurospeech_1995/plante95b_eurospeech.html) describes a 15-speaker core, while the archive preview lists ten recordings. Report the actual authenticated speakers/files; do not copy a literature count into the experiment. The [archive's conversion source](https://github.com/bastibe/Replication-Dataset-Scripts/blob/master/keele.py) reads 20 kHz audio and constructs a 100 Hz reference time axis. Pin the inspected source version and verify release metadata before using these conventions.

[Wang's KEELE evaluation, §2.5.1/Table 2-1](https://sls.csail.mit.edu/archives/root/publications/2001/Wang_phd_thesis.pdf) distinguishes clearly voiced values, clearly unvoiced zero, and uncertain negative values/−1. Verify the selected package's actual representation: conversion from period to frequency can change a negative sentinel's magnitude. Preserve uncertain/invalid categories and their counts; do not map every nonpositive number to unvoiced, take absolute values, or interpolate across uncertainty. Uncertain transitions limit what this corpus can establish about onset timing.

Audit prior acquisition records and filenames for previous KEELE use. Before the final model/rule freeze, allow only documentation, archive/file metadata and hashes, and synthetic adapter fixtures. Defer real waveform inspection/listening, reference values, feature summaries, and model-dependent outputs. Record any unavoidable constrained schema exposure explicitly. Hashing an archive is distinct from viewing its contents or scoring models; record each access stage honestly. If the corpus was already examined, label that exposure and do not claim a fresh external test. If terms, codes, or alignment cannot be established, retain an explicit external-validation block and finish the useful PTDB/browser work. Do not substitute more PTDB speech as external evidence.

### Preparation and exact time mapping

Keep original microphone and reference bytes immutable. Convert the entire microphone recording once to canonical mono float32 48 kHz using one pinned deterministic rational resampling implementation, recording sample count, endpoint/padding conventions, coefficients/settings, and hashes. Prove sample-zero anchoring with synthetic impulse/index tests. Treat this as **offline preparation** shared by every method; browser causality/latency claims begin at canonical 48 kHz PCM. Do not describe this as a measured causal 20→48 kHz device pipeline.

For a verified native reference grid `t=0+10*k ms`, use a new corpus adapter:

```text
canonical target sample = 480*k
support end              = target + 960
support start            = support end − 1920
earliest availability    = support end
```

The first complete window ends at 40 ms and targets 20 ms. With the existing common mask `t>=64 ms` and `t+32 ms<=duration`, the first eligible grid target is 70 ms. Join on exact target sample, **not** the replay endpoint's `referenceIndex`, which is two hops ahead.

`CausalVoiceReplay` already supports `referenceOriginMs: 0`; the real worker's native-grid test uses that setting. Its existing export CLI does not expose the option, and `measurement_alignment.py` hardcodes PTDB's 16 ms phase. Add an explicit KEELE adapter/CLI route and keep the PTDB helper intact. Do not borrow PTDB timing, fit a lag by accuracy, choose nearest labels, or change the 40/10/20 ms contract. If release documentation contradicts the proposed grid, resolve it before external scoring and record the source-based protocol amendment.

### External freeze and assessment

Freeze all six final checkpoints, normalization, thresholds, YIN, source/configuration, archive inventory, transforms, scoring rules, and R/20261025 export designation before external audio predictions/outcomes. The external primary set is every authenticated original recording under the four original conditions. No tuning, calibration, speaker exclusions, best-seed choice, or selecting favorable crops.

Complete and hash all predictions before opening reference arrays in the scorer. The archive may physically contain both; that is not evidence they were acquired separately. Enforce feature-only inference by interface and log reference access. Evaluate all six new hybrids and frozen comparators. Compute the same counts and equal-speaker/condition macros, with seed-paired differences. Apply the same substantive robustness/safety/clean/joint gates to external data, excluding the PTDB-specific M05 gate and retaining its prior result. Report clean-only and shifted results separately; do not claim that synthetic shifts are real different microphones.

Provide paired speaker-level uncertainty summaries, for example 2,000 bootstrap draws with a predeclared seed, retaining each sampled speaker's conditions and model pairing. With a small corpus these intervals are descriptive; frames and seeds are not additional independent participants. Publish undefined cells and missing files. A failed external gate ends advancement. No second corpus search or new fit follows the external outcome in this sprint.


## Execution and falsification ledger

| Choice | Mechanism/prediction | Smallest falsifier | Decision |
| --- | --- | --- | --- |
| Fixed raw YIN and threshold | Isolate acoustic exposure while preserving inference | Reference-voicing oracle cannot meet intended headroom | Pending baseline audit |
| Matched C/R random initialization and scaling | Remove optimizer budget and normalization confounding | Any paired tensor/schedule/valid-mask mismatch | Reject mismatched attempt |
| Continuous TRAIN channel bank | Avoid dependence on anchor condition magnitudes | Strict FN reduction fails or raises FP/clean errors beyond gates | Pending six fits |
| Causal channel filters | Preserve online support semantics | Prefix/chunk recurrence mismatch with fixed realized noise scale | Covered by synthetic tests |
| KEELE nominal zero-phase adapter | Separate corpus coordinates from PTDB | Provider contract or exact grid cannot be verified | See intake receipt; blocked corpus remains unopened |
| Finite browser replay | Verify actual transport and delivered denominator | Origin/segment/terminal or acoustic parity audit mismatch | Fail browser quality; retain diagnostic accounting |

Run from `research/voice-prosody` with the existing interpreter:

```sh
.venv/bin/python -m phenometrix_prosody.voicing_robustness freeze --output runs/voicing-robustness-v1/plan
.venv/bin/python -m phenometrix_prosody.voicing_robustness prepare --plan runs/voicing-robustness-v1/plan
.venv/bin/python -m phenometrix_prosody.voicing_robustness fit --plan runs/voicing-robustness-v1/plan
```

No external prediction/scoring may run without successful exact-model development
and intake receipts. A blocked reference contract is not an external negative
result. Natural PTDB browser replay uses first metadata-ordered complete source
per eight non-TRAIN speakers, clean/first-difference, fixed 192-zero prefix,
source-zero alignment and actual resets. H is replayed twice on development
failure. All status dimensions remain separate.
