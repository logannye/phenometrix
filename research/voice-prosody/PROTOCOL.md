# Causal voice estimation protocol

Current disposition: no learned estimator qualified for promotion; the production
DSP remains in place. See [RESULTS.md](RESULTS.md) for the final evidence.
This file preserves the experiment decisions in order. The opening sections are
the original pre-fit declarations; later sections explicitly declare the separate
20 ms task, bounded augmentation, and final no-go. Historical endpoint gates are
not retroactively passed by results under a different timing contract.

## Question and mechanism

Can a small causal model use present candidate ambiguity and recent acoustic evidence to
reject observation-induced octave changes without flattening real source changes, and
recognize when the signal does not support an F0 measurement? A microphone measures a
filtered pressure signal plus noise. Harmonics need not identify the fundamental; short
windows contain few low-frequency cycles. Temporal trust must therefore depend on present
evidence. This is a hypothesis, not an established benefit or novelty claim.

## Data roles and seal

PTDB-TUG microphone PCM alone is the inference input. Laryngograph, reference F0,
reference validity, eligibility and scorer labels never enter prediction or state.
The provider's reference is algorithmically derived from the independent laryngograph,
not infallible manual truth. Preserve ODbL/DbCL notices separately from repository MIT.
The corpus report documents missing F08 si1912/si1913; audit the actual inventory.

Seed 20261022, Python Random.shuffle of pair IDs 1..10: train [4,8,7,9,10,2],
development [1,3], sealed test [5,6]. Matching F/M pair numbers share sentence material.
Exclude sa1/sa2 from primary evaluation. Audit all remaining prompt intersections.
Every transformation/crop stays with its source. First acquisition is training-only.
No sealed audio, outcomes or threshold selection before configuration/model freeze.

## Timing and boundaries

Provider reference grid is provisionally 16 + 10*k ms (32 ms ESPS window, 10 ms hop).
Confirm actual lengths/columns and paired MIC/LAR alignment. Reference high-pass group
delay compensation is undocumented; apply no invented shift. One deterministic mapping
is used across recordings; no per-recording alignment optimization.
At common reference time t, causal support is [t-40 ms,t), last required sample t-1;
centered CREPE support is [t-32 ms,t+32 ms). Shared boundary eligibility requires both.
Reference time and input availability are distinct. Production uses native 40+10*k ms;
legacy tMs remains window start. Irregular input chunks may not change predictions.

## Primary comparisons and metrics

Preserve voice-analysis-1.1 arithmetic. Compare causal librosa YIN 0.11.0 (center=False,
40 ms frame/10 ms hop, disclosed voicing gate), torchcrepe tiny pinned to
19e2ec3d494c0797a5ff2a11408ec5838fba6681 (weighted argmax, no sequence decoder),
fixed continuity, framewise MLP ablation, and at most four GRU32-family configurations.
Bounds 50-700 Hz. Out-of-range reference F0 remains in unconditional denominators.
Candidate oracle coverage precedes training. Permit one evidence-driven candidate revision.

Primary metric: joint voicing/F0 error at 50 cents on valid, common-support frames.
Abstention is a failure, including on unvoiced frames. Also report voicing FP/FN with
reference-class denominators, reference-voiced success, jointly voiced cents/coverage,
octave errors (absolute signed cents within 50 cents of a nonzero integer octave),
raw versus qualified coverage, and all speaker/condition values. A voiced observation
without supported F0 is unavailable, never unvoiced. No averaging octave candidates.
Condition headline weights, if used, are equal per speaker then condition. Matched pairs
are the dependence groups; thousands of frames do not imply thousands of participants.

Initial declared condition set: clean, seeded additive white noise at 10 dB mixture SNR,
causal first difference channel filtering, and symmetric clipping at 0.1 peak amplitude.
Use whole clean-recording mean-square only to construct offline corruption, never as
inference normalization. Save requested/achieved SNR and split-specific seeds. These
synthetic stresses do not establish real-environment robustness. Levels may be revised
once on development evidence before freeze; every change must be recorded here.

Before training, record actual failure frequency/trace, mechanism, predicted benefit and
failure case, discriminating experiment, and keep rule. Proposed keep rule (not frozen):
lower development joint error than best simple comparator with no >2 percentage-point
loss of reference-voiced coverage, no >20 ms extra median real-transition lag, and p99
complete-worker service under 5 ms on this M4 Max. Reject a learned model that fails.
No final test scoring until alignment, model, feature, calibration, masks and rules freeze.

## Execution and evidence

All workloads local per user instruction: Apple M4 Max, 40 GPU cores, 128 GiB unified RAM.
Use PyTorch MPS for eligible training; record unsupported operations and CPU reference
work explicitly. Browser initial target is installed Chrome, dedicated worker, single
threaded WASM. CPU/WASM deployment is measured separately from GPU training.
One research environment, one performance process at a time, initial new data/artifact
budget 1 GiB; inspect before expanding to provisional 12 GiB full-corpus budget.
No microphone activation without application consent, no media/labels in observations,
no WavLM checkpoint, no paid compute. Synthetic replay is labeled replay.

G1 exit requires actual reference alignment/inventory, baseline errors and candidate
coverage, corrected clock contracts, and an executable export spike. G2-G5 remain open
until the trained mechanism, natural-speech evidence, browser parity/overload, frozen
held-out benchmark, regression checks and final report exist. A successful synthetic
smoke test does not close those gates.

## Recorded hypothesis revision before first fit

Initial training-only K3 oracle coverage was0.52457 speaker-macro. The error histogram
contains many50–200cent errors and comparatively few isolated octave errors. On3,252
reference-voiced frames with preceding valid voiced history, coverage against the
instantaneous target was0.5883 but against the preceding four-label median was0.8828.
This second calculation is a scorer-only diagnostic; it does not change the target,
features, training labels, or primary score. Actual examples include F02_si645 at
2816ms: reference260.16Hz versus candidate250Hz, and2886ms:289.86Hz versus275.86Hz.
The evidence suggests trailing-window response lag and quantization are more material
than the originally proposed isolated-octave mechanism in this small clean pilot.

The single permitted evidence revision retains the three legacy hypotheses and adds
one refined40ms ACF peak plus two refined20ms suffix ACF peaks. The short support only
covers100–700Hz (at least two nominal cycles);40ms retains50Hz support. Selection of
peaks uses acquisition evidence only, deterministic rank/order and bounded parabolic
lag refinement. V2 has6candidate slots/57features with separate support evidence.
Before seeing the v2 result, the prerequisite to fitting is75% speaker-macro candidate
coverage on the same training pilot. No second candidate-generation revision is planned.
The revised learned hypothesis is evidence-conditioned use of present local support
and history to improve endpoint tracking without flattening real transitions.

The fixed nominal provider grid is retained with explicit alignment uncertainty.
Raw laryngograph cycle probes support some stable-period labels; unknown high-pass
compensation and transition correspondence remain limitations. Do not represent them
as resolved, and do not fit per-recording shifts or remove difficult transitions.

## Follow-up: explicitly delayed estimation (declared before fitting)

The original zero-lookahead v2 experiment failed its 75% candidate-coverage
prerequisite (64.57% on the 24-recording training pilot). That result remains a
no-go under its original contract. The user subsequently requested continued
investigation. This follow-up changes the declared availability contract; it
does not retrospectively pass the failed gate or change the reference clock.

Known-truth pitch ramps independently showed approximately 20 ms of response lag
for the trailing 40 ms DSP and YIN windows. Before examining the corresponding
natural-speech result, a single fixed 20 ms measurement-delay arm was selected.
At reference time t, it uses microphone support [t-20 ms,t+20 ms), with earliest
availability t+20 ms. Recurrent state can depend on earlier microphone frames.
This is streaming with 20 ms future support relative to its reported target,
not zero-lookahead prediction. No per-recording shift or delay sweep is allowed.
The nominal provider reference grid remains 16+10*k ms. A training-only raw-LAR
cycle audit supports nominal centered timing on a selected strong-cycle subset;
its filtering/selection and unresolved reference limitations remain reported.

On the same 24 training recordings and common target mask t>=64 ms and
 t+32 ms<=recording length, delayed v2 oracle coverage is 88.21%, compared with
64.57% at zero delay. Delayed DSP joint error is 9.30% versus 15.77% at zero delay;
delayed YIN is 9.60% versus 17.21%. Centered CREPE at 32 ms lookahead is 7.28%.
These exploratory results justify a new bounded experiment, not a claim of
neural improvement, held-out generalization, or a runnable oracle.

The experiment uses the already deterministically selected first 20 recordings
per speaker: 240 training recordings from six matched pairs and 80 development
recordings from two distinct pairs. Test content and outcomes remain sealed.
Every source has the four previously declared conditions: clean, additive
white noise at 10 dB, causal first difference, and clipping at amplitude 0.1.
Offline corruption construction uses microphone data only; noise seeds derive
from seed 20261022, split, source identity and condition. No transformed source
crosses splits. Features and baseline predictions are frozen before any scorer
loads reference content. The complete acquisition manifest and exact feature
source snapshot are retained locally.

One GRU32 with projection32 and one framewise MLP32 with the same 57 features
will be fitted on local MPS, using seed 20261022, Adam learning rate 0.001,
12 fixed epochs, recording batches of 64, contiguous TBPTT chunks of 128 frames,
gradient norm clip 1, and train-only standardization. Reset state at each
recording; carry detached state across chunks. Do not use reference history.
Voicing uses unweighted binary cross entropy; conditional candidate cross
entropy is evaluated only where a valid candidate is within 50 cents. Uncovered
voiced frames retain positive voicing supervision. Padding and invalid/boundary
labels have no loss; entirely unsupervised chunks make no optimizer update.
No epoch selection, architecture search, or candidate revision is planned.

The fixed continuity comparator uses candidate maximum evidence score minus
0.25*min(abs(cents(candidate/previous accepted candidate))/1200,1). Voicing
requires RMS>=0.003 and maximum valid candidate evidence>=0.55; ties choose the
lowest sorted frequency. Reset previous pitch on unvoiced/unavailable frames
and at source boundaries. Raw neural voicing threshold is 0.5. Voiced frames
with no candidate are unavailable. DSP and YIN preserve their original gates
and receive the same 20 ms measurement delay.

The development keep rule is frozen before fitting: the GRU must lower equal-
speaker, equal-condition joint error by at least 1 absolute percentage point
against EACH of DSP, YIN, fixed continuity and the framewise MLP, while losing
no more than 2 percentage points of reference-voiced success against the best
simple comparator. It must also add no more than 10 ms median transition lag
on the predeclared known-truth steps/glides relative to delayed DSP. Clean
joint error must not worsen by more than 1 percentage point versus the best
simple comparator. All conditions and speaker results must be reported.
If this gate fails, retain the strongest supported simple estimator and stop
architectural search. The 20 ms delay itself is reported separately from
processing latency and is never presented as a neural gain.

Only a retained model proceeds to calibration (one logistic method and fixed
thresholds 0,0.5,0.8), three-seed confirmation if measured runtime permits,
actual browser integration, and sealed-test evaluation after a final freeze.
The runtime gate remains p99 complete worker service below 5 ms per 10 ms hop
(or 10 ms for a two-hop transport block), with explicit drop/coverage accounting.

### Alignment decision for the bounded follow-up

The second raw-cycle audit and source-level frame-count reconstruction are
sufficient to retain one nominal mapping for this exploratory comparison
against the provider's algorithmic reference. They do not establish pointwise
physiological ground truth or resolve the provider's exact filtering history.
Training may proceed under that explicit limitation; production qualification,
clinical interpretation, and claims about weak/irregular phonation remain
unsupported. No clock correction, reference repair, transition exclusion, or
record-specific alignment is introduced to improve model scores. The complete
alignment report remains part of the evidence required for a final freeze.

## Final bounded augmentation trial (before fitting)

The first development comparison used four previously unseen speakers (two
matched pairs), 80 sources and four conditions. Equal-speaker/equal-condition
joint errors were DSP 10.61%, YIN 11.02%, fixed continuity 11.60%, GRU 7.95% and
MLP 8.63%. The GRU failed its predeclared 1-point advantage over the MLP
(observed 0.68 points). The MLP passed its separately pre-score-declared natural
quality components. Centered CREPE, with 32 ms rather than 20 ms future support,
scored 9.56% under the same mask. These are development results, not sealed-test
estimates, and all condition/speaker values remain visible.

Both trained models then failed the fixed constant-300-Hz analytic case despite
correct candidates: MLP chose approximately 150 Hz, GRU approximately 100 Hz.
DSP and YIN succeeded. The MLP also failed the 300 Hz harmonic-change case.
The original models are therefore not qualified for production. Their passing
lag components do not erase these accuracy failures.

Independent checks reproduced the predictions in Torch and NumPy and matched
training scaling moments within 9.76e-8. The available candidate cues distinguish
the measured failure, but some synthetic feature combinations lie well outside
the natural training distribution. One final augmentation trial is allowed in
the same family: the unchanged framewise MLP32 and v2 candidates, same seed,
optimizer, final epoch and natural data. No recurrent retry, candidate change,
threshold relaxation or additional architecture search is introduced.

Freeze 128 analytic training sequences of 2.5 seconds each: 48 stationary,
24 glides, 24 steps, 24 harmonic changes and 8 aperiodic cases. The fixed
parameter manifest varies frequency, phase, amplitude and harmonic mixtures;
missing-fundamental examples retain harmonics 2 and 3. Every fourth voiced
sequence receives fixed 20 dB white noise. Repeat each synthetic training
sequence four times in scaling and fitting, approximately 15.7% of the effective
condition-expanded frame mass. This replay weight is not additional independent
data. Synthetic analytic targets and provenance remain separate from PTDB.

Freeze a disjoint 32-sequence, 80-second synthetic challenge before generating
training data or fitting. It uses separate identities/seeds/configurations,
off-grid stationary frequencies, different transition parameters, harmonic-6
mixtures, and clean/14 dB cases. Continuous glides may pass through common
instantaneous frequencies; independence concerns full generator configurations,
not an impossible exclusion of every shared frequency. Challenge content is
opened only after augmented weights freeze. The known original counterexamples
are regression checks, not newly held-out evidence.

Retain the original MLP natural-quality, coverage and lag gates. Add: every
original stationary case and post-change observation case must achieve at least
95% reference-voiced success; the new synthetic challenge must have at least
90% equal-scenario voiced success, at least 80% in each voiced scenario, and no
more than 5% false voicing in each unvoiced case. Undefined denominators/failures
remain explicit. Failure ends neural model selection; keep the tested classical
path and report the complete tradeoff. No sealed-test selection is permitted.

Only if this final configuration passes, confirm fixed seeds 20261022, 20261023
and 20261024 and report all three. The initial seed remains the deployment
candidate; never pick the best result. Calibration and frozen-test evaluation
remain later gates. Reuse of development outcomes for this targeted refinement
is acknowledged; the sealed test remains the independent final comparison.

## Final disposition after the bounded augmentation trial

The augmented MLP has 8.261998% equal-speaker/equal-condition development joint
error and 77.034217% reference-voiced success. The separately frozen synthetic
challenge has 88.070539% equal-scenario voiced success, below 90%; four voiced
cases fail the 80% per-case gate. The known constant-300-Hz regression remains
0/151 correct frames. The unchanged architecture and augmentation therefore do
not qualify. No further candidate revision, fitting configuration, threshold
change, confirmation seed, calibration, or sealed-test acquisition is performed.

The production DSP is retained. A trained MLP export may be exercised through a
research-only worker hook to test numerical and transport integration, explicitly
marked unqualified and absent from the default product path. This is a systems
fixture, not a change to the model-selection decision. Runtime evidence cannot
reverse the failed scientific gate. Public results preserve development reuse,
negative findings, timing limitations, and the untested sealed set.
