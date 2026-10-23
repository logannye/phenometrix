# Overnight development: training exposure and acoustic evidence

## Authorized window and purpose

The user authorized starting immediately for approximately eight hours. This
session runs from **2026-10-23 04:18:58 UTC to 12:18:58 UTC** (October 22,
9:18 p.m. through October 23, 5:18 a.m. America/Los_Angeles). The deadline is a
ceiling, not a requirement to keep searching. Finish early if this fixed scope
is complete. Base revision: `e15b40b5cd39b383fa5a6c64a0d832e9611166ab`.

The previous expanded-pool study preserved raw candidates but did not repair
selection. Its synthetic TRAIN bank already covered all 18,336 targets under
the original pool, so it supplied no newly coverable synthetic examples. The
next useful question is whether a targeted TRAIN distribution, additional
candidate-conditioned acoustic evidence, or their combination can help.

This is a bounded research and engineering session. It cannot promise improved
accuracy, and a well-audited negative result is a valid completion. The default
DSP estimator and all clinical boundaries remain unchanged.

## Work sequence and deliverables

| Window from start | Work | Concrete completion evidence |
| --- | --- | --- |
| 0–1 hour | Diagnose the TRAIN evidence gap, specify one prospective comparison, and freeze the exact feature, generation, exposure and evaluation contracts | Source/specification identities, fixed arm inventory and feasibility gates; no fresh evaluation content read |
| 1–3 hours | Implement the candidate-evidence feature, TRAIN-only augmentation selection, matched training path and independent audit support | Known-answer, causality, six/seven-pool retention, initialization, masking, finite-value and source-binding tests; bounded local feature-cost pilot |
| 3–4.5 hours | Execute the one frozen comparison, only after prerequisites pass | At most twelve final checkpoints; identical schedules within declared pairs; all losses/exposures and failed attempts retained |
| 4.5–6.5 hours | Freeze predictions, score all regressions and the single fresh challenge, run independent audits; qualify browser inference only if scientific gates pass and time permits | Full per-case/natural results, explicit-equation output audit, independent scores/gates, and separate browser evidence if attempted |
| Final 90 minutes | Finish the failure report, required local checks, high-level README and PR update | Reproducible report, aggregate receipts, clean validated commit, updated PR #33 and zero hosted workflow runs |

Start no training after hour 5.5 and no optional browser run after hour 6.5.
Deadline checks and per-process timeouts take precedence over the table. If a
phase overruns, reduce optional engineering work rather than relax scientific
requirements. Preserve incomplete work and state what was not validated.

## One fixed comparison, maximum twelve fits

Use the expanded seven-candidate pool throughout. Cross two interventions:

1. **TRAIN exposure:** the existing harmonic-balance bank versus one new,
   coverage-targeted TRAIN bank and fixed sampling rule.
2. **Acoustic evidence:** the existing feature path versus one compact family
   of candidate-conditioned, window-local periodicity evidence.

This makes four arms across the existing three seeds. Predesignate targeted
exposure plus acoustic evidence as the candidate; the other arms are controls,
not alternatives to promote after outcomes are known. This design estimates the
combined new-bank/sampling intervention, not the isolated effect of oversampling
versus broad new-bank exposure. No architecture or hyperparameter sweep follows.

The four arms are `existing__off`, `existing__on`, `targeted__off` and
`targeted__on`; the last is predesignated. All arms have 3,012 parameters: the
existing 3,009 plus a shared bias-free Linear(3,1) logit residual initialized
to zero. Evidence-off supplies three zeros. This preserves the old control's
initial output and parameter path, subject to exact preflight checks.
The existing source-confidence fields, original candidates and original MLP
voicing remain intact. Original feature scaling remains original-six natural
TRAIN only. Any added-channel scaling must be fixed analytically or fitted only
on natural TRAIN, declared before execution, and shared across arms.

Keep the natural schedule of 132,466 rows, twelve epochs, 396 updates per fit,
Adam learning rate .001, clipping at 1, synthetic coefficient .25 and 512
synthetic rows per update. Record conditional loss denominators, replacement
sampling and case/region exposure; matching nominal batch size alone does not
prove equal effective weighting. Restore and check the historical control
where the declared equations and schedules are unchanged.

## TRAIN selection and fresh evaluation

Generate at most **512 two-second TRAIN configurations** from one frozen broad
frequency/phase/gain specification. Select strata from analytic TRAIN truth and
original/expanded candidate coverage only. Include both examples where only an
added candidate is correct and counterexamples where the added candidate is
wrong but an old candidate is correct. Do not teach an unconditional preference
for the added or highest-frequency candidate. Freeze minimum distinct-case and
frequency-support requirements, exact stratum weights and fallback behavior
before generation. Insufficient support ends the scientific run without fits;
there is no outcome-driven enlargement or revised threshold.

The exact overnight experiment specification must be a separate file frozen
before scientific generation. Implementation may use fixed toy fixtures for
contract tests, kept separate from training and evaluation. No fresh evaluation
PCM, features or outcomes may be read during diagnosis, implementation or training.

Retain **all 74 consumed analytic cases**, including both phases at 517.3,
641.9 and 633.1 Hz. Reserve one set of **twelve fresh complete configurations**
before fits; generate its inputs only after all final checkpoints freeze.
There are 86 total analytic cases and **78 mandatory case gates per candidate
seed** (three stationary, three original changes, sixty consumed harmonic-change
cases and twelve fresh cases). Freeze every prediction before final scoring.
Never use the fresh outcomes for a second training run, model selection or
feature revision during this session.

All original natural and candidate prerequisites remain unchanged: raw fixed-
voicing oracle headroom at least one point; actual grouping loss at most 0.5
points; mean natural joint-error improvement at least 0.5 points against both
MLP and hybrid; every seed beats MLP; no condition worsens more than one point;
unconditional voiced success does not decrease. Each required analytic case
must individually reach at least 95% within 50 cents for every candidate seed.
Maintain 40-ms support, 10-ms hop and 20-ms measurement delay. All checks are
conjunctive. Training diagnostics are descriptive, not replacement gates.

TRAIN/DEV are consumed research data. Explicitly declare any consumed DEV
reference reads for prerequisite pass/stop checks; never describe the whole
experiment as globally label-blinded. No new natural-data acquisition occurs.
Metadata confirms that all PTDB speakers have contributed observed outcomes;
remaining unread prompts from them cannot establish untouched-speaker validity.

## Acoustic evidence and engineering boundary

Use a small primitive-period contrast evaluated at each observed grouped
candidate's period and proper subperiods. The purpose is to expose whether a
longer candidate period is equally explained by a shorter repetition, rather
than asking the network to infer that from source-confidence values alone.
For actual grouped frequency f, use T=48000/f and the original 1,920 float32
samples, mean-centered in float64. Compute normalized correlations at
T, T/2, T/3 and T/4 using linear interpolation and the common prefix
L=1920-floor(T)-1. All anchor/shifted energies must be at least 1e-12; otherwise
validity and evidence are zero. Clamp correlations to [-1,1]. The three model
inputs are C(T), [C(T)-max(C(T/2),C(T/3),C(T/4))]/2, and validity. Preserve all
four correlations in provenance. This is a learned input, not a hard pitch veto
or a frequency-specific correction. It does not mathematically resolve every
possible periodic ambiguity, noise condition or nonstationary waveform.

Use only the existing acquired window, with no additional lookahead, new
candidate generation, reference-derived inference input or hidden state.
Verify silence/low-energy behavior, finite outputs, candidate permutations,
empty masks, source retention, amplitude/DC behavior and exact sample support.
Independently reconstruct the feature equations and model inference. New
feature compute must fit a declared local throughput budget before full extraction.

Provide an inspectable local failure report that separates candidate absence,
voicing, ranking, grouping changes and unavailable outputs. A lightweight
synthetic-case visualization is optional if time remains. It must distinguish
analytic truth used for offline scoring from inference inputs, and it must not
label raw selector scores as calibrated correctness probabilities.

If every scientific gate passes, attempt a separately frozen export/browser
arithmetic and bounded replay check. A scientific pass does not transfer to
browser qualification. If a scientific gate fails, preserve the no-go and use
remaining time for audits, feature-contract tests and the failure report. No
failed model is put into the default product path.

## Local resource limits, stopping and shipping

Use the existing local M4 Max/128 GiB environment. One MPS fitting process;
maximum **twelve fits and 60 minutes cumulative fitting**. Cap observed Python
RSS at **16 GiB**, Node-worker RSS at **2 GiB**, MPS allocation at **8 GiB**, and
new disposable outputs at **8 GiB**; require **16 GiB free disk**. Bound each
worker request, each subprocess and the scientific wall time by the remaining
session deadline. These observations are not whole-machine memory guarantees.
No package installation, native/Rust build, duplicate environment, remote
compute or external data acquisition is authorized for this session.

Use `runs/overnight-20261023/session.json` as the progress ledger, recording
phase, process ownership, source identities, fit count, resource use and the
next action. The hourly thread heartbeat resumes only unfinished work; it must
not duplicate an active process. Silence unchanged progress. Notify on meaningful
results, actionable failure or final completion. A temporary idle-sleep inhibitor
expires at the deadline; the Mac still needs power, an open lid and the app running.

Integrity, nonfinite, resource, insufficient-support or prerequisite failures
stop dependent work. Preserve failure logs. A plumbing repair cannot alter the
frozen policy, repeat a completed fit or replace a fresh set. At completion or
deadline, pause the heartbeat and preserve a clear final status. Do not keep
working after the deadline simply because a hypothesis remains unresolved.

Before shipping, run meaningful focused tests, the complete research suite and
`npm run check`, and bind the exact validated files. Update the high-level README,
detailed study report and existing PR #33. Commit/push only coherent validated
code, aggregate evidence and documentation with `[skip ci]`; verify zero hosted
workflow runs across all event types. Do not merge. Preserve historical files
and evidence; raw media, PCM, features, references, weights and natural frame
outputs remain local and ignored.
