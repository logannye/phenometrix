# Matched voicing robustness: audited no-go

The six-fit acoustic-exposure treatment **does not pass the fixed development gates**.
Its three-seed mean shifted false-negative-or-unavailable rate is **14.8730%**,
versus **14.9978%** for matched control C and **16.3251%** for frozen hybrid H.
The reduction is only **0.1248 percentage points / 0.83% relative** against C,
and **1.4521 points / 8.89%** against H; both required 20% relative improvement.
No new model advances. Production remains DSP; the natural-browser comparison
uses H twice. KEELE is blocked and no external outcome was accessed.

Assessment execution, artifact completeness and independent numerical/scoring
verification completed. The independent audit reconstructed **12,972,330
frame/model comparisons** and the failed scientific decision. New natural-browser
execution and independent acoustic/delivery audit also completed on the second
attempt: ordinary H/H replay delivered **27,952/27,964 intended targets**, with
**24,858/27,708 (89.7142%)** delivered-correct coverage and **5.6 ms** processed-block
service p99. The failed first attempt remains separate. Required local software
checks pass; current-revision hosted status is supplied by the PR checks.

The [frozen plan](VOICING_ROBUSTNESS_PLAN.md) and
[public aggregate JSON](../../benchmarks/voice-prosody/voicing-robustness-summary.json)
contain exact gates, all speaker/condition cells, counts, six diagnostics,
synthetic case summaries, model/source identities and audit receipt hashes.

## Primary result

All values below are percentages. Shifted means the fixed noise10db,
first_difference and clip01 conditions; clean is a separate guardrail. FP uses
all reference-unvoiced frames across the four conditions. Joint error uses all
eligible valid references, including missed/unavailable output and pitch error
above 50 cents. Utterances pool within speaker/condition, then cells receive
equal weight. Seed means average scores, never predictions. These are practical
point-estimate gates, not evidence of a population-wide effect.

The primary comparison continues across the two keyed tables. Reductions are
baseline minus row; positive means improvement. `25/26/27` abbreviate the three
20261025/26/27 seeds. Full checkpoint hashes are in the aggregate.

| Model / checkpoint prefix | Shifted FN + unavailable | Reduction vs H (pp / relative) | Reduction vs C (pp / relative) | FP | Clean joint | Overall joint | Analytic / events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H / `06110469a9a6` | 16.3251 | reference | n/a | 3.1100 | 8.2796 | 9.2986 | pass |
| C25 / `989cbca003c7` | 15.8292 | +0.4959 / +3.04% | reference | 3.1099 | 8.2345 | 9.2092 | pass |
| R25 / `19bc8d76045b` | 15.0664 | +1.2587 / +7.71% | +0.7628 / +4.82% | 3.3406 | 8.4375 | 9.3053 | pass |
| C26 / `c06f12da4234` | 14.8617 | +1.4633 / +8.96% | reference | 3.2852 | 8.4135 | 9.2489 | pass |
| R26 / `bd7ef3072923` | 15.6706 | +0.6544 / +4.01% | −0.8089 / −5.44% | 3.0808 | 8.2163 | 9.1687 | pass |
| C27 / `25e8769cd970` | 14.3024 | +2.0227 / +12.39% | reference | 3.3159 | 8.4246 | 9.2236 | pass |
| R27 / `a95b779bf971` | 13.8820 | +2.4431 / +14.97% | +0.4204 / +2.94% | 3.4206 | 8.4638 | 9.2368 | pass |
| C mean; no model | 14.9978 | +1.3273 / +8.13% | reference | 3.2370 | 8.3576 | 9.2272 | all pass |
| R mean; no model | 14.8730 | +1.4521 / +8.89% | +0.1248 / +0.83% | 3.2807 | 8.3725 | 9.2369 | all pass |

| Model rows | External status | Browser identity / status | Service p99 | Delivered-correct coverage | Lost / withheld |
| --- | --- | --- | ---: | ---: | --- |
| H | blocked; no outcomes | H/H, ONNX `3df30ae54f4b`, config `be2814796e0c`; attempt 2 audited | 5.6 ms ordinary pooled | 24,858/27,708 (89.7142%) | 0 lost blocks / 0 withheld; 12 final-partial targets |
| C25, C26, C27 | blocked; no outcomes | not exported; not measured | not measured | not measured | not measured |
| R25, R26, R27 | blocked; development failed | not exported; not measured | not measured | not measured | not measured |
| C mean, R mean | blocked; no outcomes | score means; no implementation | n/a | n/a | n/a |

The separate injected-stall H case is reported below. Browser quality pools
frames from a small finite replay; it is not the equal-speaker/equal-condition
scientific metric in the first table or a fresh accuracy cohort.

R/20261025 was the sole candidate designated before fitting. Against its paired
C it improves shifted FN/unavailable by **0.7628 points / 4.82%**; against H the
change is **1.2587 points / 7.71%**. Its overall joint error **9.3053%** is worse
than both C/20261025 (**9.2092%**) and H (**9.2986%**). These are overall values;
its shifted joint error is a different quantity, **9.5945%**.

The exact failed requirements are: mean 20% shifted improvement versus C and H;
mean overall joint improvement versus C; R/20261025's 20% shifted and overall
joint improvements versus both comparators; and R/20261026's shifted
nonregression and M05 first-difference limit versus its paired C. All declared
FP and clean guardrails pass, as do the analytic/transition/event requirements.
Passing those checks does not override a failed required gate.

## Failure mechanism and strata

Most of the apparent improvement over H also appears in the matched control:
C lowers shifted FN/unavailable from 16.3251% to 14.9978%. The controlled C→R
comparison measures only the remaining 0.1248-point mean change from the new
acoustic-exposure intervention under this fixed experiment.

The modest mean C→R effect is a **0.1268-point strict-FN reduction**, partly offset
by a **0.0020-point increase in unavailable-on-voiced output**. Shifted strict FN
is 14.9481% for C and 14.8213% for R; unavailable rates are 0.0496% and 0.0517%.
Overall FP rises from 3.2370% to 3.2807%. Voiced pitch errors retain their own
denominator; recovering a voiced decision can expose an unchanged raw-YIN error.
Overall joint error rises from 9.2272% to 9.2369%, so lower mean missed voicing
does not establish better overall utility.

| Model/seed | Original four: joint error | Formerly reserved four: joint error | M05 first_difference: FN + unavailable |
| --- | ---: | ---: | ---: |
| H | 8.9224 | 9.6747 | 54.3816 |
| C-20261025 | 8.7985 | 9.6199 | 52.7160 |
| R-20261025 | 8.8857 | 9.7248 | 51.8459 |
| C-20261026 | 8.8822 | 9.6156 | 48.7632 |
| R-20261026 | 8.7545 | 9.5829 | 52.2933 |
| C-20261027 | 8.8365 | 9.6106 | 47.1722 |
| R-20261027 | 8.8521 | 9.6214 | 46.9236 |

R/20261026 worsens M05 first-difference by **3.5301 points** versus C/20261026,
above the 2-point limit. Its shifted FN/unavailable also rises from 14.8617% to
15.6706%. The aggregate retains every cell, worst cells, undefined denominators
and out-of-range reference counts. All eight speakers are now consumed
development/regression data: the original four contribute 80 sources and the
formerly reserved four 160. This is not a fresh held-out or external cohort.

The reference-voicing oracle keeps raw YIN fixed and replaces only the voicing
decision with labels. Its shifted FN/unavailable is **0.2464%**, so the required
reduction was not ruled out by unavailable YIN. Yet its shifted voiced success
is only **75.9877%**, with **6.2179%** joint error. This oracle uses reference
labels unavailable at inference; it establishes headroom and remaining pitch limitations, not a candidate.

## Fixed diagnostic and event checks

These six views reuse the same 240 sources and are dependent acoustic-setting
diagnostics. They cannot replace the primary gates. FN/unavailable percentages:

| Diagnostic | H | C mean | R mean |
| --- | ---: | ---: | ---: |
| Gain −12 dB | 7.7159 | 7.1674 | 6.6659 |
| Gain −3 dB | 5.6543 | 5.6754 | 5.2020 |
| Emphasis a=0.5 | 7.0603 | 6.6241 | 6.1184 |
| Emphasis a=0.95 | 28.3177 | 24.4677 | 24.4691 |
| White noise, 8 dB SNR | 6.1489 | 5.7274 | 5.3955 |
| AR(1) noise, 18 dB SNR | 6.1573 | 6.0893 | 5.7536 |

All seven hybrid models pass the retained 78-case analytic/transition inventory
and all 12 new voicing-event cases. Every event is detected and every plateau
passes. H's median signed onset/offset target lags are −4/21 ms; R's are −4/16 ms
for every seed. Earliest availability adds 20 ms to those target lags; negative
target lag is not negative processing latency. Per-case delays, unchanged
search windows and missing-event lists remain in the aggregate.

The original candidate MLP's lower natural joint error remains visible:

| Frozen comparator | Overall joint error | Shifted FN + unavailable |
| --- | ---: | ---: |
| H, frozen hybrid | 9.2986 | 16.3251 |
| Original candidate MLP | 8.8753 | 16.2662 |
| DSP | 11.2274 | 36.9259 |
| Gated YIN | 11.1916 | 30.7767 |
| Continuity | 11.4236 | 21.7788 |

The candidate MLP's historical pitch robustness failures still prevent promotion.
The matched exposure treatment changes the shared representation under unchanged
voicing BCE and covered-candidate cross-entropy. It does not isolate a voicing
head weight change or identify which augmentation component caused an effect.

## Training, verification and retained failures

Each arm used 1,920 source/view slots, 1,352,768 feature frames and 1,339,328 valid
supervision frames, with original frozen normalization, paired initialization,
12 epochs and identical paired schedules. Updates per arm were 2,861 / 2,862 /
2,862 for seeds 20261025 / 20261026 / 20261027. Candidate-covered supervision was
264,362 frames in C and 266,681 in R. Six final checkpoints completed on MPS with
Torch 2.9.1, Python 3.11.15 and macOS 26.2 arm64; no additional fit followed failure.

The independent NumPy replay checked every saved model output, including candidate
logits and unavailable states. Maximum absolute differences were 1.55e−6 for
voicing logits, 2.38e−6 for candidate logits and zero for state. The audit also
reconstructed preparation, paired fits, counts, aggregation and all gate checks.
It verifies the retained source snapshots, not untested later revisions.

Baseline audit attempts v1 and v2 remain failed (historical-source resolution,
then handling historical prediction tails beyond reference extent); v3 resolves those contracts and is complete.
Assessment v1 failed while hashing string module paths, **before any prediction**.
Assessment v2 is a create-new attempt with exactly `digest(module)` changed to
`digest(Path(module))` plus a regression test; all six checkpoints and inputs
were reused. An initial v2 invocation also failed before phase execution because
of its working directory. None of these failed receipts was rewritten. Both
288-file source snapshots, the repair diff and separate lineage audit are retained.
All four input-only prediction freezes precede the first reference-scoring phase.
This chronology is source/receipt evidence, not an OS-level file-access audit.

A resource-only amendment raised the initial new-artifact cap from 4 to **8 GiB**
before fitting/predictions, with a 15 GiB free-space floor. It preserved all
seeds, inputs, gates and the six-fit ceiling. Raw media, references, features,
weights and frame predictions remain ignored local evidence.

## Mechanism ledger and external/browser limits

| Choice | Observation / smallest falsifier | Decision |
| --- | --- | --- |
| Continuous TRAIN channel exposure | Required shifted/utility gates fail; one M05 regression | Reject advancement; no second bank or threshold sweep |
| Matched initialization, scaling and schedules | All three paired audits pass; a mismatch would invalidate comparison | Comparison admitted |
| Frozen raw YIN | Oracle admits voicing headroom; residual pitch error persists | Keep inference fixed; no post hoc pitch repair |
| Fixed diagnostic/event sets | All events pass; emphasis 0.95 offers no mean benefit over C | Supporting regression evidence only |
| KEELE reference contract | Metadata cautions against differing estimator parameters and documents imperfect delay correction | External blocked; no outcomes |
| Finite natural browser transport | Second 34-case execution and exact-support/parity/drain audit pass; first attempt failed | Retain bounded descriptive H replay; no new model or clinical qualification |

[KEELE intake](KEELE_INTAKE.md) stopped before waveform or reference-array access.
Its reference specifies 25.6 ms autocorrelation while this study fixes 40 ms YIN;
compatibility could not be established prospectively. The retained metadata and
synthetic-tested exact-grid adapter do not establish an external model result.
The full archive was not downloaded or checksum-verified, and redistribution
rights are not inferred. No substitute external corpus was sought.

## Natural browser execution and delivery

The [natural-browser protocol](NATURAL_BROWSER_PROTOCOL.md) uses the prescribed
eight-speaker PTDB fallback: eight full sources, clean/first-difference, H twice,
with a fresh context per source. Ordinary replay contains 280.728 seconds of
prefixed audio; the separate stall source adds 9.934 seconds and the synthetic
origin fixture precedes both. Repetition does not add independent sources.
The [browser aggregate](../../benchmarks/voice-prosody/voicing-robustness-browser.json)
retains every source/condition cell, pooled counts, exact identities and failures.

Attempt 1 completed its workload but failed one second-H M05 first-difference
origin receipt: capture coordinates advanced 128 samples beyond render coordinates
after a discontinuity. Its audit separately failed on outputs beyond the final
provider reference row. Both failed receipts and original source snapshots remain.
The repair derives natural-source coordinates directly, explicitly accounts for
gaps and grid-restart skips, and rejects backward/overlapping quanta. Reference-less
tails stay in delivery accounting and are excluded only from reference scoring.
No fit, threshold, reference value or frozen workload changed. The underlying
callback event remains unexplained; attempt 2 does not erase that limitation.

Attempt 2 passed all eight research browser tests and all 34 independent case
audits, with exact source origin/endpoints, target phase, model/config bindings,
actual reset support, numerical parity, credits and completion fences. No source
timeline discontinuity occurred in attempt 2. Features were reconstructed using
shared pure TypeScript equations outside transport; MLP equations use independent
NumPy and YIN uses librosa. Maximum absolute differences across all 34 cases are
zero for features, 1.91e−6 for voicing logits, 1.19e−6 for candidate logits and
9.67e−13 Hz for YIN. This is not an independent rederivation of feature DSP.

| H replay | Processed blocks | Service p50 / p95 / p99 / max (ms) | Blocks >10 / >20 ms | Delivered / intended | Correct / eligible valid references |
| --- | ---: | --- | ---: | ---: | ---: |
| First H | 7,012 | 2.9 / 4.7 / 5.5 / 26.2 | 16 / 10 | 13,976 / 13,982 | 12,429 / 13,854 (89.7142%) |
| Second H | 7,012 | 3.0 / 5.0 / 5.8 / 24.7 | 16 / 9 | 13,976 / 13,982 | 12,429 / 13,854 (89.7142%) |
| Ordinary pooled | 14,024 | 2.9 / 4.9 / 5.6 / 26.2 | 32 / 19 | 27,952 / 27,964 (99.9571%) | 24,858 / 27,708 (89.7142%) |
| Separate 250 ms worker stall | 483 | 3.1 / 4.9 / 5.5 / 24.0 | 1 / 1 | 960 / 990 (96.9697%) | 882 / 982 (89.8167%) |

Quantiles pool actual processed-block durations, never case quantiles. All ordinary
case p99 values and the separate stall case meet the provisional <10 ms goal;
19 ordinary blocks nevertheless exceed the 20 ms cadence. The 50-block synthetic
origin/warmup fixture has p99/max 16.9 ms and fails the per-case <10 ms goal,
while passing exact alignment and delivery; it is not natural-speech quality
evidence. Ordinary transport,
worker withholding and cancellation counts are zero, but final partial PCM totals
11,904 samples and leaves 12 intended targets undelivered. There are 160 boundary
targets and 96 targets without provider references; 84 of the latter were delivered.
All 27,708 eligible provider rows remain in the quality denominator. Missing
references cannot establish correctness and are not scored as errors.

The stall loses nine transport blocks and withholds four blocks for freshness,
with no canceled inference or host-discarded predictions. Its 30 missing targets
comprise 18 transport-loss, eight worker-withheld-support, three reset-warmup and
one final-partial target. These counts are separate from ordinary replay. The
full-denominator coverage includes unavailable output. Stall conditional accuracy
among delivered reference-valid targets is 882/953 (92.5498%), secondary to
882/982 (89.8167%) full-denominator coverage; ordinary conditional counts coincide
with the full denominator. Five stall feature resets are retained; the last
pre-gap and first recovered support endpoints are 300 ms apart on the source
grid, with 913 outputs in the recovered segment. This is a source-grid recovery
interval, not a wall-clock latency guarantee.

Same-main-clock software-render age certificates cover all ordinary delivered
frames, with p99 **50.0 ms** and maximum **66.3 ms**, under the 100 ms bound. The
certificate conservatively spans lease issuance before the block's first observed
input to main-thread receipt. There are **213 invalid cross-clock frame timings**;
these are retained as invalid, not clamped into latency. All 308 ordinary clock
probes were observed, but no single affine clock model fits all callback brackets
in any of the 32 ordinary cases; no outliers were discarded. The stall maximum certified
age is 54.4 ms. Service time, 20 ms target-relative future support, and software age
have different boundaries; none measures physical microphone-to-display latency.

Measurements used one process on Apple M4 Max, 16 logical cores, 128 GiB RAM,
macOS 26.2/Darwin 25.2.0 arm64, Chrome 156.0.8078.47 and ONNX Runtime Web 1.23.2
WASM with one thread and proxy disabled. No fitting/auditing ran concurrently;
OS background load was uncontrolled and `timingBenchmark` remains false.
The model is 14,731 bytes, manifest 4,458, WASM 11,905,541 and loader 20,321.
Four PCM credits, four result credits, three queued blocks and a 96,000-sample
ring remain bounded; ordinary observed PCM/result high-water is 2/2, stall 4/4,
and queued-block high-water is zero. Research worker/timeline ledgers cap at
20,000/64 events, raw telemetry at 200 queued/100 per batch. Sampled renderer JS
heap ranges from 6,985,476 to 13,362,624 bytes after ordinary sources; this is
neither peak heap nor a bound on total browser memory.

## Reader entry point and application wording

The aggregate is readable without local corpus artifacts. With previously
authorized local data, inspect the bound receipts from the research directory:

```sh
python -m json.tool runs/voicing-robustness-assessment-v2/decision.json
python -m json.tool runs/voicing-robustness-assessment-v2/engineering-lineage-audit.json
# Separate audit rerun only when no training or browser measurement is active.
# Choose a NEW output; the auditor refuses to overwrite an earlier receipt.
.venv/bin/python -m phenometrix_prosody.audit_voicing_robustness \
  --run runs/voicing-robustness-assessment-v2 \
  --output runs/NEW-voicing-review-audit.json
```

The [source-bound local validation receipt](../../benchmarks/voice-prosody/voicing-robustness-local-validation.json)
records **1,316 Python tests**, **356 web unit tests**, repository checks,
TypeScript checking, both production builds and research-asset exclusion, all
passing. The complete research browser route passed **8 tests with zero skips**;
the separate independent audit covered all 34 finite cases. Four known Python
warnings remain recorded. Hosted synthetic checks run on the PR revision; consult
its current checks rather than an older commit's status. They do not validate
local corpus results. Raw recordings, references, weights, frame predictions and
traces remain ignored; all public outputs contain aggregate evidence only.

Application wording, preserving the original positive result:

> Built a compact speech-measurement system combining a 3,143-parameter learned
> voicing model with causal YIN pitch estimation and an explicit 20 ms target-relative
> measurement delay. On 160 recordings from four then-reserved PTDB speakers, it
> achieved 9.6747% joint error versus DSP's 11.8482%, while retaining a documented
> accuracy tradeoff against the candidate-selecting MLP's 9.1158%. Carried the same
> frozen hybrid through finite natural-audio replay in the actual AudioWorklet and
> ONNX-WASM worker pipeline: ordinary H/H replay measured 5.6 ms service p99,
> 99.9571% target delivery and 89.7142% delivered-correct coverage over all eligible
> references, with zero lost/withheld blocks and 12 final-partial targets. These
> are descriptive nonclinical research results; 19 blocks exceeded 20 ms, the
> injected stall retained its losses, and external-corpus generalization remains
> unassessed.
