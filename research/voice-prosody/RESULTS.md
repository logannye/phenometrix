# Voice estimation results — 22 October 2026

**No learned model is promoted.** The models improved aggregate development error, but failed predeclared selection or known-truth requirements. The augmented MLP still selected a subharmonic on every scored frame of a stationary 300 Hz harmonic tone and failed the independent synthetic challenge. The classical production path remains, with its own documented failures. No sealed-test result, calibrated F0 correctness probability, seed confirmation, live microphone demonstration, or clinical qualification is claimed. The production DSP retains its legacy time semantics; the 20 ms delayed task is a separate research retiming and is not silently adopted in the product.

[Machine-readable summary](../../benchmarks/voice-prosody/final-summary.json) contains exact metrics, every speaker/condition cell, all 14 original scenarios, all 32 challenge scenarios, gates, and SHA-256 identities. [Protocol](PROTOCOL.md), [data audit](DATA_AUDIT.md), [alignment audit](ALIGNMENT_AUDIT.md), and [comparator audit](COMPARATOR_AUDIT.md) describe the decisions and limitations. Saved local predictions were frozen before reference scoring; this review checked metric-manifest hashes, actual checkpoint hashes, cell count ratios, and macro aggregation without opening audio or reference values.

The bounded PTDB-TUG experiment used 240 training recordings from 12 speakers and 80 development recordings from four other speakers, with 20 eligible recordings per speaker. Seed 20261022 assigned matched pairs to train [4,8,7,9,10,2], development [1,3], and sealed test [5,6]; shared sa1/sa2 prompts were excluded. Four conditions produce 960 training and 320 development records: clean, fixed white noise at 10 dB, causal first difference, and clipping at 0.1. These transformations are dependent observations of the same sources. The development score includes 253,292 valid common-support frames; the four speakers represent only two matched-prompt pairs.

The primary error counts incorrect voicing, pitch error over 50 cents, and unavailable outputs as failures. Recording counts are pooled within each speaker/condition and the 16 cells receive equal weight. Voiced success divides correct voiced estimates by all reference-voiced frames. Coverage below means outputs that are not unavailable, including unvoiced decisions; it is not qualified pitch accuracy. FP and FN use reference-unvoiced and reference-voiced denominators respectively. Raw voicing probabilities are uncalibrated.

The nominal reference grid remains 16 + 10k ms. The delayed experiment uses support [t−20 ms, t+20 ms) and becomes available at t+20 ms. CREPE uses [t−32 ms, t+32 ms), available at t+32 ms. The shared mask requires t≥64 ms and t+32 ms≤recording length. No per-recording shift was fitted. Measurement delay, compute time, and microphone acquisition latency are separate quantities.

**Development results.** All percentages use the aggregation above.

| Method | Future support ms | Joint error % ↓ | Voiced success % ↑ | Coverage % | FP % | FN % |
|---|---:|---:|---:|---:|---:|---:|
| DSP | 20 | 10.61 | 62.45 | 100.00 | 2.01 | 32.33 |
| YIN | 20 | 11.02 | 64.88 | 100.00 | 3.37 | 25.58 |
| Fixed continuity | 20 | 11.60 | 69.39 | 100.00 | 5.52 | 19.02 |
| Original MLP | 20 | 8.63 | 75.41 | 100.00 | 3.52 | 13.27 |
| Original GRU | 20 | 7.95 | 76.38 | 99.97 | 2.92 | 11.63 |
| Augmented MLP | 20 | 8.26 | 77.03 | 100.00 | 3.54 | 12.69 |
| CREPE tiny | 32 | 9.56 | 66.81 | 100.00 | 2.06 | 27.76 |

| Joint error % | Clean | Noise 10 dB | First difference | Clipping |
|---|---:|---:|---:|---:|
| DSP | 9.12 | 8.43 | 15.83 | 9.05 |
| YIN | 8.60 | 7.64 | 19.19 | 8.65 |
| Fixed continuity | 9.91 | 8.49 | 18.06 | 9.93 |
| Original MLP | 7.92 | 7.26 | 11.37 | 7.98 |
| Original GRU | 7.17 | 6.87 | 10.51 | 7.26 |
| Augmented MLP | 7.57 | 7.01 | 10.87 | 7.60 |
| CREPE tiny | 6.79 | 6.95 | 17.67 | 6.83 |

| Joint error %, equal conditions | F01 | F03 | M01 | M03 |
|---|---:|---:|---:|---:|
| DSP | 9.59 | 6.22 | 13.05 | 13.56 |
| YIN | 10.42 | 10.10 | 11.74 | 11.82 |
| Fixed continuity | 11.60 | 11.16 | 11.23 | 12.41 |
| Original MLP | 9.30 | 6.27 | 8.87 | 10.11 |
| Original GRU | 8.90 | 5.44 | 8.11 | 9.36 |
| Augmented MLP | 8.96 | 5.59 | 8.59 | 9.90 |
| CREPE tiny | 10.00 | 8.22 | 10.51 | 9.51 |

The original GRU's gain over the original MLP was **0.6823 percentage points**, below the required 1 point. Both MLPs passed their natural-data quality components; this did not pass the other gates. Augmentation improved MLP joint error from 8.6349% to 8.2620%, but reused development after the original outcomes. Its gain is exploratory. The candidate oracle was 88.1304% on development; it uses scorer knowledge and is not an executable estimator. CREPE's clean score is lower than either MLP, while its first-difference score is much worse; the combined ranking does not establish superiority in every condition.

**Failures and the changed time contract.** The original clean training pilot comprised 24 recordings, 12 speakers and 172.97 seconds. The initial three-candidate oracle covered 52.46% of voiced reference frames. One permitted evidence revision reached 64.5709%, below its predeclared 75% prerequisite. That zero-lookahead gate remains failed. Synthetic ramps showed trailing-window response lag, motivating a separately declared 20 ms delayed task. On the stricter shared pilot mask, its oracle reached 88.2052%; DSP joint error changed from 15.7737% to 9.2980%, YIN from 17.2117% to 9.6001%. Centered CREPE scored 7.2792%; the explicitly modified end-anchored CREPE scored 18.4181%. These are training diagnostics under different availability contracts, not learned gains or evidence that reference timestamps are wrong. Initial pilot scores under its original, less restrictive mask were DSP 15.729% and YIN 17.163%; those values must not be interchanged with the shared-mask figures.

**Source-description correction.** An earlier version called the original constant-300 fixture a pure tone. Its frozen generator, [`support_diagnostic.ts`](support_diagnostic.ts), actually uses `0.075 × [sin(φ) + 0.5 sin(2φ) + 0.25 sin(3φ)]`, with a 300 Hz fundamental in this case. The delayed evaluation reuses those exact PCM bytes. This wording correction changes no data, predictions, metrics, model identities, or historical run hashes.

All three learned checkpoints failed the original stationary 300 Hz case: **0/151 frames within 50 cents**, although a correct candidate was available. The original MLP selected about 150 Hz and GRU about 100 Hz. After augmentation the MLP still failed this case. It repaired the 300 Hz harmonic-change post-transition region from 0/95 to 95/95 correct frames; the original GRU achieved 76/95 there. The classical DSP also failed that post-change region, while YIN passed. All three learned checkpoints passed the defined median transition-lag components across two steps and six glides, but a passing signed-lag component does not erase pitch failures. The 14 original cases are now regression cases, not an independent augmentation holdout.

One final MLP configuration added 128 analytic sequences (320 seconds), repeated four times for scaling/training, about 15.70% of effective feature-frame mass. It kept the 57 features, six candidates, 32-unit layers, seed, optimizer and 12 epochs. Repetition changes training weight, not sample independence. A disjoint 32-sequence challenge was frozen before fitting and generated only after checkpoint freeze: 28 voiced and four unvoiced cases, 80 seconds total.

| Synthetic challenge | Mean voiced success % | Worst voiced case % | Voiced cases below 80% |
|---|---:|---:|---:|
| Original MLP | 72.95 | 0.00 | 13 |
| Augmented MLP | 88.07 | 2.90 | 4 |
| DSP | 75.99 | 0.00 | 14 |
| YIN | 99.82 | 98.76 | 0 |
| Fixed continuity | 62.97 | 0.00 | 12 |

The augmented MLP failed both the ≥90% equal-scenario mean and ≥80% every-voiced-case requirements. Its worst case was 2.9046%; four cases failed the individual threshold. All five compared methods had zero false voicing on the four unvoiced challenge cases (964 frames). YIN passed this challenge component but remains worse on natural-data joint error, especially under first difference. No method is declared universally robust. The augmented MLP also failed the ≥95% requirement for every original stationary case. Selection stopped: no further architecture/candidate search, seed confirmation, calibration, or sealed-test scoring followed.

**Browser evidence below is historical and functional, not a performance qualification.** The separately maintained [streaming report](STREAMING_RESULTS.md) owns later runtime and sustained-soak evidence; it does not change the model rejection.

The cited Chrome 155 / ONNX Runtime Web 1.23.2 WASM run used synthetic replay. Its manifest explicitly allows concurrent MPS training. A 3.5-second DSP replay with a 300 ms worker stall produced 312 accepted frames: frame service p50/p95/p99/max 0.40/1.20/1.40/4.20 ms; complete-block service 0.90/2.70/3.00/4.20 ms. Valid mapped sample-availability-to-main-result latency was 34.17/42.57/43.77/44.27 ms. These short-run observations do not establish an exclusive-hardware runtime gate or hardware input delay.

That replay reached four PCM and four result credits, dropped 11 transport blocks (10,560 samples) and four stale blocks (3,840 samples), and recovered a new segment. The final worklet snapshot counted 640 additional partial samples. A separate 2.2-second blocked-main replay reached four result credits, dropped 13 input blocks (12,480 samples) for result backpressure, rejected eight stale result frames, and recovered; 23 accepted frames had unavailable timing and are excluded from the valid-latency distribution. Both worker and worklet stop acknowledgements completed. A random-weight export spike matched 24 browser steps within 1.49e−7 absolute error with zero decision mismatches, including three all-masked steps. It is not trained-model quality or complete-stream learned-model runtime evidence. No 30-minute soak or live-reference microphone result is established by these artifacts.

**Identity and limits.** MLPs have 3,143 parameters; GRU has 8,423. All fits used local PyTorch 2.9.1 MPS on an Apple M4 Max, seed 20261022, Adam 0.001, 12 fixed epochs, batch 64, 128-frame chunks and train-only scaling. Exact model configurations, training/evaluation source hashes, data/source/split hashes, prediction freezes, generator plans, browser artifacts, and all metric artifacts are in the summary.

| Artifact | SHA-256 |
|---|---|
| Original MLP weights | `06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d` |
| Original GRU weights | `c7412797c3360a6234dd468f22d5b03694e2ea996dd1a06269e0f42d231dd614` |
| Augmented MLP weights | `9c18d9aca2829071dfa869a99db6c53e891216b6cf17d7f1cf59f0166d33aee2` |
| Acquisition manifest | `50d25236779d5cd95d3eeef58826c5f30825cb733af10d70d664108d8df66f28` |
| Training features manifest | `35e92d701dcd52891c256fa99b9567c5db4d80e14f4ef71858822b470edf5a87` |
| Development features manifest | `31f51e93b59f9a70c6f4b8c66cbf20b84fef900e44c75b8134aefad0cade3e8c` |
| CREPE tiny weights | `d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432` |

The reference is an algorithmic estimate from a separate laryngograph sensor. Nominal timing has source and selected-cycle support, but exact high-pass compensation and transition correspondence remain unresolved. The selected subset, only two development pairs, development reuse, single training seed, and synthetic corruptions preclude population or clinical claims. No confidence intervals, calibrated F0 correctness probabilities, seed confirmation, or correctness/coverage qualification are established. Voice duration and coverage aggregate contracts are not established from these short public utterances. Qualification stopped before live microphone demonstration or sealed-test evaluation. CREPE training overlap with PTDB is not proven absent. No sealed-test audio or outcomes were used. The [PTDB rights/source audit](DATA_AUDIT.md) preserves corpus licensing separately from repository code and comparator licenses.

Raw audio, reference trajectories, complete predictions and weights remain in ignored local research artifacts. Public hashes identify saved bytes; they do not make these artifacts downloadable or reproduce an uncommitted experiment from its base commit alone. The runs bind source snapshots based on 7e10705a562688f8582ec99c94495f9ba0b645ad. Reproduction requires the pinned sources, permitted corpus acquisition and frozen configurations. The result supports a bounded systems implementation and a documented decision to reject learned deployment; it does not validate diagnosis, disease biomarkers, or real-world microphone robustness.
