# Training-only laryngograph timing sensitivity

The selected strong-cycle evidence supports retaining the nominal reference grid `16 + 10*k` ms. Centered sensor estimates agree closely at that grid, whereas trailing estimates exhibit the expected support lag. This does not resolve the exact provider filter implementation, establish accuracy on weak/aperiodic speech, or close G1. No primary score or timestamp was changed; the original candidate-coverage gate remains failed.

## Fixed protocol and retained failed attempt

Only the original 24-recording training cohort is used. Offsets `[-30, -20, -10, 0, 10, 20, 30]` ms were fixed before reference comparison; positive means querying later raw LAR relative to the nominal reference time. Both derivative polarities are retained. There is no per-recording offset choice, offset-based model selection, development/test outcome access, or inference use of the sensor.

The first protocol used a centered 9-sample Savitzky–Golay derivative. It yielded zero common eligible frames in both polarities. Its result remains in `benchmarks/voice-prosody/lar-alignment-sensitivity.json`, and its exact source is `phenometrix_prosody/alignment_v1.py` (SHA-256 `348ce6b7a46f8ebf7d44c38fdb440130b657c207da1df8ab7c975155ef6f4111`). Raw-only inspection found many closely spaced extrema and very few repeatable accepted cycles. This failed attempt supplies no timing agreement.

Before the second reference comparison, revision 2 changed only derivative/smoothing width to 97 samples (2 ms; ±1 ms local filter support) and corresponding edge exclusion. All remaining criteria stayed fixed: degree-2 derivative, peak prominence at least six residual-MAD noise units, 50–700 Hz cycles, three neighboring periods with maximum/minimum ratio at most 1.15, and both neighboring detrended 64-phase-sample waveform correlations at least 0.85. This exploratory detector revision can prefer smoother or lower-frequency cycles and is not a population-level reference estimator.

Like-event inverse intervals estimate cycle-average frequency, not pointwise physiological F0. Alongside interval frequency, the diagnostic integrates piecewise cycle frequency over centered 32 ms, trailing 32 ms, centered 40 ms and trailing 40 ms windows, requiring 90% strong-cycle coverage. All 35 offset/support combinations use the same finite reference-voiced frame set within each polarity. A separate transition stratum is defined only from raw cycles changing at least 80 cents over 40 ms. Both complete sensor branches run before reference values are parsed.

## Observed sensitivity

The positive-polarity common set contains 741 of 4,014 reference-voiced rows (18.46%), spanning 16 recordings and 9 speakers. The negative-polarity set contains 615 rows (15.32%), spanning 18 recordings and 10 speakers. These sets overlap and must not be summed as independent observations. Neither selected common set contains a reference-unvoiced row. The raw-transition subsets contain 222 and 165 rows, respectively. The selection excludes much of the corpus and cannot qualify voicing boundaries, weak phonation or silence.

Table entries are median absolute cents error, positive / negative derivative polarity. Within each polarity the denominator is identical for every cell.

| Raw query offset | Cycle interval | Centered 32 ms | Trailing 32 ms | Centered 32 ms, raw transitions |
|---:|---:|---:|---:|---:|
| -30 ms | 42.53 / 39.42 | 41.86 / 35.04 | 56.97 / 52.30 | 90.41 / 81.85 |
| -20 ms | 30.57 / 30.11 | 29.36 / 25.51 | 48.70 / 41.18 | 68.01 / 58.62 |
| -10 ms | 18.48 / 20.09 | 17.61 / 15.80 | 36.93 / 30.84 | 40.53 / 34.75 |
| 0 ms | 8.13 / 13.66 | 5.04 / 7.68 | 24.80 / 21.99 | 11.54 / 11.85 |
| 10 ms | 10.33 / 15.90 | 7.29 / 9.93 | 12.34 / 11.48 | 18.86 / 21.39 |
| 20 ms | 21.36 / 24.97 | 19.31 / 17.65 | 2.58 / 7.34 | 47.02 / 48.14 |
| 30 ms | 32.55 / 32.57 | 33.01 / 26.92 | 12.33 / 13.34 | 73.12 / 69.93 |

At zero offset, centered 32 ms estimates are within 50 cents on 99.33% / 98.86% of selected frames; the 95th-percentile errors are 21.35 / 30.69 cents. Equal-speaker mean median errors are 5.56 / 7.78 cents, so the low pooled error is not solely due to one speaker. The raw-transition subset retains 97.75% / 96.36% within 50 cents at zero offset.

Trailing 32 ms at zero offset has median errors 24.80 / 21.99 cents, compared with 2.58 / 7.34 cents at the fixed +20 ms query. This is consistent with the support of a trailing window lying earlier than its reported endpoint. Centered 40 ms at zero offset has 5.06 / 7.59 cents error; trailing 40 ms at +20 ms is mathematically the same support and produces the same values. Consequently, a shifted trailing score cannot by itself establish a reference clock error. No optimum is selected or deployed from this sensitivity table.

## Authoritative extractor source

The [PTDB technical report](https://www2.spsc.tugraz.at/databases/PTDB-TUG/DOCUMENTATION/PTDB-TUG_REPORT.pdf) specifies ESPS/RAPT with 32 ms windows and 10 ms frames. The [official Snack manual](https://www.speech.kth.se/snack/man/snack2.2/tcl-man.html#spitch) gives the first estimate at half the window length, supporting the nominal 16 ms start. The provider does not identify its exact Snack release or high-pass filtering invocation.

The [official Snack download page](https://www.speech.kth.se/snack/download.html) links the [2.2.10 source archive](https://www.speech.kth.se/snack/dist/snack2.2.10.tar.gz), inspected as source only without installation or compilation. Archive size is 1,798,252 bytes, SHA-256 `4bfe764547ab92ba58f43b77366dbb7c7b3512d65a27cdbf9e585a9cb64ce81e`. In `generic/jkGetF0.c` (SHA-256 `118b5fd6813bdd25343b0e557b09ffea0fbb7c2828a1b6d2353d16dc8a27e2e3`), lines 82–85 default to 50–550 Hz and 10 ms frame steps. Lines 154–159 accept the requested window length; lines 796–804 round the frame and window samples and add maximum lag to correlation support. Lines 845–855 add half the downsampling filter length to the required tail; lines 763–770 count complete frames by integer division.

At 48 kHz with a 32 ms window and default minimum 50 Hz, these equations give 1,536 window samples, maximum lag 960, correlation support 2,497, and downsampling tail 120, totaling 2,617 samples. The predicted reference count `floor((N − 2617) / 480)` matches all 24 pilot files exactly. This explains the observed 54 ms gap from the last nominal reference target to each file end. The source default of 50 Hz also explains why references below the manual’s stated 60 Hz default need not imply an undocumented pitch override. Agreement with these equations supports the file/frame bookkeeping; it does not prove the provider used this exact release.

The provider’s high-pass Kaiser filter remains unresolved: order/taps convention, causal versus delay-compensated invocation and cropping are not specified. A roughly 25 ms causal delay cannot be assumed. The selected raw-cycle comparison supplies evidence against imposing a universal ±20–30 ms clock correction, but does not independently establish the processing provenance or a guaranteed error bound for all reference transitions.

## Reproduction and evidence identities

From the repository root, using the existing local environment:

```sh
PYTHONPATH=research/voice-prosody research/voice-prosody/.venv/bin/python -m phenometrix_prosody.alignment
PYTHONPATH=research/voice-prosody research/voice-prosody/.venv/bin/python -m pytest research/voice-prosody/tests/test_alignment.py -q
```

The module refuses any cohort or manifest entry not designated training and checks LAR/reference SHA-256 identities. Synthetic tests check physical period, fixed-offset sign, gap exclusion, common denominators and the identity of trailing/shifted-centered supports. Full per-recording, per-speaker and aggregate results, including uninformative records, are retained without raw waveforms or full trajectories. Re-running changes the retrieval/run timestamp and therefore the output-file digest.

| Evidence | SHA-256 |
|---|---|
| Revision 2 source | 4100cf47c9f01a1a741b86cebe96f4f7cb2d5de5851d938628ecb8d83772a88d |
| Cohort evidence | 4a0ca55c4558ed761b654bfbf01a1edbe2a09b413e3db53afb29be05fcf087c6 |
| Revision 1 result | 00e5bd2dcff3c15ac604be9133a71e74b9d99ee74a4f4c2b90b974982b5b1d48 |
| Revision 2 result | 6bfeb2d872007570f3f79961ebd7cbcb62eb6554cc66d4def68c76186ee79088 |
