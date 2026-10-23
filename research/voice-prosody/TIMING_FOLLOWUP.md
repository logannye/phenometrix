# Independent clock diagnosis and render-freshness follow-up

The operational freshness gate now uses a lease issued and verified on the same
main-thread clock. A delivered frame must carry a lease issued before the first
sample of its current PCM block and arrive within 100 ms of that issuance.
This removes the fixed audio/performance projection from the admission decision.
It bounds the age of the frame's last required **software-rendered input** in the
reported main performance coordinate. Here rendered input means the worklet
observing input during its process callback; upstream generation or buffering
and physical acquisition are outside this bound. Microphone acquisition time
and timestamp precision remain unknown. The DSP estimator and its numerical time semantics
remain unchanged.

The earlier 30-minute rise in mapped arrival time is preserved in
[STREAMING_RESULTS.md](STREAMING_RESULTS.md). Its physical cause is still
unresolved. The new implementation does not fit away that rise, periodically
re-anchor it, or turn output-device timestamps into input latency.

## Independent diagnosis before the gate change

The [follow-up protocol](FOLLOWUP_PROTOCOL.md) was declared before diagnosis.
Three separate observations were collected once per second: a bracketed main
read of AudioContext time, the browser's output-device timestamp, and a
main request / worklet render callback / main reply bracket. Each path has one
outstanding probe; expiry cannot create another credit. The output timestamp is
explicitly output-only. The [Web Audio specification](https://webaudio.github.io/web-audio-api/#dom-baseaudiocontext-currenttime)
defines currentTime in the rendering timeline, which need not be synchronized
with other clocks. None of these observations establishes hardware input delay.

A local ten-minute replay on the original mapped-age policy completed with
59,966 accepted frames, 29,990 block records, and 600 observed probes plus one
stopped probe. The source included one deliberate 300 ms worker stall at five
minutes. Complete-block p99 was 3.3 ms and maximum 5.2 ms. All raw records and
source identities passed independent reconstruction.

This run **did not reproduce the earlier upward trend**. First to last minute,
context-read residual medians changed from 27.03 to 24.67 ms, mapped arrival from
34.17 to 33.47 ms, and receipt-to-main from 1.0 to 1.2 ms. Descriptive residual
slopes were −1.65 ppm for context reads, −1.50 ppm for render-bracket midpoints,
and −2.13 ppm for output-device time. These are descriptive slopes, not a clock
calibration. The intersection of all render-bracket affine constraints was
infeasible; no outlier removal or fitted correction was applied.

The run retained 6,869 unavailable fixed-projection timing records. In each,
main arrival minus the converted worker-ready timestamp was
−0.00004863739013671875 ms. Other checked elapsed components were nonnegative.
The values were preserved without clamping. This cross-realm timestamp effect
is another reason to separate same-main-clock admission from unqualified
cross-clock diagnostics; it does not identify the old 30-minute trend's cause.

## Current operational contract

The [lease contract](RENDER_FRESHNESS_CONTRACT.md) was frozen before implementation.
Main issues a unique lease every 25 ms when its single control credit is free.
Each lease expires 100 ms after issuance. The worklet snapshots the active lease
at the **first sample** of a 20 ms block and carries that same lease through the
whole block. Renewal partway through a block cannot certify earlier input. Both
normal frame endpoints inherit that conservative lease, while earlier samples
in the 40 ms analysis support may predate issuance.

Worker admission rejects missing, malformed, expired, retired or inconsistent
leases and records the drop and state reset. Final host delivery checks the
original lease against a bounded eight-entry issuance ledger and uses actual
main receipt time. Forged fields, missing evidence and old identities withhold
output. Unknown timing precision is null. Expiry never releases a transport
credit. A matching accepted or retired acknowledgment may release the one
outstanding request, including a pre-reset acknowledgment already in transit;
reset still invalidates all verification identities.

Four PCM credits, four result-batch credits, one lease request and eight ledger
entries bound their respective transports. The lease acknowledgment does not
wait for a DSP result credit. Stop ends issuance before draining queued results;
final disposal invalidates verification. Lease renewal alone never resets DSP,
changes samples, or moves the fixed audio/performance anchor. Fixed projections,
including unavailable or rising values, remain raw diagnostics.

Production capture and research replay explicitly require
`required-render-lease-v1`. Only numerical test fixtures may explicitly opt into
`unqualified-legacy-fixture`. Worker protocol v3 and timing v2 distinguish the
new internal contract; normal DSP remains `browser-voice-dsp@1.1` with legacy
window-start `tMs`. Matched worker/worklet assets must ship together.

## Confirmation evidence

The short combined-stall browser check delivered 571 certified frames, withheld
eight expired host frames and five expired worker blocks, and recovered the
known 200 Hz tone. Independent reconstruction verified all 885 raw records.
A separate blocked-main check delivered 176 certified frames and withheld eight
expired frames. These functional fixtures do not measure physical input latency.

The final sustained confirmation and its source identities are recorded in
[timing-followup-summary.json](../../benchmarks/voice-prosody/timing-followup-summary.json).
Its raw records remain local and ignored. Complete-block service measures the
accepted PCM handler, including feature preparation, PCM recycle and data-envelope
post; it excludes waiting before the handler and its timing-completion post.
The deliberate external stalls appear in drops, withholding and recovery, not
as block service. These descriptive same-machine runs retain
`timingBenchmark: false`; they do not establish an exclusive-hardware guarantee.

### Thirty-minute confirmation

The source-frozen local Chrome run completed 1,800.0015 seconds of paced
source time, with a 300 ms worker stall at minute 10 and a 350 ms main-thread
stall at minute 20. It delivered **179,927 frames**, all certified and all reporting
the known 200 Hz tone. Recorded render-age bounds were 32.9 ms at median,
48.8 ms at p99 and **53.8 ms maximum**. Complete processed-block service was
4.3 ms at p99 and **8.7 ms maximum**. No complete block exceeded the 10 ms service
budget or 20 ms cadence. Four individual frame-preparation times exceeded the
narrower 5 ms budget; none exceeded the 10 ms frame cadence.

The run preserved **271,713 raw records in 2,718 batches**: 179,927 delivered
frames, eight discarded frames, 89,977 block records and 1,801 clock observations.
Independent arithmetic, count, hash and host-withholding audits passed. There
were 1,800 observed render probes and one stopped probe; the first output-device
timestamp was unavailable. All before/after source hashes matched.

The worker withheld five expired PCM blocks (4,800 samples). Result backpressure
dropped 13 blocks (12,480 samples); the worklet dropped ten blocks (9,600 samples).
The host withheld eight expired results. These dispositions preserve all 90,000
source blocks: 89,972 processed + 5 expired + 13 result-backpressure drops + 10
worklet drops. All 179,935 prepared frames reconcile as 179,927 delivered + 8
withheld. Recovery produced three delivered analysis segments. No absent input
was represented as silence.

Both stop acknowledgments completed. Their terminal snapshots still recorded
one PCM block and one result batch in flight at the instant of observation;
these values are preserved, not rewritten as zero or interpreted as proof of
all credits being returned. The final raw frame accounting reconciles separately.
PCM and result credit maxima were four each. All 71,988 issued leases were
acknowledged, with a maximum of one pending request and eight ledger entries.
Raw telemetry reached at most 100 pending records against its 200-record cap.

| Source minutes | Certified frames | Render-age bound p50 / p99 / max, ms | Unqualified mapped arrival p50 / p99, ms | Complete block service p99 / max, ms |
|---|---:|---|---|---|
| 0–5 | 29,996 | 32.80 / 47.90 / 52.50 | 34.17 / 43.47 | 4.00 / 5.60 |
| 5–10 | 30,000 | 33.60 / 49.30 / 53.80 | 34.07 / 43.07 | 4.30 / 7.00 |
| 10–15 | 29,969 | 33.70 / 49.70 / 52.80 | 33.97 / 43.17 | 4.20 / 6.20 |
| 15–20 | 30,000 | 32.90 / 49.00 / 53.00 | 33.97 / 42.97 | 4.50 / 7.60 |
| 20–25 | 29,961 | 31.60 / 46.50 / 49.40 | 30.57 / 40.07 | 0.80 / 1.00 |
| 25–30 | 30,000 | 32.90 / 48.80 / 53.00 | 33.97 / 42.97 | 4.60 / 8.70 |
| 30:00 endpoint | 1 | 30.60 / 30.60 / 30.60 | 25.97 / 25.97 | — |

Frame windows use the exclusive support endpoint, so the last frame exactly at
1,800 seconds is retained in its own boundary row. Block windows use support
start. Quantiles use nearest rank, without interpolation. All mapped arrival
timings in this run were nonnegative and available; availability does not qualify
the fixed projection as physical input time.

This run also did not reproduce the original rising mapped arrival trend.
Context, render-midpoint and output-device descriptive residual slopes were
−0.051, +0.068 and −0.087 ppm. All observed render brackets still had no common
affine mapping; the main stall's wide bracket remains included. The earlier
trend's physical cause remains unresolved. The accepted operational result is
freshness verification independent of that projection, with the original
uncertainty preserved.

## Reproduction and limits

Use the pinned local environment and prepare the existing research assets as
documented in the [research README](README.md). For a new output directory,
run the real browser replay with `VOICE_RESEARCH_CLOCK_DIAGNOSTICS=1`,
`VOICE_RESEARCH_METHOD=dsp`, `VOICE_RESEARCH_DURATION_MS=1800000`,
`VOICE_RESEARCH_OVERLOAD_AT_MS=600000`,
`VOICE_RESEARCH_OVERLOAD_DURATION_MS=300`,
`VOICE_RESEARCH_MAIN_STALL_AT_MS=1200000` and
`VOICE_RESEARCH_MAIN_STALL_DURATION_MS=350`. Select the
`real AudioWorklet/worker replay` test with the research Playwright config.
Run one timed process and preserve before/after source hashes.

From `research/voice-prosody`, independently reconstruct a completed saved run:

```sh
.venv/bin/python -m phenometrix_prosody.audit_browser --run runs/NEW-RUN --output runs/NEW-RUN/independent-audit.json
.venv/bin/python -m phenometrix_prosody.audit_clock --run runs/NEW-RUN --output runs/NEW-RUN/clock-audit.json
```

These results concern a looped synthetic tone through Chrome rendering and
transport. They do not establish browser heap-growth bounds, physical microphone
latency, live-speech accuracy, clinical validity, or sustained runtime of the new
offline MLP/YIN hybrid. Ordinary desktop background load and thermal state are
uncontrolled. No hosted computation was used.
