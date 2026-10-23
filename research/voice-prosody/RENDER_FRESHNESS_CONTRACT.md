# Render freshness lease contract and acceptance plan

Declared 22 October 2026 before implementation or confirmation of the revised
operational gate. The original fixed-anchor timing evidence remains unchanged.
This contract concerns software render freshness, not microphone acquisition,
output-device latency, browser timestamp precision, or clinical validity.

## Observable claim

The main thread issues a lease with a unique ID, capture epoch, mapping identity,
its performance time origin, issue time, and expiry exactly 100 ms later. The
worklet may associate that lease only with input rendered after receiving it.
A result delivered no later than expiry has a conservative upper bound of
`main result receipt - main lease issue <= 100 ms` on the age of its last required
rendered input, measured in one main-thread performance coordinate. Timer
precision remains unknown. The bound concerns the frame endpoint; earlier
samples in its 40 ms support can predate issuance.

The worklet snapshots its active lease when the first sample of a new 20 ms PCM
block is observed. It carries that same, oldest lease for the entire block,
including either of its two normal output-frame endpoints. A lease received
partway through a block cannot qualify that block retrospectively. Missing
lease at block start remains missing for the whole block. Lease changes do not
reset DSP state, alter samples, move the fixed audio/performance anchor, or
change the DSP processor identity and legacy window-start `tMs`.

The worker rejects missing, malformed, retired, expired, or clock-inconsistent
leases before analysis, with explicit counts and loss/reset behavior. Its
admission check converts its performance coordinate using the recorded main
origin, while the final host gate uses the original issuing realm directly.
The host verifies returned lease fields against its own bounded issuance ledger,
then checks final receipt. No result arrival, inferred queue delay, or audio
clock rate enters lease creation. Old fixed-anchor durations remain separate
raw diagnostics, including negative or rising values; they no longer authorize
freshness. Missing evidence is never silently treated as fresh.

## Bounds and lifecycle

The target renewal cadence is 25 ms. Only one main-to-worklet lease request may
be outstanding; a timeout cannot create another credit. The matching accepted
or retired ACK returns that credit. The worklet keeps an active lease, a pending
request and the current block snapshot. Its ACK path is independent of PCM
credits and the DSP worker. Worker admission receives leases with PCM, avoiding
a second periodic queue to the worker. The host retains at most eight issued
leases; eviction can only withhold a result, never qualify an unknown lease.

Capture/mapping reset invalidates old leases. A pending request is acknowledged
as retired before being cleared, preventing stranded credit. Late and duplicate
ACKs cannot release a newer request. Context suspension or a blocked main thread
stops valid renewal; resume requires a fresh lease before delivery. All control
and terminal paths remain bounded and independent of result credit. Missing or
expired leases cause explicit input drops and fresh segments on recovery;
result drops reset state through the existing identity-bound ACK mechanism.

Production and the research browser replay require the lease policy explicitly.
Legacy numerical worker fixtures may select a clearly named unqualified fixture
policy to preserve their old arithmetic/support tests. Such fixtures never
produce a valid freshness certificate and cannot be cited as freshness evidence.
No implicit fallback from required leases to the legacy policy is allowed.

## Acceptance criteria fixed before confirmation

Deterministic tests must cover first versus second frame in a block, partial
blocks spanning renewal, missing startup lease, exact expiry, future timestamps,
wrong origin/epoch/mapping, forged lease fields, timeout without extra credit,
retired and duplicate ACKs, main and worker stalls, reset/disposal, and fresh
recovery. A changing fixed audio/performance offset must neither expire a valid
lease nor rescue an expired lease. Delivered contiguous DSP numerical fields
must retain their baseline equality.

Actual-browser validation must retain all independent clock probes and fixed
anchor diagnostics. A 30-minute DSP confirmation must report complete raw
telemetry and both stop acknowledgments, all software freshness certificates
within 100 ms, every withheld/drop reason, bounded PCM/result/lease credits,
known-tone recovery after controlled worker and main stalls, and every
five-minute window. Complete-block p99 must be at most 10 ms and no processed
block may exceed its 20 ms cadence. Preserve failures and timing exclusions.

The renewed gate establishes an operational software bound only. It does not
repair or validate the old absolute sample-availability projection, infer a
physical acquisition timestamp, transfer model qualification, or claim browser
heap-growth bounds. Performance confirmation is run locally and sequentially
without agent training or other agent benchmarks running concurrently.
