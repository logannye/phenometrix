# Protocol registry

The active nonclinical protocol pack is the immutable
`AMBIENT_LOCAL_PROTOCOL_PACK` in
`packages/contracts/src/ambient-protocol.ts`. It defines the local source and
consent policy, five-minute limit, quality thresholds, report ordering, and
exactly 27 ambient metrics.

The JSON files in this directory are archival guided-demo fixtures. They are
not imported by the live browser, are not accepted as ObservationV3 protocol
identity, and are no longer part of the active structure validator.

No clinical protocol pack is implemented or validated. A future clinical pack
would require an intended use, population, reference standard, analytical and
clinical evidence, permitted claims, prohibited extrapolations, and governed
human workflow.
