# Trajectory Core

`@phenometrix/trajectory-core` implements the prototype's narrow Personal
Trajectory slice: a deterministic, previous-visit comparison over two validated
`ObservationV3` artifacts.

The package applies exact compatibility rules and returns a terminal decision
for every allowlisted metric. It computes only `current - reference` in the
metric's native unit. It does not infer change in health, classify a condition,
score severity, interpolate missing measurements, or establish clinical
significance.

Inputs and outputs remain structured, local, and page-memory-only. Callers must
provide an explicitly accepted reference and a current demo context; this
package performs no persistence or identity verification.
