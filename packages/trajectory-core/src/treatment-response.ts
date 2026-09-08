import {
  TreatmentResponseProtocolV1Schema, TreatmentResponseSpecificationV1Schema,
  TreatmentResponseSnapshotInputV1Schema, TreatmentResponseSnapshotV1Schema,
  TreatmentResponseRunV1Schema, canonicalTreatmentResponseJson,
  sealTreatmentResponseArtifact, treatmentResponseDigest, verifyTreatmentResponseArtifact,
  type TreatmentResponseSnapshotInputV1, type TreatmentResponseSnapshotV1,
  type TreatmentResponseProtocolV1, type TreatmentResponseSpecificationV1,
  type TreatmentResponseRunV1, type TreatmentResponseRowV1, type TreatmentResponsePointV1,
  type TreatmentResponseExclusionCodeV1, type DurableObservationV1, type DurableMetricV1,
  type VersionedArtifactRefV1
} from "@phenometrix/contracts";

export const TREATMENT_RESPONSE_ENGINE_VERSION = "descriptive-treatment-alignment.1.0.1";
const DAY = 86_400_000;
const lexical = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;
const same = (left: unknown, right: unknown): boolean => canonicalTreatmentResponseJson(left) === canonicalTreatmentResponseJson(right);

function freeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

interface Revision { revisionId: string; supersedesRevisionId: string | null; recordedAt: string }

/** Resolve a complete append-only history. Forks and dangling revisions require reconciliation. */
function resolveRevisions<T extends Revision>(records: readonly T[], identity: (record: T) => string): Map<string, T> {
  const byRevision = new Map(records.map(record => [record.revisionId, record]));
  const successor = new Map<string, string>();
  const grouped = new Map<string, T[]>();
  for (const record of records) {
    const group = grouped.get(identity(record)) ?? [];
    group.push(record); grouped.set(identity(record), group);
    if (record.supersedesRevisionId) {
      const previous = byRevision.get(record.supersedesRevisionId);
      if (!previous || identity(previous) !== identity(record) || Date.parse(previous.recordedAt) > Date.parse(record.recordedAt)) throw new Error("treatment-response-invalid-revision-chain");
      if (successor.has(previous.revisionId)) throw new Error("treatment-response-revision-fork");
      successor.set(previous.revisionId, record.revisionId);
    }
  }
  const result = new Map<string, T>();
  for (const [id, group] of grouped) {
    const roots = group.filter(record => record.supersedesRevisionId === null);
    const leaves = group.filter(record => !successor.has(record.revisionId));
    if (roots.length !== 1 || leaves.length !== 1) throw new Error("treatment-response-invalid-revision-chain");
    let cursor = roots[0].revisionId;
    const seen = new Set<string>();
    while (!seen.has(cursor)) {
      seen.add(cursor);
      const next = successor.get(cursor);
      if (!next) break;
      cursor = next;
    }
    if (seen.size !== group.length) throw new Error("treatment-response-invalid-revision-chain");
    result.set(id, leaves[0]);
  }
  return result;
}

function assertRevisionHistory(snapshot: TreatmentResponseSnapshotInputV1): void {
  resolveRevisions(snapshot.observations, record => record.observationId);
  resolveRevisions(snapshot.treatments, record => record.treatmentId);
  resolveRevisions(snapshot.contexts, record => record.contextId);
}

/** Freeze only knowledge available by asOf; no wall clock or storage access. */
export async function createTreatmentResponseSnapshot(input: TreatmentResponseSnapshotInputV1): Promise<TreatmentResponseSnapshotV1> {
  const parsed = TreatmentResponseSnapshotInputV1Schema.parse(input);
  const cutoff = Date.parse(parsed.asOf);
  const byRevision = <T extends Revision>(records: T[]): T[] => records.filter(record => Date.parse(record.recordedAt) <= cutoff).sort((a, b) => lexical(a.revisionId, b.revisionId));
  const content = {
    ...parsed,
    observations: byRevision(parsed.observations), treatments: byRevision(parsed.treatments), contexts: byRevision(parsed.contexts),
    consents: parsed.consents.filter(record => Date.parse(record.grantedAt) <= cutoff).sort((a, b) => lexical(a.consentId, b.consentId)),
    bindings: parsed.bindings.filter(record => Date.parse(record.recordedAt) <= cutoff).sort((a, b) => lexical(a.bindingId, b.bindingId))
  };
  assertRevisionHistory(content);
  const snapshot = await sealTreatmentResponseArtifact({
    schemaVersion: "phenometric.treatment-response-snapshot.v1" as const,
    snapshotId: `snapshot-${await treatmentResponseDigest(content)}`,
    contentSha256: "0".repeat(64), ...content
  });
  return freeze(TreatmentResponseSnapshotV1Schema.parse(snapshot));
}

function ref(id: string, artifact: { version: string; contentSha256: string }): VersionedArtifactRefV1 {
  return { id, version: artifact.version, contentSha256: artifact.contentSha256 };
}

/** Compatibility includes actual acquisition mode/device class, not only algorithm semver. */
export async function treatmentResponseCompatibilityFingerprint(observation: DurableObservationV1, metric: DurableMetricV1): Promise<string> {
  const { clockUncertaintyMs: _clock, rawMediaRetained: _media, ...capture } = observation.capture;
  return treatmentResponseDigest({ measurementProtocolRef: observation.measurementProtocolRef, capture,
    metric: { metricCode: metric.metricCode, modality: metric.modality, unit: metric.unit, context: metric.context,
      algorithmVersion: metric.algorithmVersion, processorRef: metric.processorRef } });
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
}

function observationReasons(observation: DurableObservationV1, snapshot: TreatmentResponseSnapshotV1, protocol: TreatmentResponseProtocolV1, specification: TreatmentResponseSpecificationV1, metric: DurableMetricV1): TreatmentResponseExclusionCodeV1[] {
  const reasons: TreatmentResponseExclusionCodeV1[] = [];
  const start = Date.parse(observation.startedAt);
  const end = Date.parse(observation.endedAt);
  const asOf = Date.parse(snapshot.asOf);
  const consent = snapshot.consents.find(record => record.consentId === observation.consentId);
  if (!consent) reasons.push("consent-missing");
  else if (!consent.permissions.capture || !consent.permissions.derivedAnalysis || !consent.permissions.derivedRetention || !consent.permissions.modalities.includes(metric.modality) || Date.parse(consent.grantedAt) > start || (consent.withdrawnAt !== null && Date.parse(consent.withdrawnAt) <= asOf) || (consent.expiresAt !== null && Date.parse(consent.expiresAt) <= Math.max(end, asOf))) reasons.push("consent-not-applicable");
  const binding = snapshot.bindings.find(record => record.bindingId === observation.bindingId);
  if (!binding) reasons.push("binding-missing");
  else {
    if (binding.status !== "verified" || binding.verificationMethod === "unverified" || (binding.verificationMethod === "synthetic-fixture" && observation.capture.sourceKind !== "synthetic") || Date.parse(binding.recordedAt) > start || (binding.revokedAt !== null && Date.parse(binding.revokedAt) <= asOf)) reasons.push("binding-not-verified");
    if (binding.encounterId !== observation.encounterId) reasons.push("binding-encounter-mismatch");
  }
  if (!protocol.permittedSourceKinds.includes(observation.capture.sourceKind)) reasons.push("source-kind-not-permitted");
  if (!same(observation.measurementProtocolRef, protocol.measurementProtocolRef)) reasons.push("measurement-protocol-mismatch");
  if (observation.capture.clockUncertaintyMs > specification.maximumClockUncertaintyMs ||
    (observation.capture.clockSource?.kind === "synthetic" && observation.capture.sourceKind !== "synthetic")) reasons.push("clock-uncertain");
  return reasons;
}

export interface AnalyzeTreatmentResponseInput {
  snapshot: TreatmentResponseSnapshotV1;
  specification: TreatmentResponseSpecificationV1;
  protocol: TreatmentResponseProtocolV1;
}

/** Descriptive alignment only. Never fits a curve, imputes a visit or infers treatment efficacy. */
export async function analyzeTreatmentResponse(input: AnalyzeTreatmentResponseInput): Promise<TreatmentResponseRunV1> {
  const snapshot = TreatmentResponseSnapshotV1Schema.parse(input.snapshot);
  const specification = TreatmentResponseSpecificationV1Schema.parse(input.specification);
  const protocol = TreatmentResponseProtocolV1Schema.parse(input.protocol);
  for (const artifact of [snapshot, specification, protocol]) {
    if (!await verifyTreatmentResponseArtifact(artifact)) throw new Error("treatment-response-artifact-digest-mismatch");
  }
  if (!same(snapshot.scope, specification.scope)) throw new Error("treatment-response-scope-mismatch");
  const protocolRef = ref(protocol.protocolId, protocol);
  if (!same(specification.protocolRef, protocolRef)) throw new Error("treatment-response-protocol-reference-mismatch");
  const reconstructed = await createTreatmentResponseSnapshot({ scope: snapshot.scope, asOf: snapshot.asOf,
    observations: snapshot.observations, treatments: snapshot.treatments, contexts: snapshot.contexts,
    consents: snapshot.consents, bindings: snapshot.bindings });
  if (!same(reconstructed, snapshot)) throw new Error("treatment-response-noncanonical-snapshot");

  const latestObservations = resolveRevisions(snapshot.observations, record => record.observationId);
  const latestTreatments = resolveRevisions(snapshot.treatments, record => record.treatmentId);
  const latestContexts = resolveRevisions(snapshot.contexts, record => record.contextId);
  const anchor = latestTreatments.get(specification.anchorTreatmentId);
  const anchorReasons: string[] = [];
  if (!anchor) anchorReasons.push("treatment-missing");
  else {
    if (anchor.status !== "administered") anchorReasons.push("treatment-not-administered");
    if (anchor.kind !== "botulinum-injection") anchorReasons.push("treatment-kind-not-supported");
    if (anchor.verification !== "verified") anchorReasons.push("treatment-not-verified");
    if (anchor.effectiveTime.precision === "unknown") anchorReasons.push("treatment-time-unknown");
    if (anchor.effectiveTime.latest && Date.parse(anchor.effectiveTime.latest) > Date.parse(snapshot.asOf)) anchorReasons.push("treatment-time-in-future");
  }
  const anchorAvailable = anchor !== undefined && anchorReasons.length === 0;
  const anchorStart = anchorAvailable ? Date.parse(anchor.effectiveTime.earliest!) : NaN;
  const anchorEnd = anchorAvailable ? Date.parse(anchor.effectiveTime.latest!) : NaN;
  const contextFlags: TreatmentResponseRunV1["contextFlags"] = [];
  for (const record of [...latestObservations.values(), ...latestTreatments.values(), ...latestContexts.values()]) {
    if (record.supersedesRevisionId !== null) contextFlags.push({ revisionId: record.revisionId,
      kind: "source-correction", reason: "analysis-input-history-includes-corrected-source" });
  }
  const relevantChanges: { earliest: number | null; revisionId: string }[] = [];
  const nextCycleStarts: number[] = [];
  for (const treatment of latestTreatments.values()) {
    if (treatment.treatmentId === specification.anchorTreatmentId || treatment.status !== "administered") continue;
    const start = treatment.effectiveTime.earliest ? Date.parse(treatment.effectiveTime.earliest) : null;
    const end = treatment.effectiveTime.latest ? Date.parse(treatment.effectiveTime.latest) : null;
    if (anchorAvailable && start !== null && start > anchorEnd + specification.followupEndDay * DAY) continue;
    contextFlags.push({ revisionId: treatment.revisionId, kind: treatment.kind,
      reason: start === null ? "other-treatment-time-unknown" : end !== null && end < anchorStart ? "prior-treatment-carryover-not-estimated" : "concurrent-or-subsequent-treatment" });
    if (start === null || (end !== null && end >= anchorStart)) relevantChanges.push({ earliest: start, revisionId: treatment.revisionId });
    if (anchorAvailable && treatment.kind === "botulinum-injection" && treatment.courseId === anchor.courseId && treatment.verification === "verified" && start !== null && start > anchorEnd) nextCycleStarts.push(start);
  }
  for (const context of latestContexts.values()) {
    if (context.status !== "active") continue;
    const start = context.effectiveTime.earliest ? Date.parse(context.effectiveTime.earliest) : null;
    const end = context.effectiveTime.latest ? Date.parse(context.effectiveTime.latest) : null;
    if (anchorAvailable && start !== null && start > anchorEnd + specification.followupEndDay * DAY) continue;
    if (anchorAvailable && end !== null && end < anchorStart + specification.baseline.startDay * DAY) continue;
    contextFlags.push({ revisionId: context.revisionId, kind: context.kind, reason: "clinical-context-not-causally-adjusted" });
    relevantChanges.push({ earliest: start, revisionId: context.revisionId });
  }
  if (anchorAvailable && !contextFlags.some(flag => flag.reason === "prior-treatment-carryover-not-estimated")) contextFlags.push({ revisionId: anchor.revisionId, kind: "treatment-history", reason: "absence-of-recorded-prior-treatment-does-not-establish-washout" });
  contextFlags.sort((a, b) => lexical(`${a.revisionId}:${a.kind}`, `${b.revisionId}:${b.kind}`));

  const observations = [...snapshot.observations].sort((a, b) => Date.parse(a.startedAt) - Date.parse(b.startedAt) || lexical(a.revisionId, b.revisionId));
  const rows: TreatmentResponseRowV1[] = [];
  for (const definition of protocol.metrics) {
    const exclusions: TreatmentResponseRowV1["exclusions"] = [];
    const candidates: { observation: DurableObservationV1; metric: DurableMetricV1; fingerprint: string; days: { minimum: number; maximum: number } | null; baseline: boolean }[] = [];
    for (const observation of observations) {
      const reasons: TreatmentResponseExclusionCodeV1[] = [];
      if (latestObservations.get(observation.observationId)?.revisionId !== observation.revisionId) reasons.push("superseded");
      if (observation.status === "entered-in-error") reasons.push("entered-in-error");
      const metric = observation.metrics.find(candidate => candidate.metricCode === definition.metricCode);
      if (!metric) reasons.push("metric-missing");
      else {
        reasons.push(...observationReasons(observation, snapshot, protocol, specification, metric));
        if (metric.modality !== definition.modality || metric.unit !== definition.unit || metric.context !== definition.context || metric.algorithmVersion !== definition.algorithmVersion) reasons.push("metric-definition-mismatch");
        if (metric.status === "withheld") reasons.push("metric-withheld");
        else {
          if (metric.usableDurationMs < definition.minimumUsableDurationMs) reasons.push("insufficient-usable-duration");
          if (metric.technicalQualityScore === null || metric.technicalQualityScore < definition.minimumTechnicalQualityScore) reasons.push("technical-quality-failed");
        }
      }
      const days = anchorAvailable ? {
        minimum: (Date.parse(observation.startedAt) - observation.capture.clockUncertaintyMs - anchorEnd) / DAY,
        maximum: (Date.parse(observation.endedAt) + observation.capture.clockUncertaintyMs - anchorStart) / DAY
      } : null;
      const baseline = days !== null && days.maximum < 0 && days.minimum >= specification.baseline.startDay && days.maximum <= specification.baseline.endDay;
      const followup = days !== null && days.minimum >= 0 && days.maximum <= specification.followupEndDay;
      if (days !== null && !baseline && !followup) reasons.push(days.minimum < 0 && days.maximum >= 0 ? "treatment-time-overlap" : "outside-analysis-window");
      if (anchorAvailable && !baseline && specification.concurrentTreatmentPolicy === "exclude-after-change" && relevantChanges.some(change => change.earliest === null || Date.parse(observation.endedAt) + observation.capture.clockUncertaintyMs >= change.earliest)) reasons.push("concurrent-treatment-change");
      if (anchorAvailable && !baseline && nextCycleStarts.some(start => Date.parse(observation.endedAt) + observation.capture.clockUncertaintyMs >= start)) reasons.push("subsequent-treatment-cycle");
      if (reasons.length > 0) exclusions.push({ observationId: observation.observationId, revisionId: observation.revisionId, reasonCodes: [...new Set(reasons)] });
      else if (metric) candidates.push({ observation, metric, days, baseline, fingerprint: await treatmentResponseCompatibilityFingerprint(observation, metric) });
    }
    // A deterministic pre-treatment reference fixes the acquisition stratum. If
    // none exists, one follow-up stratum can still show absolute observed values.
    const reference = candidates.find(candidate => candidate.baseline) ?? candidates[0];
    const compatible = candidates.filter(candidate => {
      if (candidate.fingerprint === reference?.fingerprint) return true;
      exclusions.push({ observationId: candidate.observation.observationId, revisionId: candidate.observation.revisionId, reasonCodes: ["source-incompatible"] });
      return false;
    });
    const baselineCandidates = compatible.filter(candidate => candidate.baseline);
    const encounters = new Map<string, number[]>();
    for (const candidate of baselineCandidates) {
      const values = encounters.get(candidate.observation.encounterId) ?? [];
      values.push(candidate.metric.value!); encounters.set(candidate.observation.encounterId, values);
    }
    const sufficientBaseline = baselineCandidates.length >= specification.baseline.minimumObservations && encounters.size >= specification.baseline.minimumEncounters;
    const baselineValue = sufficientBaseline ? median([...encounters.values()].map(median)) : null;
    const points: TreatmentResponsePointV1[] = compatible.flatMap(candidate => {
      const delta = baselineValue === null ? null : candidate.metric.value! - baselineValue;
      if (delta !== null && !Number.isFinite(delta)) {
        exclusions.push({ observationId: candidate.observation.observationId, revisionId: candidate.observation.revisionId, reasonCodes: ["non-finite-delta"] });
        return [];
      }
      return [{
      observationId: candidate.observation.observationId, revisionId: candidate.observation.revisionId,
      encounterId: candidate.observation.encounterId, startedAt: candidate.observation.startedAt, endedAt: candidate.observation.endedAt,
      value: candidate.metric.value!, delta,
      modality: candidate.metric.modality, context: candidate.metric.context, algorithmVersion: candidate.metric.algorithmVersion,
      processorRef: candidate.metric.processorRef, usableDurationMs: candidate.metric.usableDurationMs,
      technicalQualityScore: candidate.metric.technicalQualityScore!, captureAdapterId: candidate.observation.capture.adapterId,
      captureAdapterVersion: candidate.observation.capture.adapterVersion, sourceKind: candidate.observation.capture.sourceKind,
      processorFingerprint: candidate.observation.capture.processorFingerprint,
      daysSinceTreatment: candidate.days,
      phaseIds: candidate.baseline || candidate.days === null ? [] : specification.phases.filter(phase => candidate.days!.minimum >= phase.startDay && candidate.days!.maximum < phase.endDay).map(phase => phase.phaseId),
      sourceWindowIds: candidate.metric.sourceWindowIds
    }]; });
    // Counts are encounters, not windows or repeat captures from the same visit.
    const phaseCoverage = specification.phases.map(phase => {
      const count = new Set(points.filter(point => point.phaseIds.includes(phase.phaseId)).map(point => point.encounterId)).size;
      return { phaseId: phase.phaseId, count, minimumCount: phase.minimumCount,
        status: count === 0 ? "missing" as const : count < phase.minimumCount ? "sparse" as const : "sufficient" as const };
    });
    exclusions.sort((a, b) => lexical(a.revisionId, b.revisionId));
    rows.push({ metricCode: definition.metricCode, label: definition.label, unit: definition.unit,
      status: points.length === 0 ? "no-comparable-data" : !sufficientBaseline ? "insufficient-baseline" : points.some(point => point.daysSinceTreatment !== null && point.daysSinceTreatment.minimum >= 0) ? "observed" : "insufficient-followup",
      baseline: { status: sufficientBaseline ? "available" : "insufficient-data", value: baselineValue,
        observationRevisionIds: baselineCandidates.map(candidate => candidate.observation.revisionId), encounterCount: encounters.size,
        compatibilityFingerprint: reference?.fingerprint ?? null,
        reasonCodes: sufficientBaseline ? [] : [!anchorAvailable ? "anchor-unavailable" : baselineCandidates.length < specification.baseline.minimumObservations ? "insufficient-baseline-observations" : "insufficient-baseline-encounters"] },
      points, exclusions, phaseCoverage,
      uncertainty: { measurementError: "unknown", minimumDetectableChange: "unknown", confidenceInterval: null }
    });
  }
  const specificationRef = ref(specification.specificationId, specification);
  const runId = `run-${await treatmentResponseDigest({ snapshot: snapshot.contentSha256, protocol: protocol.contentSha256, specification: specification.contentSha256, engine: TREATMENT_RESPONSE_ENGINE_VERSION })}`;
  const run = await sealTreatmentResponseArtifact({
    schemaVersion: "phenometric.treatment-response-run.v1" as const, runId, contentSha256: "0".repeat(64),
    scope: snapshot.scope, snapshotId: snapshot.snapshotId, snapshotSha256: snapshot.contentSha256,
    protocolRef, specificationRef, engineVersion: TREATMENT_RESPONSE_ENGINE_VERSION, generatedAt: snapshot.asOf,
    status: !anchorAvailable ? "anchor-unavailable" as const : rows.some(row => row.status === "observed") ? "available" as const : "insufficient-data" as const,
    anchor: { treatmentId: specification.anchorTreatmentId, revisionId: anchor?.revisionId ?? null, cycleId: anchor?.cycleId ?? null,
      effectiveTime: anchor?.effectiveTime ?? null, reasonCodes: anchorReasons },
    rows, contextFlags, clinicalClaim: "descriptive-only" as const, fittedModels: false as const
  });
  return freeze(TreatmentResponseRunV1Schema.parse(run));
}
