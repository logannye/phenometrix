import {
  TreatmentResponseRunV1Schema, TreatmentResponseReviewV1Schema,
  canonicalTreatmentResponseJson, verifyTreatmentResponseArtifact,
  type TreatmentResponseRunV1, type TreatmentResponseReviewV1, type TreatmentResponseScopeV1
} from "@phenometrix/contracts";

export interface TreatmentResponseEvidence {
  run: TreatmentResponseRunV1;
  status: "generated" | "reviewed" | "dismissed" | "correction-requested" | "superseded";
  reviewedAt: string | null;
  latestReview: TreatmentResponseReviewV1 | null;
  supersededByRunId: string | null;
  headline: string;
  encounterCount: number;
  hasObservations: boolean;
  limitations: readonly string[];
}

function freeze<T>(value: T): T {
  if (value && typeof value === "object") {
    Object.freeze(value);
    for (const child of Object.values(value)) if (child && typeof child === "object" && !Object.isFrozen(child)) freeze(child);
  }
  return value;
}

/** Source-preserving projection. Authentication belongs to the host/service boundary. */
export async function buildTreatmentResponseEvidence(input: {
  run: unknown; expectedScope: TreatmentResponseScopeV1;
  reviews?: readonly unknown[]; supersededByRunId?: string | null;
}): Promise<TreatmentResponseEvidence> {
  const run = TreatmentResponseRunV1Schema.parse(input.run);
  if (canonicalTreatmentResponseJson(run.scope) !== canonicalTreatmentResponseJson(input.expectedScope)) throw new Error("evidence-scope-mismatch");
  if (!(await verifyTreatmentResponseArtifact(run))) throw new Error("evidence-run-digest-mismatch");
  const reviews = (input.reviews ?? []).map(value => TreatmentResponseReviewV1Schema.parse(value));
  if (reviews.some(review => canonicalTreatmentResponseJson(review.scope) !== canonicalTreatmentResponseJson(run.scope))) throw new Error("evidence-review-scope-mismatch");
  if (reviews.some(review => review.runId === run.runId && review.runSha256 !== run.contentSha256)) throw new Error("evidence-review-digest-mismatch");
  if (reviews.some(review => review.runId === run.runId && Date.parse(review.recordedAt) < Date.parse(run.generatedAt))) throw new Error("evidence-review-chronology");
  if (new Set(reviews.map(review => review.reviewId)).size !== reviews.length) throw new Error("evidence-duplicate-review");
  const latestReview = reviews.filter(review => review.runId === run.runId)
    .sort((a, b) => Date.parse(a.recordedAt) - Date.parse(b.recordedAt) || a.reviewId.localeCompare(b.reviewId)).at(-1) ?? null;
  const supersededByRunId = input.supersededByRunId ?? null;
  if (supersededByRunId === run.runId) throw new Error("evidence-cannot-supersede-itself");
  const status = supersededByRunId ? "superseded" : latestReview?.disposition === "acknowledged" ? "reviewed" : latestReview?.disposition ?? "generated";
  const encounterCount = new Set(run.rows.flatMap(row => row.points.map(point => point.encounterId))).size;
  return freeze({
    run, status, latestReview, reviewedAt: latestReview?.recordedAt ?? null, supersededByRunId,
    headline: run.status === "anchor-unavailable" ? "Treatment timing unavailable" : run.rows.some(row => row.points.some(point => point.daysSinceTreatment !== null && point.daysSinceTreatment.minimum >= 0)) ? "Observations following treatment" : "Pre-treatment observations",
    encounterCount, hasObservations: encounterCount > 0,
    limitations: [
      "Descriptive research measurements; changes do not establish benefit, harm, or treatment causation.",
      "Measurement error and minimum detectable change are unknown.",
      "Onset, durability, recurrence, and forecasts are not estimated. Gaps remain unobserved.",
      "Voluntary capacity and unobserved adverse symptoms are not assessed.",
      ...(run.contextFlags.length ? ["Treatment-history or clinical-context limitations are recorded; inspect the source history."] : [])
    ]
  });
}
