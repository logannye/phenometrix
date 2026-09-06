import { describe, it, expect } from "vitest";
import { buildTreatmentResponseEvidence } from "@phenometrix/evidence-core";
import { syntheticRun, SYNTHETIC_SCOPE } from "./synthetic.js";

describe("chart evidence boundary", () => {
  it("preserves source values and leaves clinical claims disabled", async () => {
    const run = await syntheticRun();
    const evidence = await buildTreatmentResponseEvidence({ run, expectedScope: SYNTHETIC_SCOPE });
    expect(evidence.run).toEqual(run); expect(evidence.status).toBe("generated");
    expect(evidence.encounterCount).toBe(5); expect(evidence.run.fittedModels).toBe(false);
    expect(Object.isFrozen(evidence.run.rows[0].points[0])).toBe(true);
  });
  it("rejects changed measurements and cross-patient results", async () => {
    const run = await syntheticRun(); run.rows[0].label = "tampered";
    await expect(buildTreatmentResponseEvidence({ run, expectedScope: SYNTHETIC_SCOPE })).rejects.toThrow("digest");
    await expect(buildTreatmentResponseEvidence({ run: await syntheticRun(), expectedScope: { ...SYNTHETIC_SCOPE, participantId: "other" } })).rejects.toThrow("scope");
  });
  it("does not transfer an old review onto new evidence", async () => {
    const run = await syntheticRun();
    const review = { schemaVersion: "phenometric.treatment-response-review.v1", reviewId: "review-1", scope: SYNTHETIC_SCOPE, runId: "old-run", runSha256: "d".repeat(64), actorId: "research-clinician", recordedAt: "2026-08-24T13:00:00.000Z", disposition: "acknowledged", supersedesReviewId: null, reasonCode: null };
    const evidence = await buildTreatmentResponseEvidence({ run, expectedScope: SYNTHETIC_SCOPE, reviews: [review] });
    expect(evidence.status).toBe("generated"); expect(evidence.latestReview).toBe(null);
    const current = { ...review, runId: run.runId, runSha256: run.contentSha256 };
    expect((await buildTreatmentResponseEvidence({ run, expectedScope: SYNTHETIC_SCOPE, reviews: [current] })).status).toBe("reviewed");
    expect((await buildTreatmentResponseEvidence({ run, expectedScope: SYNTHETIC_SCOPE, reviews: [current], supersededByRunId: "new-run" })).status).toBe("superseded");
    await expect(buildTreatmentResponseEvidence({ run, expectedScope: SYNTHETIC_SCOPE, reviews: [{ ...current, recordedAt: "2025-01-01T00:00:00.000Z" }] })).rejects.toThrow("chronology");
  });
});
