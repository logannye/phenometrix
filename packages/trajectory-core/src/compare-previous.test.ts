import { describe, expect, it } from "vitest";
import {
  PREVIOUS_VISIT_CLAIM_BOUNDARY,
  PreviousVisitComparisonV1Schema
} from "@phenometrix/contracts";
import {
  comparePreviousVisit,
  createPreviousVisitComparisonId
} from "./compare-previous.js";
import {
  createAcceptedReferenceFixture,
  createComparisonInput,
  createContextFixture,
  createObservationFixture,
  metricOverride
} from "./test-fixtures.js";

describe("comparePreviousVisit", () => {
  it("returns deterministic raw current-minus-reference rows with exact sources", () => {
    const input = createComparisonInput();

    const first = comparePreviousVisit(input);
    const replay = comparePreviousVisit(structuredClone(input));

    expect(first).toEqual(replay);
    expect(PreviousVisitComparisonV1Schema.safeParse(first).success).toBe(true);
    expect(first.comparisonId).toBe(
      createPreviousVisitComparisonId(
        input.acceptedReference,
        input.currentObservation,
        input.currentContext
      )
    );
    expect(first.generatedAt).toBe(input.currentObservation.endedAt);
    expect(first.analyticalRepeatability).toBe("unknown");
    expect(first.minimumDetectableChange).toBe("unknown");
    expect(first.claimBoundary).toBe(PREVIOUS_VISIT_CLAIM_BOUNDARY);
    expect(first.rows).toHaveLength(6);
    for (const row of first.rows) {
      expect(row.status).toBe("measured");
      expect(row.decision).toBe("included");
      expect(row.compatibilityReasonCodes).toEqual([]);
      expect(row.delta).toBeCloseTo(0.005, 12);
      if (row.status !== "measured") throw new Error("Expected measured row.");
      expect(row.reference.outcome.evidence.refs.length).toBeGreaterThan(1);
      expect(row.current.outcome.evidence.refs.length).toBeGreaterThan(1);
      expect(row.delta).toBe(
        row.current.outcome.value - row.reference.outcome.value
      );
    }
  });

  it("preserves anatomical sign and does not transform by asserted side", () => {
    const metricCode =
      "ambient.face.rest_mouth_corner_asymmetry.signed" as const;
    const reference = createObservationFixture({
      sessionId: "session-signed-reference",
      observationId: "observation-signed-reference",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:01:00.000Z",
      metricOverrides: metricOverride(metricCode, { value: -0.02 })
    });
    const current = createObservationFixture({
      sessionId: "session-signed-current",
      observationId: "observation-signed-current",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      metricOverrides: metricOverride(metricCode, { value: 0.03 })
    });
    const rightContext = createContextFixture({ side: "right" });
    const input = createComparisonInput({
      referenceObservation: reference,
      currentObservation: current,
      referenceContext: rightContext,
      currentContext: rightContext
    });

    const comparison = comparePreviousVisit(input);
    const row = comparison.rows.find(
      (candidate) => candidate.metricCode === metricCode
    );

    expect(comparison.demoContext.assertedAffectedSide.side).toBe("right");
    expect(row?.status).toBe("measured");
    expect(row?.delta).toBe(0.05);
  });

  it("preserves prior and current withheld states without a delta", () => {
    const metricCode =
      "ambient.face.spontaneous_excursion_asymmetry.median" as const;
    const reference = createObservationFixture({
      sessionId: "session-withheld-reference",
      observationId: "observation-withheld-reference",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:01:00.000Z",
      metricOverrides: metricOverride(metricCode, {
        withheldReason: "insufficient-events"
      })
    });
    const current = createObservationFixture({
      sessionId: "session-withheld-current",
      observationId: "observation-withheld-current",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      metricOverrides: metricOverride(metricCode, {
        withheldReason: "quality-threshold-failed"
      })
    });
    const comparison = comparePreviousVisit(
      createComparisonInput({
        referenceObservation: reference,
        currentObservation: current
      })
    );
    const row = comparison.rows.find(
      (candidate) => candidate.metricCode === metricCode
    );

    expect(row).toMatchObject({
      status: "withheld",
      decision: "excluded",
      compatibilityReasonCodes: [
        "reference-withheld",
        "current-withheld"
      ],
      delta: null
    });
    if (!row || row.status !== "withheld") {
      throw new Error("Expected withheld row.");
    }
    expect(row.reference.status).toBe("withheld");
    expect(row.reference.withheldReasonCode).toBe("insufficient-events");
    expect(row.current.status).toBe("withheld");
    expect(row.current.withheldReasonCode).toBe(
      "quality-threshold-failed"
    );
    expect(row.reference).not.toHaveProperty("value");
    expect(row.reference).not.toHaveProperty("outcome");
    expect(row.current).not.toHaveProperty("value");
    expect(row.current).not.toHaveProperty("outcome");
  });

  it("returns metric-missing for a missing current terminal outcome", () => {
    const metricCode =
      "ambient.face.lid_closure_completeness.right" as const;
    const current = createObservationFixture({
      sessionId: "session-missing-current",
      observationId: "observation-missing-current",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      omittedMetric: metricCode
    });

    const comparison = comparePreviousVisit(
      createComparisonInput({ currentObservation: current })
    );
    const row = comparison.rows.find(
      (candidate) => candidate.metricCode === metricCode
    );

    expect(row).toMatchObject({
      metricCode,
      status: "incompatible",
      decision: "excluded",
      compatibilityReasonCodes: ["metric-missing"],
      nativeUnit: null,
      current: null,
      delta: null
    });
    if (!row || row.status !== "incompatible" || row.reference === null) {
      throw new Error("Expected an available reference trace.");
    }
    expect(row.reference.metricCode).toBe(metricCode);
    expect(row.reference).not.toHaveProperty("value");
    expect(row.reference).not.toHaveProperty("outcome");
  });

  it("rejects an accepted reference that lacks an allowlisted outcome", () => {
    const reference = createObservationFixture({
      sessionId: "session-invalid-reference",
      observationId: "observation-invalid-reference",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:01:00.000Z",
      omittedMetric: "ambient.face.lid_closure_completeness.left"
    });

    expect(() =>
      createAcceptedReferenceFixture(reference, createContextFixture())
    ).toThrow();
  });

  it("rejects structurally non-finite input at the ObservationV3 boundary", () => {
    const input = createComparisonInput();
    const outcome = input.currentObservation.metricOutcomes.find(
      (candidate) => candidate.status === "measured"
    );
    if (!outcome || outcome.status !== "measured") {
      throw new Error("Fixture needs a measured outcome.");
    }
    outcome.value = Number.NaN;

    expect(() => comparePreviousVisit(input)).toThrow();
  });

  it("excludes pairwise-equal fabricated metadata that claims the canonical pack", () => {
    const metricCode =
      "ambient.face.rest_mouth_corner_asymmetry.signed" as const;
    const reference = createObservationFixture({
      sessionId: "session-fabricated-reference",
      observationId: "observation-fabricated-reference",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:01:00.000Z",
      metricOverrides: metricOverride(metricCode, { unit: "diagnosis-score" })
    });
    const current = createObservationFixture({
      sessionId: "session-fabricated-current",
      observationId: "observation-fabricated-current",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      metricOverrides: metricOverride(metricCode, { unit: "diagnosis-score" })
    });

    const row = comparePreviousVisit(
      createComparisonInput({
        referenceObservation: reference,
        currentObservation: current
      })
    ).rows.find((candidate) => candidate.metricCode === metricCode);

    expect(row).toMatchObject({
      status: "incompatible",
      compatibilityReasonCodes: ["metric-definition-mismatch"],
      delta: null
    });
  });

  it("rejects condition evidence bound to another observation", () => {
    const input = createComparisonInput();
    const outcome = input.currentObservation.metricOutcomes.find(
      (candidate) =>
        candidate.metricCode ===
        "ambient.face.rest_mouth_corner_asymmetry.signed"
    );
    if (!outcome) throw new Error("Fixture needs the condition outcome.");
    outcome.evidence.refs[0].observationId = "another-observation";

    expect(() => comparePreviousVisit(input)).toThrow(
      /bound to another observation/
    );
  });

  it("reason-codes subtraction overflow instead of emitting infinity", () => {
    const metricCode =
      "ambient.face.rest_eye_aperture_asymmetry.signed" as const;
    const reference = createObservationFixture({
      sessionId: "session-overflow-reference",
      observationId: "observation-overflow-reference",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:01:00.000Z",
      metricOverrides: metricOverride(metricCode, {
        value: -Number.MAX_VALUE
      })
    });
    const current = createObservationFixture({
      sessionId: "session-overflow-current",
      observationId: "observation-overflow-current",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      metricOverrides: metricOverride(metricCode, {
        value: Number.MAX_VALUE
      })
    });

    const comparison = comparePreviousVisit(
      createComparisonInput({
        referenceObservation: reference,
        currentObservation: current
      })
    );
    const row = comparison.rows.find(
      (candidate) => candidate.metricCode === metricCode
    );

    expect(row).toMatchObject({
      status: "incompatible",
      compatibilityReasonCodes: ["non-finite-delta"],
      delta: null
    });
    if (!row || row.status !== "incompatible") {
      throw new Error("Expected incompatible overflow row.");
    }
    expect(row.reference).not.toHaveProperty("value");
    expect(row.reference).not.toHaveProperty("outcome");
    expect(row.current).not.toHaveProperty("value");
    expect(row.current).not.toHaveProperty("outcome");
  });
});
