import { describe, expect, it } from "vitest";
import {
  CONDITION_DEMO_METRIC_CODES,
  ConditionDemoContextV1Schema,
  ConditionDemoProfileV1Schema,
  verifyConditionDemoProfileDigest
} from "@phenometrix/contracts";
import {
  UNILATERAL_FACIAL_MOVEMENT_PROFILE,
  UNILATERAL_FACIAL_MOVEMENT_PROFILE_REF,
  createUnilateralFacialMovementDemoContext
} from "./index.js";

describe("unilateral facial movement research profile", () => {
  it("is strict, immutable, complete, and bound to its canonical digest", async () => {
    expect(
      ConditionDemoProfileV1Schema.parse(
        UNILATERAL_FACIAL_MOVEMENT_PROFILE
      )
    ).toEqual(UNILATERAL_FACIAL_MOVEMENT_PROFILE);
    expect(
      await verifyConditionDemoProfileDigest(
        UNILATERAL_FACIAL_MOVEMENT_PROFILE
      )
    ).toBe(true);
    expect(Object.isFrozen(UNILATERAL_FACIAL_MOVEMENT_PROFILE)).toBe(true);
    expect(Object.isFrozen(UNILATERAL_FACIAL_MOVEMENT_PROFILE.metrics)).toBe(
      true
    );
    expect(
      UNILATERAL_FACIAL_MOVEMENT_PROFILE.metrics.map(
        (metric) => metric.metricCode
      )
    ).toEqual(CONDITION_DEMO_METRIC_CODES);
    expect(UNILATERAL_FACIAL_MOVEMENT_PROFILE_REF).toEqual({
      profileId: UNILATERAL_FACIAL_MOVEMENT_PROFILE.profileId,
      version: UNILATERAL_FACIAL_MOVEMENT_PROFILE.version,
      contentSha256: UNILATERAL_FACIAL_MOVEMENT_PROFILE.contentSha256
    });
  });

  it("creates a page-memory-only context with participant-asserted laterality", () => {
    const context = createUnilateralFacialMovementDemoContext({
      subjectRef: "demo-subject-1",
      assertedAffectedSide: "right"
    });

    expect(ConditionDemoContextV1Schema.parse(context)).toEqual(context);
    expect(context.assertedAffectedSide).toEqual({
      side: "right",
      source: "participant-asserted",
      verified: false
    });
    expect(context.persistence).toBe("page-memory-only");
    expect(Object.isFrozen(context)).toBe(true);
  });

  it("does not permit unsafe metric relabeling", () => {
    const invalid = structuredClone(
      UNILATERAL_FACIAL_MOVEMENT_PROFILE
    ) as unknown as {
      metrics: Array<{ displayLabel: string }>;
    };
    invalid.metrics[5].displayLabel = "Synkinesis detected";
    expect(ConditionDemoProfileV1Schema.safeParse(invalid).success).toBe(false);
  });
});
