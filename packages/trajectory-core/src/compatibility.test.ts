import { describe, expect, it } from "vitest";
import {
  AMBIENT_LOCAL_PROTOCOL_REF,
  type ProtocolRef
} from "@phenometrix/contracts";
import {
  comparePreviousVisit,
  type ComparePreviousVisitInput
} from "./compare-previous.js";
import {
  createComparisonInput,
  createContextFixture,
  createObservationFixture,
  metricOverride
} from "./test-fixtures.js";

function reasonsForEveryRow(input: ComparePreviousVisitInput): string[][] {
  return comparePreviousVisit(input).rows.map(
    (row) => row.compatibilityReasonCodes
  );
}

describe("previous-visit global compatibility", () => {
  it("excludes every metric for a different subject", () => {
    const current = createObservationFixture({
      sessionId: "session-other-subject",
      observationId: "observation-other-subject",
      subjectRef: "other-subject",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z"
    });
    const input = createComparisonInput({
      currentObservation: current,
      currentContext: createContextFixture({ subjectRef: "other-subject" })
    });

    expect(reasonsForEveryRow(input)).toEqual(
      Array.from({ length: 6 }, () => ["subject-mismatch"])
    );
  });

  it("keeps profile version, digest, and participant-asserted side exact", () => {
    const input = createComparisonInput({
      currentContext: createContextFixture({
        profileVersion: "1.1.0",
        profileDigest: "c".repeat(64),
        side: "right"
      })
    });

    expect(reasonsForEveryRow(input)).toEqual(
      Array.from({ length: 6 }, () => [
        "profile-version-mismatch",
        "profile-digest-mismatch",
        "asserted-side-mismatch"
      ])
    );
  });

  it("keeps protocol version and content digest exact", () => {
    const currentProtocol: ProtocolRef = {
      packId: "ambient-local-observation",
      version: "9.9.9",
      contentSha256: "d".repeat(64)
    };
    const current = createObservationFixture({
      sessionId: "session-new-protocol",
      observationId: "observation-new-protocol",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      protocolRef: currentProtocol
    });
    const input = createComparisonInput({
      currentObservation: current,
      currentContext: createContextFixture({ protocolRef: currentProtocol })
    });

    expect(reasonsForEveryRow(input)).toEqual(
      Array.from({ length: 6 }, () => [
        "protocol-version-mismatch",
        "protocol-digest-mismatch"
      ])
    );
  });

  it("keeps capture-adapter id and version exact", () => {
    const current = createObservationFixture({
      sessionId: "session-new-adapter",
      observationId: "observation-new-adapter",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      captureAdapter: { id: "different-adapter", version: "2.0.0" }
    });

    expect(
      reasonsForEveryRow(
        createComparisonInput({ currentObservation: current })
      )
    ).toEqual(
      Array.from({ length: 6 }, () => [
        "capture-adapter-id-mismatch",
        "capture-adapter-version-mismatch"
      ])
    );
  });

  it("requires the accepted reference and its capture to precede current", () => {
    const current = createObservationFixture({
      sessionId: "session-overlap",
      observationId: "observation-overlap",
      startedAt: "2026-08-21T10:01:00.000Z",
      endedAt: "2026-08-21T10:02:00.000Z"
    });

    expect(
      reasonsForEveryRow(
        createComparisonInput({ currentObservation: current })
      )
    ).toEqual(
      Array.from({ length: 6 }, () => ["reference-not-before-current"])
    );
  });

  it("requires the explicit reference-acceptance action to precede current", () => {
    const input = createComparisonInput();
    input.acceptedReference.acceptedAt = "2026-08-21T10:02:30.000Z";

    expect(reasonsForEveryRow(input)).toEqual(
      Array.from({ length: 6 }, () => ["reference-not-before-current"])
    );
  });

  it("does not treat simultaneous reference acceptance and current start as ordered", () => {
    const input = createComparisonInput();
    input.acceptedReference.acceptedAt = input.currentObservation.startedAt;

    expect(reasonsForEveryRow(input)).toEqual(
      Array.from({ length: 6 }, () => ["reference-not-before-current"])
    );
  });

  it("reason-codes profile and protocol pack IDs at the validated boundary", () => {
    const input = createComparisonInput({
      currentContext: createContextFixture({
        profileId: "future-condition-profile",
        protocolRef: {
          ...AMBIENT_LOCAL_PROTOCOL_REF,
          packId: "future-protocol-pack"
        }
      })
    });

    expect(reasonsForEveryRow(input)).toEqual(
      Array.from({ length: 6 }, () => [
        "profile-id-mismatch",
        "protocol-pack-id-mismatch"
      ])
    );
  });
});

describe("per-metric compatibility", () => {
  const metricCode =
    "ambient.face.rest_mouth_corner_asymmetry.signed" as const;

  it.each([
    [
      "context",
      { context: "ambient-speech-turn" as const },
      ["metric-definition-mismatch", "context-mismatch"]
    ],
    [
      "modality",
      { modality: "voice" as const },
      ["metric-definition-mismatch", "modality-mismatch"]
    ],
    [
      "unit",
      { unit: "different-native-unit" },
      ["metric-definition-mismatch", "unit-mismatch"]
    ],
    [
      "algorithm",
      { algorithmVersion: "2.0.0" },
      ["metric-definition-mismatch", "algorithm-version-mismatch"]
    ]
  ])("excludes a %s mismatch", (_label, override, expectedReasons) => {
    const current = createObservationFixture({
      sessionId: `session-${_label}`,
      observationId: `observation-${_label}`,
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      metricOverrides: metricOverride(metricCode, override),
      ...("modality" in override && override.modality === "voice"
        ? { processor: { modality: "voice" as const } }
        : {})
    });

    const comparison = comparePreviousVisit(
      createComparisonInput({ currentObservation: current })
    );
    const row = comparison.rows.find(
      (candidate) => candidate.metricCode === metricCode
    );

    expect(row?.status).toBe("incompatible");
    expect(row?.compatibilityReasonCodes).toEqual(expectedReasons);
    expect(row?.delta).toBeNull();
  });

  it.each([
    [
      "processor ref",
      { processorRef: "face-geometry@2.0.0" },
      ["processor-ref-mismatch"]
    ],
    [
      "processor modality",
      { modality: "voice" as const },
      ["modality-mismatch"]
    ],
    [
      "runtime",
      { runtime: "different-runtime" },
      ["processor-runtime-mismatch"]
    ],
    [
      "runtime version",
      { runtimeVersion: "0.11.0" },
      ["processor-runtime-version-mismatch"]
    ],
    [
      "asset path",
      { assetPath: "assets/different.task" },
      ["processor-asset-path-mismatch"]
    ],
    [
      "asset digest",
      { assetSha256: "e".repeat(64) },
      ["processor-asset-digest-mismatch"]
    ],
    [
      "asset verification",
      { assetIntegrityVerified: false },
      ["processor-asset-integrity-mismatch"]
    ]
  ])("excludes a %s mismatch", (_label, processor, expectedReasons) => {
    const current = createObservationFixture({
      sessionId: `session-processor-${_label.replaceAll(" ", "-")}`,
      observationId: `observation-processor-${_label.replaceAll(" ", "-")}`,
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      processor
    });

    expect(
      reasonsForEveryRow(
        createComparisonInput({ currentObservation: current })
      )
    ).toEqual(Array.from({ length: 6 }, () => expectedReasons));
  });

  it("excludes missing processor provenance and nulls the unavailable source", () => {
    const current = createObservationFixture({
      sessionId: "session-processor-missing",
      observationId: "observation-processor-missing",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      processors: []
    });

    const comparison = comparePreviousVisit(
      createComparisonInput({ currentObservation: current })
    );
    for (const row of comparison.rows) {
      expect(row).toMatchObject({
        status: "incompatible",
        compatibilityReasonCodes: ["processor-provenance-missing"],
        current: null,
        delta: null
      });
    }
  });

  it("excludes ambiguous processor provenance", () => {
    const current = createObservationFixture({
      sessionId: "session-processor-ambiguous",
      observationId: "observation-processor-ambiguous",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z"
    });
    current.processors.push(structuredClone(current.processors[0]));

    expect(
      reasonsForEveryRow(
        createComparisonInput({ currentObservation: current })
      )
    ).toEqual(
      Array.from({ length: 6 }, () => [
        "processor-provenance-ambiguous"
      ])
    );
  });

  it("does not let a withheld outcome mask incompatible processor provenance", () => {
    const withheldMetric =
      "ambient.face.spontaneous_excursion_asymmetry.median" as const;
    const current = createObservationFixture({
      sessionId: "session-withheld-processor-mismatch",
      observationId: "observation-withheld-processor-mismatch",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      metricOverrides: metricOverride(withheldMetric, {
        withheldReason: "insufficient-events"
      }),
      processor: { runtimeVersion: "0.11.0" }
    });

    const row = comparePreviousVisit(
      createComparisonInput({ currentObservation: current })
    ).rows.find((candidate) => candidate.metricCode === withheldMetric);

    expect(row).toMatchObject({
      status: "incompatible",
      compatibilityReasonCodes: ["processor-runtime-version-mismatch"],
      delta: null
    });
  });
});
