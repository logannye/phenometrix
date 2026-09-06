import {
  CONDITION_DEMO_BOUNDARY_STATEMENT,
  CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
  ConditionEvidenceCardReviewV1Schema,
  ConditionEvidenceCardV1Schema,
  PreviousVisitComparisonV1Schema,
  type ConditionDemoProfileRefV1,
  type ConditionEvidenceCardReviewV1,
  type ConditionEvidenceCardV1,
  type PreviousVisitComparisonRowV1,
  type PreviousVisitComparisonV1
} from "@phenometrix/contracts";
import {
  UNILATERAL_FACIAL_MOVEMENT_PROFILE,
  UNILATERAL_FACIAL_MOVEMENT_PROFILE_REF
} from "@phenometrix/condition-profiles";

export interface BuildConditionEvidenceCardInput {
  comparison: PreviousVisitComparisonV1;
  generatedAt: string;
  cardId?: string;
  review?: ConditionEvidenceCardReviewV1;
}

function sameProfileRef(
  profile: ConditionDemoProfileRefV1,
  comparison: ConditionDemoProfileRefV1
): boolean {
  return (
    profile.profileId === comparison.profileId &&
    profile.version === comparison.version &&
    profile.contentSha256 === comparison.contentSha256
  );
}

function minimumMeasuredQuality(
  rows: readonly PreviousVisitComparisonRowV1[],
  sourceName: "reference" | "current"
): number | null {
  const scores = rows.flatMap((row) => {
    const source = row[sourceName];
    if (source?.status !== "measured") return [];
    return [
      "outcome" in source
        ? source.outcome.technicalQualityScore
        : source.technicalQualityScore
    ];
  });
  return scores.length === 0 ? null : Math.min(...scores);
}

/**
 * Projects a validated comparison into the fixed, inspectable condition card.
 *
 * This function does not draft prose or reinterpret a metric. Each row embeds
 * the exact comparison row (and therefore its exact source outcomes), while
 * display labels come only from the strict condition-profile schema. In
 * particular, the internal synkinesis-oriented metric code is presented with
 * the approved nonclinical "Oculo-oral coupling difference" label.
 */
export function buildConditionEvidenceCard(
  input: BuildConditionEvidenceCardInput
): ConditionEvidenceCardV1 {
  const profile = UNILATERAL_FACIAL_MOVEMENT_PROFILE;
  const comparison = PreviousVisitComparisonV1Schema.parse(input.comparison);
  const review = ConditionEvidenceCardReviewV1Schema.parse(
    input.review ?? {
      status: "pending",
      action: null,
      actor: null,
      recordedAt: null
    }
  );

  if (
    !sameProfileRef(
      UNILATERAL_FACIAL_MOVEMENT_PROFILE_REF,
      comparison.demoContext.profileRef
    )
  ) {
    throw new Error(
      "Condition evidence card profile does not match the comparison context."
    );
  }
  if (
    profile.sourceProtocolRef.packId !==
      comparison.demoContext.sourceProtocolRef.packId ||
    profile.sourceProtocolRef.version !==
      comparison.demoContext.sourceProtocolRef.version ||
    profile.sourceProtocolRef.contentSha256 !==
      comparison.demoContext.sourceProtocolRef.contentSha256
  ) {
    throw new Error(
      "Condition evidence card source protocol does not match the comparison context."
    );
  }
  if (Date.parse(input.generatedAt) < Date.parse(comparison.generatedAt)) {
    throw new Error(
      "Condition evidence card cannot be generated before its comparison."
    );
  }

  const rows = profile.metrics.map((metric, index) => ({
    metric,
    // Preserve the parsed row object verbatim. The card schema verifies this
    // against the embedded comparison again, so a display value cannot drift
    // away from the outcome that supplied it.
    comparison: comparison.rows[index]
  }));

  return ConditionEvidenceCardV1Schema.parse({
    schemaVersion: "phenometric.condition-evidence-card.v1",
    cardId: input.cardId ?? `card-${comparison.comparisonId}`,
    generatedAt: input.generatedAt,
    comparison,
    rows,
    qualitySummary: {
      measuredComparisonCount: comparison.rows.filter(
        (row) => row.status === "measured"
      ).length,
      withheldComparisonCount: comparison.rows.filter(
        (row) => row.status === "withheld"
      ).length,
      incompatibleComparisonCount: comparison.rows.filter(
        (row) => row.status === "incompatible"
      ).length,
      referenceMinimumTechnicalQualityScore: minimumMeasuredQuality(
        comparison.rows,
        "reference"
      ),
      currentMinimumTechnicalQualityScore: minimumMeasuredQuality(
        comparison.rows,
        "current"
      )
    },
    review,
    boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
    sourceDisclosure: CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
    persistence: "page-memory-only",
    exportAvailable: false
  });
}
