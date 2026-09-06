import { z } from "zod";
import {
  CONDITION_DEMO_BOUNDARY_STATEMENT,
  CONDITION_DEMO_METRIC_CODES,
  CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
  ConditionDemoMetricDefinitionV1Schema
} from "./condition-demo.js";
import {
  PreviousVisitComparisonRowV1Schema,
  PreviousVisitComparisonV1Schema
} from "./previous-visit-comparison.js";

const IdSchema = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const IsoTimestampSchema = z.string().datetime({ offset: true });

export const ConditionEvidenceCardReviewV1Schema = z.discriminatedUnion(
  "status",
  [
    z
      .object({
        status: z.literal("pending"),
        action: z.null(),
        actor: z.null(),
        recordedAt: z.null()
      })
      .strict(),
    z
      .object({
        status: z.literal("accepted"),
        action: z.literal("accept-card"),
        actor: z.literal("local-demo-user"),
        recordedAt: IsoTimestampSchema
      })
      .strict(),
    z
      .object({
        status: z.literal("dismissed"),
        action: z.literal("dismiss-card"),
        actor: z.literal("local-demo-user"),
        recordedAt: IsoTimestampSchema
      })
      .strict()
  ]
);
export type ConditionEvidenceCardReviewV1 = z.infer<
  typeof ConditionEvidenceCardReviewV1Schema
>;

export const ConditionEvidenceCardRowV1Schema = z
  .object({
    metric: ConditionDemoMetricDefinitionV1Schema,
    comparison: PreviousVisitComparisonRowV1Schema
  })
  .strict()
  .superRefine((row, context) => {
    if (row.metric.metricCode !== row.comparison.metricCode) {
      context.addIssue({
        code: "custom",
        path: ["comparison", "metricCode"],
        message: "Evidence-card metric must match its comparison row."
      });
    }
  });
export type ConditionEvidenceCardRowV1 = z.infer<
  typeof ConditionEvidenceCardRowV1Schema
>;

const ConditionEvidenceCardQualitySummaryV1Schema = z
  .object({
    measuredComparisonCount: z.number().int().nonnegative(),
    withheldComparisonCount: z.number().int().nonnegative(),
    incompatibleComparisonCount: z.number().int().nonnegative(),
    referenceMinimumTechnicalQualityScore: z
      .number()
      .finite()
      .min(0)
      .max(1)
      .nullable(),
    currentMinimumTechnicalQualityScore: z
      .number()
      .finite()
      .min(0)
      .max(1)
      .nullable()
  })
  .strict();
export type ConditionEvidenceCardQualitySummaryV1 = z.infer<
  typeof ConditionEvidenceCardQualitySummaryV1Schema
>;

function minimumMeasuredQuality(
  rows: readonly z.infer<typeof PreviousVisitComparisonRowV1Schema>[],
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
  return scores.length > 0 ? Math.min(...scores) : null;
}

export const ConditionEvidenceCardV1Schema = z
  .object({
    schemaVersion: z.literal("phenometric.condition-evidence-card.v1"),
    cardId: IdSchema,
    generatedAt: IsoTimestampSchema,
    comparison: PreviousVisitComparisonV1Schema,
    rows: z.array(ConditionEvidenceCardRowV1Schema).length(6),
    qualitySummary: ConditionEvidenceCardQualitySummaryV1Schema,
    review: ConditionEvidenceCardReviewV1Schema,
    boundaryStatement: z.literal(CONDITION_DEMO_BOUNDARY_STATEMENT),
    sourceDisclosure: z.literal(CONDITION_EVIDENCE_SOURCE_DISCLOSURE),
    persistence: z.literal("page-memory-only"),
    exportAvailable: z.literal(false)
  })
  .strict()
  .superRefine((card, context) => {
    if (
      Date.parse(card.generatedAt) <
      Date.parse(card.comparison.generatedAt)
    ) {
      context.addIssue({
        code: "custom",
        path: ["generatedAt"],
        message: "Evidence-card generation cannot precede its comparison."
      });
    }
    card.rows.forEach((row, index) => {
      if (
        row.metric.metricCode !== CONDITION_DEMO_METRIC_CODES[index] ||
        row.comparison.metricCode !== CONDITION_DEMO_METRIC_CODES[index]
      ) {
        context.addIssue({
          code: "custom",
          path: ["rows", index],
          message: "Evidence-card rows must follow the profile metric order."
        });
      }
      const comparisonRow = card.comparison.rows[index];
      if (
        JSON.stringify(row.comparison) !== JSON.stringify(comparisonRow)
      ) {
        context.addIssue({
          code: "custom",
          path: ["rows", index, "comparison"],
          message: "Evidence-card row must preserve its comparison artifact."
        });
      }
    });

    const expectedCounts = {
      measuredComparisonCount: card.comparison.rows.filter(
        (row) => row.status === "measured"
      ).length,
      withheldComparisonCount: card.comparison.rows.filter(
        (row) => row.status === "withheld"
      ).length,
      incompatibleComparisonCount: card.comparison.rows.filter(
        (row) => row.status === "incompatible"
      ).length,
      referenceMinimumTechnicalQualityScore: minimumMeasuredQuality(
        card.comparison.rows,
        "reference"
      ),
      currentMinimumTechnicalQualityScore: minimumMeasuredQuality(
        card.comparison.rows,
        "current"
      )
    };
    for (const [key, expected] of Object.entries(expectedCounts)) {
      if (
        card.qualitySummary[
          key as keyof typeof card.qualitySummary
        ] !== expected
      ) {
        context.addIssue({
          code: "custom",
          path: ["qualitySummary", key],
          message: "Evidence-card quality summary must be derived from its rows."
        });
      }
    }
    if (
      card.review.status !== "pending" &&
      Date.parse(card.review.recordedAt) < Date.parse(card.generatedAt)
    ) {
      context.addIssue({
        code: "custom",
        path: ["review", "recordedAt"],
        message: "Card review cannot precede card generation."
      });
    }
  });
export type ConditionEvidenceCardV1 = z.infer<
  typeof ConditionEvidenceCardV1Schema
>;
