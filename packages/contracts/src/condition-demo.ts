import { z } from "zod";
import { calculateSha256Hex } from "./ambient-protocol.js";
import { ObservationV3Schema } from "./observation-v3.js";
import { ProtocolRefSchema } from "./protocol.js";

const IdSchema = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const IsoTimestampSchema = z.string().datetime({ offset: true });
const SemanticVersionSchema = z
  .string()
  .regex(/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/);
const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

export const UNILATERAL_FACIAL_MOVEMENT_PROFILE_ID =
  "unilateral-facial-movement-research-demo" as const;

export const CONDITION_DEMO_BOUNDARY_STATEMENT =
  "Research-only numerical comparison of two local sessions. Not for sudden facial weakness, stroke screening, emergencies, diagnosis, severity grading, treatment, or health-state interpretation." as const;

export const PREVIOUS_VISIT_CLAIM_BOUNDARY =
  "Raw current-minus-reference differences are engineering quantities only and must not be interpreted as improvement, worsening, recovery, progression, or clinically meaningful health change." as const;

export const CONDITION_EVIDENCE_SOURCE_DISCLOSURE =
  "The affected side is participant-asserted and unverified. Facial source attribution requires exactly one visible face and does not verify identity." as const;

export const CONDITION_DEMO_PROHIBITED_CLAIMS = [
  "diagnosis",
  "severity",
  "prognosis",
  "cause",
  "treatment",
  "emergency-guidance",
  "health-state-change",
  "clinical-meaningfulness",
  "validated-scale-equivalence",
  "automatic-affected-side-inference"
] as const;

export const ConditionDemoProhibitedClaimSchema = z.enum(
  CONDITION_DEMO_PROHIBITED_CLAIMS
);
export type ConditionDemoProhibitedClaim = z.infer<
  typeof ConditionDemoProhibitedClaimSchema
>;

export const CONDITION_DEMO_METRIC_CODES = [
  "ambient.face.rest_mouth_corner_asymmetry.signed",
  "ambient.face.rest_eye_aperture_asymmetry.signed",
  "ambient.face.lid_closure_completeness.left",
  "ambient.face.lid_closure_completeness.right",
  "ambient.face.spontaneous_excursion_asymmetry.median",
  "ambient.face.oculo_oral_synkinesis_index"
] as const;

export const ConditionDemoMetricCodeSchema = z.enum(
  CONDITION_DEMO_METRIC_CODES
);
export type ConditionDemoMetricCode = z.infer<
  typeof ConditionDemoMetricCodeSchema
>;

const RestMouthMetricV1Schema = z
  .object({
    metricCode: z.literal(
      "ambient.face.rest_mouth_corner_asymmetry.signed"
    ),
    displayLabel: z.literal(
      "Resting mouth difference (subject-left minus subject-right)"
    ),
    displayRole: z.literal("primary"),
    laterality: z.literal("subject-left-minus-subject-right")
  })
  .strict();

const RestEyeMetricV1Schema = z
  .object({
    metricCode: z.literal(
      "ambient.face.rest_eye_aperture_asymmetry.signed"
    ),
    displayLabel: z.literal(
      "Resting eye-aperture difference (subject-left minus subject-right)"
    ),
    displayRole: z.literal("primary"),
    laterality: z.literal("subject-left-minus-subject-right")
  })
  .strict();

const LeftClosureMetricV1Schema = z
  .object({
    metricCode: z.literal(
      "ambient.face.lid_closure_completeness.left"
    ),
    displayLabel: z.literal("Subject-left lid closure completeness"),
    displayRole: z.literal("primary"),
    laterality: z.literal("explicit-subject-left")
  })
  .strict();

const RightClosureMetricV1Schema = z
  .object({
    metricCode: z.literal(
      "ambient.face.lid_closure_completeness.right"
    ),
    displayLabel: z.literal("Subject-right lid closure completeness"),
    displayRole: z.literal("primary"),
    laterality: z.literal("explicit-subject-right")
  })
  .strict();

const ExcursionMetricV1Schema = z
  .object({
    metricCode: z.literal(
      "ambient.face.spontaneous_excursion_asymmetry.median"
    ),
    displayLabel: z.literal(
      "Spontaneous excursion difference (subject-left minus subject-right)"
    ),
    displayRole: z.literal("experimental"),
    laterality: z.literal("subject-left-minus-subject-right")
  })
  .strict();

const CouplingMetricV1Schema = z
  .object({
    metricCode: z.literal("ambient.face.oculo_oral_synkinesis_index"),
    displayLabel: z.literal("Oculo-oral coupling difference"),
    displayRole: z.literal("experimental"),
    laterality: z.literal("subject-left-minus-subject-right")
  })
  .strict();

export const ConditionDemoMetricDefinitionV1Schema = z.discriminatedUnion(
  "metricCode",
  [
    RestMouthMetricV1Schema,
    RestEyeMetricV1Schema,
    LeftClosureMetricV1Schema,
    RightClosureMetricV1Schema,
    ExcursionMetricV1Schema,
    CouplingMetricV1Schema
  ]
);
export type ConditionDemoMetricDefinitionV1 = z.infer<
  typeof ConditionDemoMetricDefinitionV1Schema
>;

const ConditionDemoMetricsTupleSchema = z.tuple([
  RestMouthMetricV1Schema,
  RestEyeMetricV1Schema,
  LeftClosureMetricV1Schema,
  RightClosureMetricV1Schema,
  ExcursionMetricV1Schema,
  CouplingMetricV1Schema
]);

const ConditionDemoMetricCodesTupleSchema = z.tuple([
  z.literal(CONDITION_DEMO_METRIC_CODES[0]),
  z.literal(CONDITION_DEMO_METRIC_CODES[1]),
  z.literal(CONDITION_DEMO_METRIC_CODES[2]),
  z.literal(CONDITION_DEMO_METRIC_CODES[3]),
  z.literal(CONDITION_DEMO_METRIC_CODES[4]),
  z.literal(CONDITION_DEMO_METRIC_CODES[5])
]);

const ConditionDemoProhibitedClaimsTupleSchema = z.tuple([
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[0]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[1]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[2]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[3]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[4]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[5]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[6]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[7]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[8]),
  z.literal(CONDITION_DEMO_PROHIBITED_CLAIMS[9])
]);

export const ConditionDemoProfileRefV1Schema = z
  .object({
    profileId: IdSchema,
    version: SemanticVersionSchema,
    contentSha256: Sha256Schema
  })
  .strict();
export type ConditionDemoProfileRefV1 = z.infer<
  typeof ConditionDemoProfileRefV1Schema
>;

export const ConditionDemoSourceProtocolRefV1Schema = z
  .object({
    packId: IdSchema,
    version: SemanticVersionSchema,
    contentSha256: Sha256Schema
  })
  .strict();
export type ConditionDemoSourceProtocolRefV1 = z.infer<
  typeof ConditionDemoSourceProtocolRefV1Schema
>;

export const ConditionDemoProfileV1Schema = z
  .object({
    schemaVersion: z.literal("phenometric.condition-demo-profile.v1"),
    profileId: z.literal(UNILATERAL_FACIAL_MOVEMENT_PROFILE_ID),
    version: SemanticVersionSchema,
    contentSha256: Sha256Schema,
    status: z.literal("nonclinical-research-demo"),
    displayName: z.literal("Unilateral Facial Movement Research Demo"),
    sourceProtocolRef: ProtocolRefSchema,
    intendedUse: z.literal(
      "Within-page repeatability demonstration for a participant with previously established unilateral peripheral facial palsy."
    ),
    targetPopulation: z.literal(
      "Adults participating in a nonclinical research demonstration who assert a previously established unilateral peripheral facial palsy."
    ),
    affectedSidePolicy: z
      .object({
        source: z.literal("participant-asserted"),
        verified: z.literal(false),
        automaticInference: z.literal(false)
      })
      .strict(),
    metrics: ConditionDemoMetricsTupleSchema,
    boundaryStatement: z.literal(CONDITION_DEMO_BOUNDARY_STATEMENT),
    prohibitedClaims: ConditionDemoProhibitedClaimsTupleSchema,
    clinicalValidation: z.literal("none"),
    validatedClaim: z.literal("none"),
    persistence: z.literal("page-memory-only")
  })
  .strict();
export type ConditionDemoProfileV1 = z.infer<
  typeof ConditionDemoProfileV1Schema
>;

export function conditionDemoProfileRefFor(
  profile: ConditionDemoProfileV1
): ConditionDemoProfileRefV1 {
  return ConditionDemoProfileRefV1Schema.parse({
    profileId: profile.profileId,
    version: profile.version,
    contentSha256: profile.contentSha256
  });
}

function canonicalize(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalize(record[key])}`)
    .join(",")}}`;
}

export function conditionDemoProfileDigestInput(
  profile: ConditionDemoProfileV1
): string {
  const { contentSha256: _contentSha256, ...content } = profile;
  return canonicalize(content);
}

export async function verifyConditionDemoProfileDigest(
  profile: ConditionDemoProfileV1
): Promise<boolean> {
  const parsed = ConditionDemoProfileV1Schema.safeParse(profile);
  if (!parsed.success) return false;
  return (
    (await calculateSha256Hex(
      conditionDemoProfileDigestInput(parsed.data)
    )) ===
    parsed.data.contentSha256
  );
}

export const ParticipantAssertedSideV1Schema = z
  .object({
    side: z.enum(["left", "right"]),
    source: z.literal("participant-asserted"),
    verified: z.literal(false)
  })
  .strict();
export type ParticipantAssertedSideV1 = z.infer<
  typeof ParticipantAssertedSideV1Schema
>;

export const ConditionDemoContextV1Schema = z
  .object({
    schemaVersion: z.literal("phenometric.condition-demo-context.v1"),
    profileRef: ConditionDemoProfileRefV1Schema,
    subjectRef: IdSchema,
    assertedAffectedSide: ParticipantAssertedSideV1Schema,
    sourceProtocolRef: ConditionDemoSourceProtocolRefV1Schema,
    allowlistedMetricCodes: ConditionDemoMetricCodesTupleSchema,
    boundaryStatement: z.literal(CONDITION_DEMO_BOUNDARY_STATEMENT),
    prohibitedClaims: ConditionDemoProhibitedClaimsTupleSchema,
    persistence: z.literal("page-memory-only")
  })
  .strict();
export type ConditionDemoContextV1 = z.infer<
  typeof ConditionDemoContextV1Schema
>;

const LocalReferenceReviewerActionV1Schema = z
  .object({
    action: z.literal("use-as-in-memory-reference"),
    actor: z.literal("local-demo-user"),
    status: z.literal("accepted")
  })
  .strict();

function sameProtocolRef(
  left: { packId: string; version: string; contentSha256: string },
  right: { packId: string; version: string; contentSha256: string }
): boolean {
  return (
    left.packId === right.packId &&
    left.version === right.version &&
    left.contentSha256 === right.contentSha256
  );
}

export const AcceptedReferenceV1Schema = z
  .object({
    schemaVersion: z.literal("phenometric.accepted-reference.v1"),
    referenceId: IdSchema,
    demoContext: ConditionDemoContextV1Schema,
    observation: ObservationV3Schema,
    acceptedAt: IsoTimestampSchema,
    localReviewerAction: LocalReferenceReviewerActionV1Schema,
    persistence: z.literal("page-memory-only")
  })
  .strict()
  .superRefine((reference, context) => {
    if (reference.observation.subjectRef !== reference.demoContext.subjectRef) {
      context.addIssue({
        code: "custom",
        path: ["observation", "subjectRef"],
        message: "The accepted observation must belong to the demo subject."
      });
    }
    if (
      !sameProtocolRef(
        reference.observation.protocolRef,
        reference.demoContext.sourceProtocolRef
      )
    ) {
      context.addIssue({
        code: "custom",
        path: ["observation", "protocolRef"],
        message: "The accepted observation must use the demo source protocol."
      });
    }
    if (
      Date.parse(reference.acceptedAt) <
      Date.parse(reference.observation.endedAt)
    ) {
      context.addIssue({
        code: "custom",
        path: ["acceptedAt"],
        message: "Reference acceptance cannot precede observation completion."
      });
    }
    const outcomeCodes = new Set(
      reference.observation.metricOutcomes.map((outcome) => outcome.metricCode)
    );
    for (const metricCode of reference.demoContext.allowlistedMetricCodes) {
      if (!outcomeCodes.has(metricCode)) {
        context.addIssue({
          code: "custom",
          path: ["observation", "metricOutcomes"],
          message: `The accepted observation is missing ${metricCode}.`
        });
      }
    }
  });
export type AcceptedReferenceV1 = z.infer<
  typeof AcceptedReferenceV1Schema
>;
