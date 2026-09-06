import {
  AMBIENT_LOCAL_PROTOCOL_REF,
  CONDITION_DEMO_BOUNDARY_STATEMENT,
  CONDITION_DEMO_METRIC_CODES,
  CONDITION_DEMO_PROHIBITED_CLAIMS,
  ConditionDemoContextV1Schema,
  ConditionDemoProfileV1Schema,
  conditionDemoProfileRefFor,
  type ConditionDemoContextV1,
  type ConditionDemoProfileV1
} from "@phenometrix/contracts";

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
}

const rawProfile = {
  schemaVersion: "phenometric.condition-demo-profile.v1",
  profileId: "unilateral-facial-movement-research-demo",
  version: "1.0.0",
  contentSha256:
    "28b9d9045f8afe1bd08d283779646a6d5c68f0787db6d10f3fd0339c93d1b7a5",
  status: "nonclinical-research-demo",
  displayName: "Unilateral Facial Movement Research Demo",
  sourceProtocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
  intendedUse:
    "Within-page repeatability demonstration for a participant with previously established unilateral peripheral facial palsy.",
  targetPopulation:
    "Adults participating in a nonclinical research demonstration who assert a previously established unilateral peripheral facial palsy.",
  affectedSidePolicy: {
    source: "participant-asserted",
    verified: false,
    automaticInference: false
  },
  metrics: [
    {
      metricCode: "ambient.face.rest_mouth_corner_asymmetry.signed",
      displayLabel:
        "Resting mouth difference (subject-left minus subject-right)",
      displayRole: "primary",
      laterality: "subject-left-minus-subject-right"
    },
    {
      metricCode: "ambient.face.rest_eye_aperture_asymmetry.signed",
      displayLabel:
        "Resting eye-aperture difference (subject-left minus subject-right)",
      displayRole: "primary",
      laterality: "subject-left-minus-subject-right"
    },
    {
      metricCode: "ambient.face.lid_closure_completeness.left",
      displayLabel: "Subject-left lid closure completeness",
      displayRole: "primary",
      laterality: "explicit-subject-left"
    },
    {
      metricCode: "ambient.face.lid_closure_completeness.right",
      displayLabel: "Subject-right lid closure completeness",
      displayRole: "primary",
      laterality: "explicit-subject-right"
    },
    {
      metricCode: "ambient.face.spontaneous_excursion_asymmetry.median",
      displayLabel:
        "Spontaneous excursion difference (subject-left minus subject-right)",
      displayRole: "experimental",
      laterality: "subject-left-minus-subject-right"
    },
    {
      metricCode: "ambient.face.oculo_oral_synkinesis_index",
      displayLabel: "Oculo-oral coupling difference",
      displayRole: "experimental",
      laterality: "subject-left-minus-subject-right"
    }
  ],
  boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
  prohibitedClaims: CONDITION_DEMO_PROHIBITED_CLAIMS,
  clinicalValidation: "none",
  validatedClaim: "none",
  persistence: "page-memory-only"
} as const;

export const UNILATERAL_FACIAL_MOVEMENT_PROFILE: ConditionDemoProfileV1 =
  deepFreeze(ConditionDemoProfileV1Schema.parse(rawProfile));

export const UNILATERAL_FACIAL_MOVEMENT_PROFILE_REF = deepFreeze(
  conditionDemoProfileRefFor(UNILATERAL_FACIAL_MOVEMENT_PROFILE)
);

export interface CreateUnilateralFacialMovementDemoContextInput {
  subjectRef: string;
  assertedAffectedSide: "left" | "right";
}

export function createUnilateralFacialMovementDemoContext(
  input: CreateUnilateralFacialMovementDemoContextInput
): ConditionDemoContextV1 {
  return deepFreeze(
    ConditionDemoContextV1Schema.parse({
      schemaVersion: "phenometric.condition-demo-context.v1",
      profileRef: UNILATERAL_FACIAL_MOVEMENT_PROFILE_REF,
      subjectRef: input.subjectRef,
      assertedAffectedSide: {
        side: input.assertedAffectedSide,
        source: "participant-asserted",
        verified: false
      },
      sourceProtocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      allowlistedMetricCodes: CONDITION_DEMO_METRIC_CODES,
      boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
      prohibitedClaims: CONDITION_DEMO_PROHIBITED_CLAIMS,
      persistence: "page-memory-only"
    })
  );
}
