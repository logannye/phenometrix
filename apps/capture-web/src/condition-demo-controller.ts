import {
  AcceptedReferenceV1Schema,
  ConditionEvidenceCardV1Schema,
  type AcceptedReferenceV1,
  type ConditionDemoContextV1,
  type ConditionEvidenceCardV1,
  type ObservationV3
} from "@phenometrix/contracts";
import { createUnilateralFacialMovementDemoContext } from "@phenometrix/condition-profiles";

export type ParticipantAssertedSide = "left" | "right";
export type ConditionCardReviewAction = "accept-card" | "dismiss-card";

function sameObservationRef(
  cardRef: ConditionEvidenceCardV1["comparison"]["referenceObservation"],
  observation: ObservationV3
): boolean {
  return (
    cardRef.observationId === observation.observationId &&
    cardRef.sessionId === observation.sessionId &&
    cardRef.startedAt === observation.startedAt &&
    cardRef.endedAt === observation.endedAt &&
    cardRef.protocolRef.packId === observation.protocolRef.packId &&
    cardRef.protocolRef.version === observation.protocolRef.version &&
    cardRef.protocolRef.contentSha256 === observation.protocolRef.contentSha256 &&
    cardRef.captureAdapter.id === observation.captureAdapter.id &&
    cardRef.captureAdapter.version === observation.captureAdapter.version
  );
}

/**
 * Owns only page-memory condition-demo state. Media, derived frame streams,
 * timers, workers, and workflow state remain under CaptureRuntime/main.ts.
 */
export class ConditionDemoController {
  #context: ConditionDemoContextV1 | null = null;
  #acceptedReference: AcceptedReferenceV1 | null = null;
  #latestObservation: ObservationV3 | null = null;
  #latestCard: ConditionEvidenceCardV1 | null = null;

  get context(): ConditionDemoContextV1 | null {
    return this.#context;
  }

  get acceptedReference(): AcceptedReferenceV1 | null {
    return this.#acceptedReference;
  }

  get latestObservation(): ObservationV3 | null {
    return this.#latestObservation;
  }

  get latestCard(): ConditionEvidenceCardV1 | null {
    return this.#latestCard;
  }

  startParticipant(
    subjectRef: string,
    assertedAffectedSide: ParticipantAssertedSide
  ): ConditionDemoContextV1 {
    if (this.#context) {
      if (
        this.#context.subjectRef !== subjectRef ||
        this.#context.assertedAffectedSide.side !== assertedAffectedSide
      ) {
        throw new Error("condition-demo-participant-context-mismatch");
      }
      return this.#context;
    }
    this.#context = createUnilateralFacialMovementDemoContext({
      subjectRef,
      assertedAffectedSide
    });
    return this.#context;
  }

  recordFinalizedObservation(observation: ObservationV3): void {
    if (!this.#context) throw new Error("condition-demo-context-unavailable");
    if (observation.subjectRef !== this.#context.subjectRef) {
      throw new Error("condition-demo-observation-subject-mismatch");
    }
    this.#latestObservation = observation;
    this.#latestCard = null;
  }

  acceptLatestAsReference(input: {
    referenceId: string;
    acceptedAt: string;
  }): AcceptedReferenceV1 {
    if (!this.#context || !this.#latestObservation) {
      throw new Error("condition-demo-reference-source-unavailable");
    }
    if (this.#acceptedReference) {
      throw new Error("condition-demo-reference-already-accepted");
    }
    const acceptedReference = AcceptedReferenceV1Schema.parse({
      schemaVersion: "phenometric.accepted-reference.v1",
      referenceId: input.referenceId,
      demoContext: this.#context,
      observation: this.#latestObservation,
      acceptedAt: input.acceptedAt,
      localReviewerAction: {
        action: "use-as-in-memory-reference",
        actor: "local-demo-user",
        status: "accepted"
      },
      persistence: "page-memory-only"
    });
    this.#acceptedReference = acceptedReference;
    return acceptedReference;
  }

  prepareFollowUp(): void {
    if (!this.#acceptedReference) {
      throw new Error("condition-demo-reference-unavailable");
    }
    this.#latestObservation = null;
    this.#latestCard = null;
  }

  recordCard(card: ConditionEvidenceCardV1): ConditionEvidenceCardV1 {
    if (!this.#acceptedReference || !this.#latestObservation) {
      throw new Error("condition-demo-card-source-unavailable");
    }
    if (
      card.comparison.acceptedReferenceId !==
        this.#acceptedReference.referenceId ||
      !sameObservationRef(
        card.comparison.referenceObservation,
        this.#acceptedReference.observation
      ) ||
      !sameObservationRef(
        card.comparison.currentObservation,
        this.#latestObservation
      ) ||
      JSON.stringify(card.comparison.demoContext) !==
        JSON.stringify(this.#context)
    ) {
      throw new Error("condition-demo-card-source-mismatch");
    }
    this.#latestCard = ConditionEvidenceCardV1Schema.parse(card);
    return this.#latestCard;
  }

  reviewLatestCard(
    action: ConditionCardReviewAction,
    recordedAt: string
  ): ConditionEvidenceCardV1 {
    if (!this.#latestCard) throw new Error("condition-demo-card-unavailable");
    if (this.#latestCard.review.status !== "pending") {
      throw new Error("condition-demo-card-already-reviewed");
    }
    this.#latestCard = ConditionEvidenceCardV1Schema.parse({
      ...this.#latestCard,
      review: {
        status: action === "accept-card" ? "accepted" : "dismissed",
        action,
        actor: "local-demo-user",
        recordedAt
      }
    });
    return this.#latestCard;
  }

  clear(): void {
    this.#context = null;
    this.#acceptedReference = null;
    this.#latestObservation = null;
    this.#latestCard = null;
  }
}
