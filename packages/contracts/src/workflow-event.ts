import { z } from "zod";
import {
  EvidenceRefSchema,
  WithheldReasonCodeSchema
} from "./observation-v3.js";
import {
  AmbientModalitySchema,
  MetricCodeSchema,
  ProtocolRefSchema
} from "./protocol.js";

const IdSchema = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);

export const WorkflowStageV1Schema = z.enum([
  "requesting-permission",
  "calibrating",
  "observing",
  "finalizing",
  "report",
  "discarded",
  "error"
]);
export type WorkflowStageV1 = z.infer<typeof WorkflowStageV1Schema>;

export const WorkflowActorV1Schema = z.discriminatedUnion("kind", [
  z
    .object({
      kind: z.literal("application"),
      id: z.enum(["capture-web", "report-builder"]),
      version: z.string().min(1)
    })
    .strict(),
  z
    .object({
      kind: z.literal("processor"),
      id: z.enum(["voice-analysis", "facial-analysis"]),
      version: z.string().min(1)
    })
    .strict()
]);
export type WorkflowActorV1 = z.infer<typeof WorkflowActorV1Schema>;

const WorkflowEventBaseShape = {
  schemaVersion: z.literal("phenometric.workflow-event.v1"),
  eventId: z.string().uuid(),
  sequence: z.number().int().positive(),
  occurredAt: z.string().datetime({ offset: true }),
  sessionId: IdSchema,
  subjectRef: IdSchema,
  protocolRef: ProtocolRefSchema,
  actor: WorkflowActorV1Schema,
  stage: WorkflowStageV1Schema,
  summary: z.string().min(1).max(240),
  evidenceRefs: z.array(EvidenceRefSchema),
  causedByEventId: z.string().uuid().optional()
} as const;

function eventSchema<T extends string, S extends z.ZodRawShape>(
  type: T,
  payload: S
) {
  return z
    .object({
      ...WorkflowEventBaseShape,
      type: z.literal(type),
      payload: z.object(payload).strict()
    })
    .strict();
}

export const WorkflowEventV1Schema = z.discriminatedUnion("type", [
  eventSchema("consent.recorded", {
    consentId: IdSchema
  }),
  eventSchema("capture.permission.requested", {
    modalities: z.array(AmbientModalitySchema).min(1)
  }),
  eventSchema("capture.calibration.started", {
    modalities: z.array(AmbientModalitySchema).min(1)
  }),
  eventSchema("capture.lane.ready", {
    modality: AmbientModalitySchema
  }),
  eventSchema("capture.started", {
    startedAt: z.string().datetime({ offset: true })
  }),
  eventSchema("capture.quality.changed", {
    modality: AmbientModalitySchema,
    state: z.enum(["eligible", "withheld"]),
    reasonCodes: z.array(z.string().min(1))
  }),
  eventSchema("measurement.recorded", {
    measurementId: IdSchema,
    aggregateId: IdSchema,
    metricCode: MetricCodeSchema
  }),
  eventSchema("measurement.withheld", {
    outcomeId: IdSchema,
    aggregateId: IdSchema,
    metricCode: MetricCodeSchema,
    reasonCode: WithheldReasonCodeSchema
  }),
  eventSchema("capture.finalizing", {
    reason: z.enum(["manual", "maximum-duration"])
  }),
  eventSchema("capture.discarded", {
    reason: z.enum([
      "consent-withdrawn",
      "document-hidden",
      "user-cancelled"
    ])
  }),
  eventSchema("observation.created", {
    observationId: IdSchema
  }),
  eventSchema("report.created", {
    reportId: IdSchema,
    observationId: IdSchema
  }),
  eventSchema("capture.error", {
    code: z.string().min(1),
    modality: AmbientModalitySchema.optional(),
    recoverable: z.boolean()
  })
]).superRefine((event, context) => {
  const allowedStages: Record<typeof event.type, WorkflowStageV1[]> = {
    "consent.recorded": ["requesting-permission"],
    "capture.permission.requested": ["requesting-permission"],
    "capture.calibration.started": ["calibrating"],
    "capture.lane.ready": ["calibrating"],
    "capture.started": ["observing"],
    "capture.quality.changed": ["observing"],
    "measurement.recorded": ["observing", "finalizing"],
    "measurement.withheld": ["observing", "finalizing"],
    "capture.finalizing": ["finalizing"],
    "capture.discarded": ["discarded"],
    "observation.created": ["finalizing"],
    "report.created": ["report"],
    "capture.error": ["error"]
  };
  if (!allowedStages[event.type].includes(event.stage)) {
    context.addIssue({
      code: "custom",
      path: ["stage"],
      message: `${event.type} cannot occur in the ${event.stage} stage.`
    });
  }
});
export type WorkflowEventV1 = z.infer<typeof WorkflowEventV1Schema>;

type GeneratedEventFields =
  | "schemaVersion"
  | "eventId"
  | "sequence"
  | "occurredAt";

export type WorkflowEventInputV1 = WorkflowEventV1 extends infer Event
  ? Event extends WorkflowEventV1
    ? Omit<Event, GeneratedEventFields>
    : never
  : never;
