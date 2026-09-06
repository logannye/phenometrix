import type { TreatmentResponseReviewV1 } from "@phenometrix/contracts";
export type JsonObject = Record<string, unknown>;
export type RecordKind = "session" | "clinical-event" | "consent" | "binding" | "annotation";
export type DataClass = "synthetic" | "consented-research";

export interface Episode {
  id: string;
  tenantId: string;
  subjectRef: string;
  dataClass: DataClass;
  createdAt: string;
  revision: number;
  metadata: JsonObject;
}

/** Immutable inputs. logicalId identifies the entity; replacesRecordId is an explicit correction. */
export interface InputRecord {
  id: string;
  episodeId: string;
  revision: number;
  kind: RecordKind;
  logicalId: string;
  replacesRecordId: string | null;
  idempotencyKey: string;
  contentHash: string;
  payload: JsonObject;
  sourceProvenance: JsonObject | null;
  actorId: string;
  recordedAt: string;
}

export interface AppendInput {
  tenantId: string;
  episodeId: string;
  kind: RecordKind;
  logicalId: string;
  replacesRecordId?: string | null;
  expectedRevision?: number;
  idempotencyKey: string;
  payload: JsonObject;
  sourceProvenance?: JsonObject | null;
  actorId: string;
  recordedAt: string;
}

export interface Job {
  id: string;
  tenantId: string;
  episodeId: string;
  inputRevision: number;
  status: "pending" | "running" | "completed" | "superseded" | "failed";
  attempts: number;
  leaseToken: string | null;
  leaseExpiresAt: string | null;
  errorCode: string | null;
}

export interface EpisodeState {
  episode: Episode;
  records: InputRecord[];
}

export interface EvidenceSnapshot {
  id: string;
  episodeId: string;
  inputRevision: number;
  algorithmVersion: string;
  inputHash: string;
  analysis: JsonObject;
  createdAt: string;
}

export interface Review {
  id: string;
  episodeId: string;
  snapshotId: string;
  inputRevision: number;
  payload: TreatmentResponseReviewV1;
  recordedAt: string;
}

export interface AccessAudit {
  id:string;tenantId:string;episodeId:string;actorId:string;recordedAt:string;
  action:"capture-context-read"|"evidence-read"|"evidence-history-read";
  result:"allowed"|"denied";inputRevision:number;artifactId:string|null;
}

export interface Repository {
  createEpisode(input: Omit<Episode, "revision">): Promise<Episode>;
  getState(tenantId: string, episodeId: string): Promise<EpisodeState | null>;
  append(input: AppendInput): Promise<{ record: InputRecord; duplicate: boolean }>;
  appendBatch(inputs: AppendInput[]): Promise<{ records: InputRecord[]; duplicate: boolean }>;
  claimJob(now: string, leaseMs?: number): Promise<Job | null>;
  publish(job: Job, snapshot: EvidenceSnapshot): Promise<boolean>;
  failJob(job: Job, errorCode: string): Promise<void>;
  getEvidence(tenantId: string, episodeId: string): Promise<{ snapshots: EvidenceSnapshot[]; reviews: Review[]; jobs: Job[] }>;
  getEvidenceState(tenantId:string,episodeId:string):Promise<{state:EpisodeState;snapshots:EvidenceSnapshot[];reviews:Review[];jobs:Job[]}|null>;
  addReview(tenantId: string, review: Review, idempotencyKey: string): Promise<Review>;
  appendAccessAudit(event:AccessAudit):Promise<void>;
}

/** Resolve corrections deterministically without mutating or deleting source versions. */
export function effectiveRecords(records: InputRecord[]): InputRecord[] {
  const latest = new Map<string, InputRecord>();
  for (const record of [...records].sort((a, b) => a.revision - b.revision)) {
    latest.set(`${record.kind}:${record.logicalId}`, record);
  }
  return [...latest.values()];
}
