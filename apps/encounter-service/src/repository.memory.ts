import { randomUUID } from "node:crypto";
import { digest } from "./canonical.js";
import { invariant } from "./errors.js";
import type { AccessAudit,AppendInput, Episode, EvidenceSnapshot, InputRecord, Job, Repository, Review } from "./types.js";
import { effectiveRecords } from "./types.js";

const copy = <T>(value: T): T => structuredClone(value);
export function inputHash(input: AppendInput): string {
  return digest({ kind: input.kind, logicalId: input.logicalId, replacesRecordId: input.replacesRecordId ?? null, payload: input.payload,sourceProvenance:input.sourceProvenance??null });
}

/** Deterministic, isolated synthetic/test implementation. Never used by the live launcher. */
export class MemoryRepository implements Repository {
  private episodes = new Map<string, Episode>();
  private records: InputRecord[] = [];
  private jobs: Job[] = [];
  private snapshots: EvidenceSnapshot[] = [];
  private reviews: Array<Review & { key: string }> = [];
  private readonly audits:AccessAudit[]=[];
  get accessAudit():AccessAudit[]{return copy(this.audits);}

  async createEpisode(input: Omit<Episode, "revision">): Promise<Episode> {
    const previous = this.episodes.get(input.id);
    if (previous) {
      invariant(previous.tenantId === input.tenantId && previous.subjectRef === input.subjectRef && previous.dataClass === input.dataClass && digest(previous.metadata) === digest(input.metadata), 409, "episode-conflict", "Episode identity already exists with different content.");
      return copy(previous);
    }
    const episode = { ...copy(input), revision: 0 };
    this.episodes.set(episode.id, episode);
    return copy(episode);
  }

  async getState(tenantId: string, episodeId: string) {
    const episode = this.episodes.get(episodeId);
    return episode?.tenantId === tenantId ? copy({ episode, records: this.records.filter((record) => record.episodeId === episodeId) }) : null;
  }

  async append(input: AppendInput) {
    const result = await this.appendBatch([input]);
    return { record: result.records[0]!, duplicate: result.duplicate };
  }

  async appendBatch(inputs: AppendInput[]) {
    invariant(inputs.length > 0, 400, "empty-batch", "At least one input is required.");
    const first = inputs[0]!;
    const stored = this.episodes.get(first.episodeId);
    invariant(stored?.tenantId === first.tenantId, 404, "episode-not-found", "Episode not found.");
    invariant(inputs.every((input) => input.episodeId === first.episodeId && input.tenantId === first.tenantId), 400, "mixed-batch", "A batch must belong to one episode.");
    const episode = copy(stored);
    const working = copy(this.records);
    const results: InputRecord[] = [];
    let changed = false;
    for (const input of inputs) {
      const hash = inputHash(input);
      const previous = working.find((record) => record.episodeId === episode.id && record.idempotencyKey === input.idempotencyKey);
      if (previous) {
        invariant(previous.contentHash === hash, 409, "idempotency-conflict", "The idempotency key was used for different content.");
        results.push(previous);
        continue;
      }
      invariant(input.expectedRevision === undefined || input.expectedRevision === stored.revision, 409, "stale-revision", "Episode changed; refresh before writing.");
      const current = effectiveRecords(working.filter((record) => record.episodeId === episode.id))
        .find((record) => record.kind === input.kind && record.logicalId === input.logicalId);
      invariant(current ? input.replacesRecordId === current.id : !input.replacesRecordId, 409, "correction-conflict", "Corrections must explicitly replace the current source version.");
      const record: InputRecord = { id: randomUUID(), episodeId: episode.id, revision: ++episode.revision,
        kind: input.kind, logicalId: input.logicalId, replacesRecordId: input.replacesRecordId ?? null,
        idempotencyKey: input.idempotencyKey, contentHash: hash, payload: copy(input.payload),sourceProvenance:copy(input.sourceProvenance??null), actorId: input.actorId, recordedAt: input.recordedAt };
      working.push(record);
      results.push(record);
      changed = true;
    }
    if (changed) {
      this.records = working;
      this.episodes.set(episode.id, episode);
      for (const job of this.jobs) if (job.episodeId === episode.id && ["pending", "running"].includes(job.status)) job.status = "superseded";
      this.jobs.push({ id: randomUUID(), tenantId: episode.tenantId, episodeId: episode.id, inputRevision: episode.revision, status: "pending", attempts: 0, leaseToken: null, leaseExpiresAt: null, errorCode: null });
    }
    return copy({ records: results, duplicate: !changed });
  }

  async claimJob(now: string, leaseMs = 30_000) {
    const job = this.jobs.find((candidate) => candidate.status === "pending" || (candidate.status === "running" && Date.parse(candidate.leaseExpiresAt!) <= Date.parse(now)));
    if (!job) return null;
    job.status = "running";
    job.attempts++;
    job.leaseToken = randomUUID();
    job.leaseExpiresAt = new Date(Date.parse(now) + leaseMs).toISOString();
    return copy(job);
  }

  async publish(job: Job, snapshot: EvidenceSnapshot) {
    const current = this.jobs.find((item) => item.id === job.id);
    if (current?.status !== "running" || current.leaseToken !== job.leaseToken || this.episodes.get(job.episodeId)?.revision !== job.inputRevision) return false;
    invariant(snapshot.episodeId === job.episodeId && snapshot.inputRevision === job.inputRevision, 400, "snapshot-mismatch", "Snapshot does not match its claimed job.");
    this.snapshots.push(copy(snapshot));
    current.status = "completed";
    return true;
  }

  async failJob(job: Job, errorCode: string) {
    const current = this.jobs.find((item) => item.id === job.id);
    if (current?.status === "running" && current.leaseToken === job.leaseToken) {
      current.status = current.attempts < 3 ? "pending" : "failed";
      current.errorCode = errorCode;
    }
  }

  async getEvidence(tenantId: string, episodeId: string) {
    invariant(this.episodes.get(episodeId)?.tenantId === tenantId, 404, "episode-not-found", "Episode not found.");
    return copy({ snapshots: this.snapshots.filter((item) => item.episodeId === episodeId),
      reviews: this.reviews.filter((item) => item.episodeId === episodeId).map(({ key: _key, ...review }) => review),
      jobs: this.jobs.filter((item) => item.episodeId === episodeId) });
  }

  async getEvidenceState(tenantId:string,episodeId:string){
    const episode=this.episodes.get(episodeId);
    if(episode?.tenantId!==tenantId)return null;
    return copy({state:{episode,records:this.records.filter(record=>record.episodeId===episodeId)},
      snapshots:this.snapshots.filter(item=>item.episodeId===episodeId),
      reviews:this.reviews.filter(item=>item.episodeId===episodeId).map(({key:_key,...review})=>review),
      jobs:this.jobs.filter(item=>item.episodeId===episodeId)});
  }

  async addReview(tenantId: string, review: Review, idempotencyKey: string) {
    const episode = this.episodes.get(review.episodeId);
    invariant(episode?.tenantId === tenantId, 404, "episode-not-found", "Episode not found.");
    const prior = this.reviews.find((item) => item.episodeId === review.episodeId && item.key === idempotencyKey);
    if (prior) {
      invariant(prior.snapshotId === review.snapshotId && reviewHash(prior) === reviewHash(review), 409, "idempotency-conflict", "Review key already has different content.");
      const { key: _key, ...stored } = prior;
      return copy(stored);
    }
    invariant(episode.revision === review.inputRevision && this.snapshots.some((item) => item.id === review.snapshotId && item.episodeId === episode.id && item.inputRevision === review.inputRevision), 409, "stale-evidence", "Review must reference the current immutable evidence snapshot.");
    const snapshot=this.snapshots.find(item=>item.id===review.snapshotId)!;
    invariant(snapshot.analysis.runId===review.payload.runId && snapshot.analysis.contentSha256===review.payload.runSha256,409,"review-run-mismatch","Review must bind the exact run and content digest.");
    this.reviews.push({ ...copy(review), key: idempotencyKey });
    return copy(review);
  }
  async appendAccessAudit(event:AccessAudit){
    invariant(this.episodes.get(event.episodeId)?.tenantId===event.tenantId,404,"episode-not-found","Episode not found.");
    this.audits.push(copy(event));
  }
}

export function reviewHash(review: Review): string {
  const {reviewId:_id,recordedAt:_at,...content}=review.payload;
  return digest(content);
}
