import { randomUUID } from "node:crypto";
import type { Pool, PoolClient } from "pg";
import { digest } from "./canonical.js";
import { invariant } from "./errors.js";
import { inputHash, reviewHash } from "./repository.memory.js";
import type { AccessAudit,AppendInput, Episode, EvidenceSnapshot, InputRecord, Job, Repository, Review } from "./types.js";

type Row = Record<string, any>;
const iso = (value: Date | string): string => new Date(value).toISOString();
const episodeFrom = (r: Row): Episode => ({ id: r.id, tenantId: r.tenant_id, subjectRef: r.subject_ref, dataClass: r.data_class, createdAt: iso(r.created_at), revision: r.revision, metadata: r.metadata });
const inputFrom = (r: Row): InputRecord => ({ id: r.id, episodeId: r.episode_id, revision: r.revision, kind: r.kind, logicalId: r.logical_id, replacesRecordId: r.replaces_record_id, idempotencyKey: r.idempotency_key, contentHash: r.content_hash, payload: r.payload,sourceProvenance:r.source_provenance??null, actorId: r.actor_id, recordedAt: iso(r.recorded_at) });
const jobFrom = (r: Row): Job => ({ id: r.id, tenantId: r.tenant_id, episodeId: r.episode_id, inputRevision: r.input_revision, status: r.status, attempts: r.attempts, leaseToken: r.lease_token, leaseExpiresAt: r.lease_expires_at ? iso(r.lease_expires_at) : null, errorCode: r.error_code });
const snapshotFrom = (r: Row): EvidenceSnapshot => ({ id: r.id, episodeId: r.episode_id, inputRevision: r.input_revision, algorithmVersion: r.algorithm_version, inputHash: r.input_hash, analysis: r.analysis, createdAt: iso(r.created_at) });
const reviewFrom = (r: Row): Review => ({ id: r.id, episodeId: r.episode_id, snapshotId: r.snapshot_id, inputRevision: r.input_revision, payload:r.payload, recordedAt: iso(r.recorded_at) });

export class PostgresRepository implements Repository {
  constructor(private readonly pool: Pool) {}

  private async transaction<T>(run: (client: PoolClient) => Promise<T>): Promise<T> {
    const client = await this.pool.connect();
    try { await client.query("BEGIN"); const result = await run(client); await client.query("COMMIT"); return result; }
    catch (error) { await client.query("ROLLBACK"); throw error; }
    finally { client.release(); }
  }

  async createEpisode(input: Omit<Episode, "revision">) {
    return this.transaction(async client => {
      await client.query("INSERT INTO encounter_episodes(id,tenant_id,subject_ref,data_class,created_at,metadata) VALUES($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING", [input.id,input.tenantId,input.subjectRef,input.dataClass,input.createdAt,input.metadata]);
      const result = await client.query("SELECT * FROM encounter_episodes WHERE id=$1", [input.id]);
      const episode = episodeFrom(result.rows[0]);
      invariant(episode.tenantId === input.tenantId && episode.subjectRef === input.subjectRef && episode.dataClass === input.dataClass && digest(episode.metadata) === digest(input.metadata),409,"episode-conflict","Episode identity already exists with different content.");
      return episode;
    });
  }

  async getState(tenantId: string, episodeId: string) {
    return this.transaction(async client => {
      // A shared episode lock keeps the complete source set and revision consistent.
      const result = await client.query("SELECT * FROM encounter_episodes WHERE id=$1 AND tenant_id=$2 FOR SHARE",[episodeId,tenantId]);
      if (!result.rowCount) return null;
      const records = await client.query("SELECT * FROM encounter_inputs WHERE episode_id=$1 ORDER BY revision",[episodeId]);
      return { episode: episodeFrom(result.rows[0]), records: records.rows.map(inputFrom) };
    });
  }

  async append(input: AppendInput) {
    const result = await this.appendBatch([input]);
    return { record: result.records[0]!, duplicate: result.duplicate };
  }

  async appendBatch(inputs: AppendInput[]) {
    invariant(inputs.length > 0,400,"empty-batch","At least one input is required.");
    const first = inputs[0]!;
    invariant(inputs.every(input => input.episodeId === first.episodeId && input.tenantId === first.tenantId),400,"mixed-batch","A batch must belong to one episode.");
    return this.transaction(async client => {
      const episodeResult = await client.query("SELECT * FROM encounter_episodes WHERE id=$1 AND tenant_id=$2 FOR UPDATE",[first.episodeId,first.tenantId]);
      invariant(episodeResult.rowCount,404,"episode-not-found","Episode not found.");
      const episode = episodeFrom(episodeResult.rows[0]);
      const originalRevision = episode.revision;
      const records: InputRecord[] = [];
      let changed = false;
      for (const input of inputs) {
        const hash = inputHash(input);
        const previous = await client.query("SELECT * FROM encounter_inputs WHERE episode_id=$1 AND idempotency_key=$2",[episode.id,input.idempotencyKey]);
        if (previous.rowCount) {
          const record = inputFrom(previous.rows[0]);
          invariant(record.contentHash === hash,409,"idempotency-conflict","The idempotency key was used for different content.");
          records.push(record); continue;
        }
        invariant(input.expectedRevision === undefined || input.expectedRevision === originalRevision,409,"stale-revision","Episode changed; refresh before writing.");
        const current = await client.query("SELECT id FROM encounter_inputs WHERE episode_id=$1 AND kind=$2 AND logical_id=$3 ORDER BY revision DESC LIMIT 1",[episode.id,input.kind,input.logicalId]);
        invariant(current.rowCount ? input.replacesRecordId === current.rows[0].id : !input.replacesRecordId,409,"correction-conflict","Corrections must explicitly replace the current source version.");
        const result = await client.query("INSERT INTO encounter_inputs(id,episode_id,revision,kind,logical_id,replaces_record_id,idempotency_key,content_hash,payload,actor_id,recorded_at,source_provenance) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING *",[randomUUID(),episode.id,++episode.revision,input.kind,input.logicalId,input.replacesRecordId ?? null,input.idempotencyKey,hash,input.payload,input.actorId,input.recordedAt,input.sourceProvenance??null]);
        records.push(inputFrom(result.rows[0])); changed = true;
      }
      if (changed) {
        await client.query("UPDATE encounter_episodes SET revision=$2 WHERE id=$1",[episode.id,episode.revision]);
        await client.query("UPDATE encounter_jobs SET status='superseded' WHERE episode_id=$1 AND status IN ('pending','running')",[episode.id]);
        await client.query("INSERT INTO encounter_jobs(id,tenant_id,episode_id,input_revision,status) VALUES($1,$2,$3,$4,'pending')",[randomUUID(),episode.tenantId,episode.id,episode.revision]);
      }
      return { records, duplicate: !changed };
    });
  }

  async claimJob(now: string, leaseMs = 30_000) {
    return this.transaction(async client => {
      const result = await client.query("SELECT * FROM encounter_jobs WHERE status='pending' OR (status='running' AND lease_expires_at <= $1) ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1",[now]);
      if (!result.rowCount) return null;
      const updated = await client.query("UPDATE encounter_jobs SET status='running',attempts=attempts+1,lease_token=$2,lease_expires_at=$3 WHERE id=$1 RETURNING *",[result.rows[0].id,randomUUID(),new Date(Date.parse(now)+leaseMs).toISOString()]);
      return jobFrom(updated.rows[0]);
    });
  }

  async publish(job: Job, snapshot: EvidenceSnapshot) {
    return this.transaction(async client => {
      const episode = await client.query("SELECT revision FROM encounter_episodes WHERE id=$1 AND tenant_id=$2 FOR UPDATE",[job.episodeId,job.tenantId]);
      const result = await client.query("SELECT * FROM encounter_jobs WHERE id=$1 FOR UPDATE",[job.id]);
      const current = result.rowCount ? jobFrom(result.rows[0]) : null;
      if (episode.rows[0]?.revision !== job.inputRevision || current?.status !== "running" || current.leaseToken !== job.leaseToken) return false;
      invariant(snapshot.episodeId === job.episodeId && snapshot.inputRevision === job.inputRevision,400,"snapshot-mismatch","Snapshot does not match its claimed job.");
      await client.query("INSERT INTO encounter_snapshots(id,episode_id,input_revision,algorithm_version,input_hash,analysis,created_at) VALUES($1,$2,$3,$4,$5,$6,$7)",[snapshot.id,snapshot.episodeId,snapshot.inputRevision,snapshot.algorithmVersion,snapshot.inputHash,snapshot.analysis,snapshot.createdAt]);
      await client.query("UPDATE encounter_jobs SET status='completed' WHERE id=$1",[job.id]);
      return true;
    });
  }

  async failJob(job: Job, errorCode: string) {
    await this.pool.query("UPDATE encounter_jobs SET status=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,error_code=$3 WHERE id=$1 AND lease_token=$2 AND status='running'",[job.id,job.leaseToken,errorCode]);
  }

  async getEvidence(tenantId: string, episodeId: string) {
    const episode = await this.pool.query("SELECT id FROM encounter_episodes WHERE id=$1 AND tenant_id=$2",[episodeId,tenantId]);
    invariant(episode.rowCount,404,"episode-not-found","Episode not found.");
    const results = await Promise.all([
      this.pool.query("SELECT * FROM encounter_snapshots WHERE episode_id=$1 ORDER BY input_revision",[episodeId]),
      this.pool.query("SELECT * FROM encounter_reviews WHERE episode_id=$1 ORDER BY recorded_at,id",[episodeId]),
      this.pool.query("SELECT * FROM encounter_jobs WHERE episode_id=$1 ORDER BY input_revision",[episodeId])
    ]);
    return { snapshots: results[0].rows.map(snapshotFrom),reviews: results[1].rows.map(reviewFrom),jobs: results[2].rows.map(jobFrom) };
  }

  async getEvidenceState(tenantId:string,episodeId:string){
    return this.transaction(async client=>{
      const episode=await client.query("SELECT * FROM encounter_episodes WHERE id=$1 AND tenant_id=$2 FOR SHARE",[episodeId,tenantId]);
      if(!episode.rowCount)return null;
      const records=await client.query("SELECT * FROM encounter_inputs WHERE episode_id=$1 ORDER BY revision",[episodeId]);
      const snapshots=await client.query("SELECT * FROM encounter_snapshots WHERE episode_id=$1 ORDER BY input_revision",[episodeId]);
      const reviews=await client.query("SELECT * FROM encounter_reviews WHERE episode_id=$1 ORDER BY recorded_at,id",[episodeId]);
      const jobs=await client.query("SELECT * FROM encounter_jobs WHERE episode_id=$1 ORDER BY input_revision",[episodeId]);
      return {state:{episode:episodeFrom(episode.rows[0]),records:records.rows.map(inputFrom)},snapshots:snapshots.rows.map(snapshotFrom),reviews:reviews.rows.map(reviewFrom),jobs:jobs.rows.map(jobFrom)};
    });
  }

  async addReview(tenantId: string, review: Review, idempotencyKey: string) {
    return this.transaction(async client => {
      const episode = await client.query("SELECT revision FROM encounter_episodes WHERE id=$1 AND tenant_id=$2 FOR UPDATE",[review.episodeId,tenantId]);
      invariant(episode.rowCount,404,"episode-not-found","Episode not found.");
      const previous = await client.query("SELECT * FROM encounter_reviews WHERE episode_id=$1 AND idempotency_key=$2",[review.episodeId,idempotencyKey]);
      if (previous.rowCount) {
        const prior = reviewFrom(previous.rows[0]);
        invariant(prior.snapshotId === review.snapshotId && reviewHash(prior) === reviewHash(review),409,"idempotency-conflict","Review key already has different content.");
        return prior;
      }
      const evidence = await client.query("SELECT id,analysis FROM encounter_snapshots WHERE id=$1 AND episode_id=$2 AND input_revision=$3",[review.snapshotId,review.episodeId,review.inputRevision]);
      invariant(episode.rows[0].revision === review.inputRevision && evidence.rowCount,409,"stale-evidence","Review must reference the current immutable evidence snapshot.");
      invariant(evidence.rows[0].analysis.runId===review.payload.runId && evidence.rows[0].analysis.contentSha256===review.payload.runSha256,409,"review-run-mismatch","Review must bind the exact run and content digest.");
      await client.query("INSERT INTO encounter_reviews(id,episode_id,snapshot_id,input_revision,payload,recorded_at,idempotency_key) VALUES($1,$2,$3,$4,$5,$6,$7)",[review.id,review.episodeId,review.snapshotId,review.inputRevision,review.payload,review.recordedAt,idempotencyKey]);
      return review;
    });
  }
  async appendAccessAudit(event:AccessAudit){
    const result=await this.pool.query("INSERT INTO encounter_access_audit(id,tenant_id,episode_id,actor_id,recorded_at,action,result,input_revision,artifact_id) SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9 WHERE EXISTS(SELECT 1 FROM encounter_episodes WHERE id=$3 AND tenant_id=$2)",[event.id,event.tenantId,event.episodeId,event.actorId,event.recordedAt,event.action,event.result,event.inputRevision,event.artifactId]);
    invariant(result.rowCount,404,"episode-not-found","Episode not found.");
  }
}
