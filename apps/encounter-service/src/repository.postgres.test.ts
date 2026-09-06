import { randomUUID } from "node:crypto";
import { Pool } from "pg";
import { beforeAll,afterAll,describe,it,expect } from "vitest";
import { PostgresRepository } from "./repository.postgres.js";
import { migrate } from "./migration.js";
import { EncounterService } from "./service.js";
import { seedSyntheticEpisode,SYNTHETIC_PRINCIPAL as principal,SYNTHETIC_EPISODE_ID as syntheticId,syntheticFixture } from "./synthetic.js";
import { runOneJob } from "./worker.js";
import type { AppendInput,EvidenceSnapshot,Job } from "./types.js";

const databaseUrl=process.env.TEST_DATABASE_URL;
if(!databaseUrl)throw new Error("TEST_DATABASE_URL is required for actual PostgreSQL integration tests.");
const schema=`encounter_test_${randomUUID().replaceAll("-","")}`;
const administration=new Pool({connectionString:databaseUrl,max:2});
const pool=new Pool({connectionString:databaseUrl,max:12,options:`-c search_path=${schema}`});
const repository=new PostgresRepository(pool);
const now="2026-09-06T12:00:00.000Z";
beforeAll(async()=>{await administration.query(`CREATE SCHEMA ${schema}`);await migrate(pool);});
afterAll(async()=>{await pool.end();await administration.query(`DROP SCHEMA ${schema} CASCADE`);await administration.end();});
async function episode(){const id=randomUUID();await repository.createEpisode({id,tenantId:"test-tenant",subjectRef:"test-subject",dataClass:"synthetic",createdAt:now,metadata:{}});return id;}
function input(episodeId:string,key="event-1"):AppendInput{return {tenantId:"test-tenant",episodeId,kind:"session",logicalId:key,idempotencyKey:key,payload:{value:1},actorId:"test-actor",recordedAt:now};}
function snapshot(job:Job):EvidenceSnapshot{return {id:randomUUID(),episodeId:job.episodeId,inputRevision:job.inputRevision,algorithmVersion:"test-1",inputHash:"a".repeat(64),analysis:{runId:"test-run",contentSha256:"b".repeat(64),rows:[]},createdAt:now};}
async function claimFor(id:string){for(let i=0;i<30;i++){const job=await repository.claimJob(now);if(!job)throw new Error("no pending job");if(job.episodeId===id)return job;await repository.publish(job,snapshot(job));}throw new Error("job not reached");}

describe("real PostgreSQL transactions and restart",()=>{
  it("atomically deduplicates concurrent ingestion and generates exactly one job",async()=>{
    const id=await episode();
    const results=await Promise.all(Array.from({length:12},()=>repository.append(input(id))));
    expect(results.filter(result=>!result.duplicate)).toHaveLength(1);
    expect(new Set(results.map(result=>result.record.id)).size).toBe(1);
    const state=await repository.getState("test-tenant",id);expect(state?.episode.revision).toBe(1);expect(state?.records).toHaveLength(1);
    expect((await repository.getEvidence("test-tenant",id)).jobs).toHaveLength(1);
    await expect(repository.append({...input(id),payload:{value:2}})).rejects.toMatchObject({code:"idempotency-conflict"});
  });

  it("rolls back a whole batch and its job if any item conflicts",async()=>{
    const id=await episode();await repository.append(input(id,"existing"));
    await expect(repository.appendBatch([input(id,"new"),{...input(id,"existing"),payload:{value:99}}])).rejects.toMatchObject({code:"idempotency-conflict"});
    const state=await repository.getState("test-tenant",id);expect(state?.episode.revision).toBe(1);expect(state?.records).toHaveLength(1);
    expect((await repository.getEvidence("test-tenant",id)).jobs).toHaveLength(1);
  });

  it("does not publish stale analysis after a correction and fences expired worker leases",async()=>{
    const id=await episode();const first=await repository.append(input(id));const stale=await claimFor(id);
    await repository.append({...input(id,"correction"),logicalId:"event-1",replacesRecordId:first.record.id,payload:{value:2},expectedRevision:1});
    expect(await repository.publish(stale,snapshot(stale))).toBe(false);
    const worker1=await claimFor(id);
    const worker2=await repository.claimJob("2026-09-06T12:01:00.000Z");
    expect(worker2?.id).toBe(worker1.id);
    expect(await repository.publish(worker1,snapshot(worker1))).toBe(false);
    expect(await repository.publish(worker2!,snapshot(worker2!))).toBe(true);
  });

  it("enforces immutable source/evidence tables inside PostgreSQL",async()=>{
    const id=await episode();const record=(await repository.append(input(id))).record;const job=await claimFor(id);const evidence=snapshot(job);await repository.publish(job,evidence);
    await expect(pool.query("UPDATE encounter_inputs SET payload=$2 WHERE id=$1",[record.id,{value:42}])).rejects.toThrow("append-only");
    await expect(pool.query("DELETE FROM encounter_snapshots WHERE id=$1",[evidence.id])).rejects.toThrow("append-only");
    expect(await repository.getState("wrong-tenant",id)).toBeNull();
  });

  it("persists a complete consent/capture/worker/review workflow across repository restart",async()=>{
    const service=new EncounterService(repository,{mode:"synthetic",now:()=>now});await seedSyntheticEpisode(service,now);
    let result=await runOneJob(repository,{now:()=>now});
    for(let i=0;i<30&&(await service.evidence(principal,syntheticId)).status!=="ready";i++)result=await runOneJob(repository,{now:()=>now});
    expect(result).toBe("published");
    const evidence=await service.evidence(principal,syntheticId);expect(evidence.snapshot?.analysis.status).toBe("available");
    const review=await service.review(principal,syntheticId,{snapshotId:evidence.snapshot!.id,inputRevision:evidence.inputRevision,disposition:"acknowledged"},"review");
    const restart=new Pool({connectionString:databaseUrl,max:2,options:`-c search_path=${schema}`});
    try{
      const reloaded=new EncounterService(new PostgresRepository(restart),{mode:"synthetic",now:()=>now});
      const after=await reloaded.evidence(principal,syntheticId,true);
      expect(after.snapshot).toEqual(evidence.snapshot);expect(after.reviews).toEqual([review]);
      expect((await reloaded.captureContext(principal,syntheticId,"synthetic-encounter-1")).consent.consentId).toBe("synthetic-consent");
      const audit=await restart.query("SELECT * FROM encounter_access_audit WHERE episode_id=$1 ORDER BY recorded_at",[syntheticId]);
      expect(audit.rows.some(row=>row.action==="capture-context-read"&&row.actor_id===principal.sub&&row.result==="allowed")).toBe(true);
      expect((await reloaded.repository.getState(principal.tenantId,syntheticId))?.episode.revision).toBe(evidence.inputRevision);
      await expect(restart.query("DELETE FROM encounter_access_audit WHERE id=$1",[audit.rows[0].id])).rejects.toThrow("append-only");
    }finally{await restart.end();}
  });

  it("uses real row locking to reject a capture checked before consent withdrawal",async()=>{
    const fixture=await syntheticFixture(now);const state=(await repository.getState(principal.tenantId,syntheticId))!;
    const consentRecord=state.records.find(record=>record.kind==="consent")!;
    await repository.append({tenantId:principal.tenantId,episodeId:syntheticId,kind:"consent",logicalId:fixture.consent.consentId,replacesRecordId:consentRecord.id,expectedRevision:state.episode.revision,idempotencyKey:"withdraw",payload:{...fixture.consent,withdrawnAt:now},actorId:principal.sub,recordedAt:now});
    await expect(repository.append({tenantId:principal.tenantId,episodeId:syntheticId,kind:"session",logicalId:"racing-capture",expectedRevision:state.episode.revision,idempotencyKey:"racing-capture",payload:{},actorId:principal.sub,recordedAt:now})).rejects.toMatchObject({code:"stale-revision"});
  });
});
