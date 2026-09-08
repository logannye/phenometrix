import { describe,it,expect } from "vitest";
import { MemoryRepository } from "./repository.memory.js";
import { EncounterService } from "./service.js";
import { runOneJob } from "./worker.js";
import { syntheticFixture,seedSyntheticEpisode,SYNTHETIC_PRINCIPAL as principal,SYNTHETIC_EPISODE_ID as episodeId } from "./synthetic.js";

const now="2026-09-06T12:00:00.000Z";
async function setup(){
  const repository=new MemoryRepository();
  const service=new EncounterService(repository,{mode:"synthetic",now:()=>now});
  await seedSyntheticEpisode(service,now);
  return {repository,service,fixture:await syntheticFixture(now)};
}

describe("consented encounter workflow",()=>{
  it("persists source history, coalesces jobs and publishes a source-linked analysis",async()=>{
    const {repository,service}=await setup();
    const before=await service.evidence(principal,episodeId);
    expect(before.status).toBe("pending");expect(before.snapshot).toBeNull();
    expect(before.jobs.filter(job=>job.status==="pending")).toHaveLength(1);
    expect(await runOneJob(repository,{now:()=>now})).toBe("published");
    const evidence=await service.evidence(principal,episodeId);
    expect(evidence.status).toBe("ready");
    expect(evidence.snapshot?.analysis.status).toBe("available");
    expect(evidence.snapshot?.analysis.clinicalClaim).toBe("descriptive-only");
    expect((evidence.snapshot?.analysis.rows as any[])[0].points).toHaveLength(4);
    expect(await runOneJob(repository,{now:()=>now})).toBe("idle");
  });

  it("returns exactly the same input for a retry and rejects conflicting retries",async()=>{
    const {service,fixture,repository}=await setup();
    const observation=fixture.sessions[0]!.observation;
    const previous=(await repository.getState(principal.tenantId,episodeId))!;
    const duplicate=await service.write(principal,episodeId,"sessions",{data:observation},observation.revisionId);
    expect(duplicate.duplicate).toBe(true);
    expect((await repository.getState(principal.tenantId,episodeId))?.episode.revision).toBe(previous.episode.revision);
    const changed=structuredClone(observation);changed.metrics[0]!.value=0.42;
    await expect(service.write(principal,episodeId,"sessions",{data:changed},observation.revisionId)).rejects.toMatchObject({code:"idempotency-conflict"});
  });

  it("prevents cross-tenant, study, participant and modality access",async()=>{
    const {service,fixture}=await setup();
    await expect(service.getEpisode({...principal,tenantId:"other"},episodeId)).rejects.toMatchObject({status:404});
    await expect(service.getEpisode({...principal,studyIds:["other"]},episodeId)).rejects.toMatchObject({code:"study-out-of-scope"});
    await expect(service.getEpisode({...principal,subjectRefs:["other"]},episodeId)).rejects.toMatchObject({code:"subject-out-of-scope"});
    const observation=structuredClone(fixture.sessions[0]!.observation);observation.metrics[0]!.modality="voice";
    await expect(service.write(principal,episodeId,"sessions",{data:observation},"voice-forgery")).rejects.toMatchObject({code:"modality-not-consented"});
  });

  it("rejects synthetic clock claims on live observations before persisting or queueing",async()=>{
    const fixture=await syntheticFixture(now),repository=new MemoryRepository();
    const service=new EncounterService(repository,{mode:"live",now:()=>now,allowedProtocolDigests:[fixture.episode.protocol.contentSha256]});
    const livePrincipal={...principal,dataClass:"consented-research" as const};
    await service.createEpisode(livePrincipal,{...fixture.episode,dataClass:"consented-research"});
    await service.write(livePrincipal,episodeId,"consents",{data:fixture.consent},"consent");
    await service.write(livePrincipal,episodeId,"bindings",{data:{...fixture.sessions[0]!.binding,verificationMethod:"clinic-enrollment"}},"binding");
    const observation=fixture.sessions[0]!.observation;
    const captured={...observation,capture:{...observation.capture,sourceKind:"patient-local-pre-codec",clockSource:{sourceId:"synthetic-local-clock",kind:"synthetic"},clockUncertaintyMs:0}};
    const revision=(await repository.getState(principal.tenantId,episodeId))!.episode.revision;
    await expect(service.write(livePrincipal,episodeId,"sessions",{data:captured},"live-clock")).rejects.toMatchObject({status:403,code:"clock-source-mode-mismatch"});
    expect((await repository.getState(principal.tenantId,episodeId))!.episode.revision).toBe(revision);
    expect((await service.write(livePrincipal,episodeId,"sessions",{data:{...captured,capture:{...captured.capture,clockSource:{sourceId:"qualified-monitor",kind:"monitored-utc"}}}},"live-clock")).record.kind).toBe("session");
  });

  it("rejects orphan, duplicate-ID and chronologically invalid corrections before queueing",async()=>{
    const {service,fixture,repository}=await setup();
    await expect(service.write(principal,episodeId,"clinical-events",{data:{...fixture.treatment,treatmentId:"new",revisionId:"new-revision",supersedesRevisionId:"missing"}},"orphan")).rejects.toMatchObject({code:"correction-parent-missing"});
    await expect(service.write(principal,episodeId,"clinical-events",{data:{...fixture.treatment,treatmentId:"new"}},"duplicate-revision")).rejects.toMatchObject({code:"revision-identity-conflict"});
    await expect(service.write(principal,episodeId,"clinical-events",{data:{...fixture.treatment,revisionId:"correction",supersedesRevisionId:fixture.treatment.revisionId,recordedAt:"2020-01-01T00:00:00Z"}},"old-correction")).rejects.toMatchObject({code:"correction-chronology"});
    const state=(await repository.getState(principal.tenantId,episodeId))!;
    const originalTreatment=state.records.find(record=>record.kind==="clinical-event")!;
    await expect(service.write(principal,episodeId,"clinical-events",{data:{...fixture.treatment,product:"Changed under same revision"},supersedesRecordId:originalTreatment.id},"reused-revision")).rejects.toMatchObject({code:"revision-identity-conflict"});
    const originalSession=state.records.find(record=>record.kind==="session")!;
    await expect(service.write(principal,episodeId,"sessions",{data:fixture.sessions[0]!.observation,supersedesRecordId:originalSession.id},"reused-session-revision")).rejects.toMatchObject({code:"revision-identity-conflict"});
    await expect(service.write(principal,episodeId,"clinical-events",{data:{schemaVersion:"phenometric.clinical-context-revision.v1",contextId:fixture.treatment.treatmentId,revisionId:"cross-type",supersedesRevisionId:fixture.treatment.revisionId,scope:fixture.episode.scope,recordedAt:now,effectiveTime:fixture.treatment.effectiveTime,status:"active",kind:"procedure",valueCode:"other",sourceRef:fixture.treatment.sourceRef}},"cross-type")).rejects.toMatchObject({code:"clinical-record-class-change"});
  });

  it("returns capture context only for the verified encounter and rejects raw/prompted additions",async()=>{
    const {service,fixture}=await setup();
    const context=await service.captureContext(principal,episodeId,fixture.sessions[0]!.observation.encounterId);
    expect(context.binding.bindingId).toBe(fixture.sessions[0]!.binding.bindingId);
    await expect(service.captureContext(principal,episodeId,"wrong-encounter")).rejects.toMatchObject({code:"binding-required"});
    await expect(service.write(principal,episodeId,"sessions",{data:{...fixture.sessions[0]!.observation,rawVideo:"not-allowed"}},"raw")).rejects.toHaveProperty("name","ZodError");
    const observation=structuredClone(fixture.sessions[0]!.observation);observation.metrics[0]!.context="prompted-smile";
    await expect(service.write(principal,episodeId,"sessions",{data:observation},"prompt")).rejects.toMatchObject({code:"ambient-only"});
    await expect(service.write(principal,episodeId,"sessions",{data:{...fixture.sessions[0]!.observation,recordedAt:"2030-01-01T00:00:00Z"}},"future-knowledge")).rejects.toMatchObject({code:"recorded-at-future"});
  });

  it("keeps binding identity immutable and hides revoked-binding historical evidence",async()=>{
    const {service,repository,fixture}=await setup();await runOneJob(repository,{now:()=>now});
    const state=(await repository.getState(principal.tenantId,episodeId))!;
    const record=state.records.find(item=>item.kind==="binding")!;
    const binding=fixture.sessions[0]!.binding;
    await expect(service.write(principal,episodeId,"bindings",{data:{...binding,platformParticipantId:"different-person"},supersedesRecordId:record.id},"rebinding")).rejects.toMatchObject({code:"binding-identity-immutable"});
    await service.write(principal,episodeId,"bindings",{data:{...binding,status:"revoked",revokedAt:now},supersedesRecordId:record.id},"revoke-binding");
    expect((await service.evidence(principal,episodeId,true)).snapshots).toHaveLength(0);
    await expect(service.captureContext(principal,episodeId,binding.encounterId)).rejects.toMatchObject({code:"binding-required"});
  });

  it("preserves earlier evidence, prevents stale publication and binds review to the exact run",async()=>{
    const {service,repository,fixture}=await setup();
    await runOneJob(repository,{now:()=>now});
    const initial=await service.evidence(principal,episodeId);
    const review=await service.review(principal,episodeId,{snapshotId:initial.snapshot!.id,inputRevision:initial.inputRevision,disposition:"acknowledged"},"review-1");
    expect(review.runSha256).toBe(initial.snapshot!.analysis.contentSha256);
    const treatment={...fixture.treatment,revisionId:"synthetic-treatment-v2",supersedesRevisionId:fixture.treatment.revisionId,product:"Corrected synthetic product"};
    await service.write(principal,episodeId,"clinical-events",{data:treatment},"treatment-correction");
    expect((await service.evidence(principal,episodeId)).snapshot).toBeNull();
    await expect(service.review(principal,episodeId,{snapshotId:initial.snapshot!.id,inputRevision:initial.inputRevision,disposition:"dismissed"},"review-stale")).rejects.toMatchObject({code:"stale-evidence"});
    let changed=false;
    const result=await runOneJob(repository,{now:()=>now,analyze:async()=>{
      const correction={...treatment,revisionId:"synthetic-treatment-v3",supersedesRevisionId:treatment.revisionId,status:"entered-in-error"};
      await service.write(principal,episodeId,"clinical-events",{data:correction},"treatment-retracted");changed=true;
      return initial.snapshot!.analysis;
    }});
    expect(changed).toBe(true);expect(result).toBe("superseded");
    await runOneJob(repository,{now:()=>now});
    const history=await service.evidence(principal,episodeId,true);
    expect(history.snapshots).toHaveLength(2);
    expect(history.snapshots?.[0]?.analysis).toEqual(initial.snapshot!.analysis);
    expect(history.snapshot?.analysis.status).toBe("anchor-unavailable");
    expect((await service.write(principal,episodeId,"clinical-events",{data:fixture.treatment},"synthetic-treatment")).duplicate).toBe(true);
  });

  it("revocation blocks ingestion/access and a new grant cannot expose old snapshots",async()=>{
    const {service,repository,fixture}=await setup();await runOneJob(repository,{now:()=>now});
    const state=(await repository.getState(principal.tenantId,episodeId))!;
    const consentRecord=state.records.find(record=>record.kind==="consent")!;
    await service.write(principal,episodeId,"consents",{data:{...fixture.consent,withdrawnAt:now},supersedesRecordId:consentRecord.id},"withdraw");
    await expect(service.evidence(principal,episodeId,true)).rejects.toMatchObject({code:"analysis-consent-inactive"});
    await expect(service.write(principal,episodeId,"sessions",{data:fixture.sessions[0]!.observation},"after-withdraw")).rejects.toMatchObject({code:"consent-required"});
    await service.write(principal,episodeId,"consents",{data:{...fixture.consent,consentId:"new-consent",grantedAt:now}},"new-consent");
    expect((await service.evidence(principal,episodeId,true)).snapshots).toHaveLength(0);
    expect(repository.accessAudit.some(event=>event.action==="evidence-history-read"&&event.result==="denied")).toBe(true);
    await expect(service.write(principal,episodeId,"consents",{data:fixture.consent,supersedesRecordId:consentRecord.id},"revive")).rejects.toMatchObject({code:"consent-withdrawal-immutable"});
  });

  it("requires a configured exact-clip authority, not a generic research checkbox",async()=>{
    const {service,fixture}=await setup();
    await service.write(principal,episodeId,"consents",{data:{...fixture.consent,consentId:"research-consent",permissions:{...fixture.consent.permissions,researchClips:true}}},"research-consent");
    await expect(service.write(principal,episodeId,"annotations",{data:{annotationId:"a",scope:fixture.episode.scope,encounterId:fixture.sessions[0]!.binding.encounterId,kind:"blinded-clip-annotation",code:"closure",value:1,sourceRef:{system:"research",resourceId:"clip",versionId:"1"},observedAt:now,clipRef:"arbitrary-clip"}},"annotation")).rejects.toMatchObject({code:"clip-access-denied"});
  });

  it("deduplicates FHIR versions across sync batches and persists corrected source provenance",async()=>{
    const {repository,fixture}=await setup();
    const service=new EncounterService(repository,{mode:"synthetic",now:()=>now,fhirSources:{ehr:{sourceSystem:"https://ehr.example/fhir",treatmentCodes:[{system:"urn:test",code:"toxin",product:"Synthetic toxin",kind:"botulinum-injection"}]}}});
    const resource={resourceType:"MedicationAdministration",id:"injection",meta:{versionId:"1",lastUpdated:"2026-08-01T00:00:00Z"},status:"completed",subject:{reference:fixture.episode.fhirPatientReference},medicationCodeableConcept:{coding:[{system:"urn:test",code:"toxin"}]},effectiveDateTime:"2026-08-01T00:00:00Z"};
    const first=await service.importFhir(principal,episodeId,{source:"ehr",bundle:resource},"sync-1");
    const retry=await service.importFhir(principal,episodeId,{source:"ehr",bundle:resource},"sync-2");
    expect(retry.duplicate).toBe(true);expect(retry.records[0]?.id).toBe(first.records[0]?.id);
    const changed=structuredClone(resource);changed.meta.versionId="2";changed.status="entered-in-error";
    const correction=await service.importFhir(principal,episodeId,{source:"ehr",bundle:changed},"sync-3");
    expect(correction.records[0]?.replacesRecordId).toBe(first.records[0]?.id);
    expect(correction.records[0]?.sourceProvenance?.versionId).toBe("2");
    expect((await service.importFhir(principal,episodeId,{source:"ehr",bundle:resource},"sync-old-retry")).duplicate).toBe(true);
    const forged=structuredClone(changed);forged.effectiveDateTime="2026-07-01T00:00:00Z";
    await expect(service.importFhir(principal,episodeId,{source:"ehr",bundle:forged},"same-version-mutation")).rejects.toMatchObject({code:"idempotency-conflict"});
  });
});
