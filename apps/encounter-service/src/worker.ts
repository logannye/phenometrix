import { randomUUID } from "node:crypto";
import {
  ClinicalContextRevisionV1Schema, DerivedDataConsentV1Schema, DurableObservationV1Schema,
  ParticipantBindingV1Schema, TreatmentResponseProtocolV1Schema, TreatmentResponseSpecificationV1Schema,
  TreatmentRevisionV1Schema, type TreatmentResponseRunV1
} from "@phenometrix/contracts";
import { analyzeTreatmentResponse,createTreatmentResponseSnapshot,TREATMENT_RESPONSE_ENGINE_VERSION } from "@phenometrix/trajectory-core";
import { effectiveRecords,type EpisodeState,type JsonObject,type Repository } from "./types.js";
import { digest } from "./canonical.js";

/** The clinical engine sees all revision chains; grants/bindings resolve to their latest immutable versions. */
export async function analyzeEpisode(state:EpisodeState,asOf:string):Promise<TreatmentResponseRunV1> {
  const current=effectiveRecords(state.records);
  const snapshot=await createTreatmentResponseSnapshot({
    scope:{tenantId:state.episode.tenantId,studyId:String(state.episode.metadata.studyId),participantId:state.episode.subjectRef},asOf,
    observations:state.records.filter(record=>record.kind==="session").map(record=>DurableObservationV1Schema.parse(record.payload)),
    consents:current.filter(record=>record.kind==="consent").map(record=>DerivedDataConsentV1Schema.parse(record.payload)),
    bindings:current.filter(record=>record.kind==="binding").map(record=>ParticipantBindingV1Schema.parse(record.payload)),
    treatments:state.records.filter(record=>record.kind==="clinical-event" && record.payload.schemaVersion==="phenometric.treatment-revision.v1").map(record=>TreatmentRevisionV1Schema.parse(record.payload)),
    contexts:state.records.filter(record=>record.kind==="clinical-event" && record.payload.schemaVersion==="phenometric.clinical-context-revision.v1").map(record=>ClinicalContextRevisionV1Schema.parse(record.payload))
  });
  return analyzeTreatmentResponse({snapshot,protocol:TreatmentResponseProtocolV1Schema.parse(state.episode.metadata.protocol),specification:TreatmentResponseSpecificationV1Schema.parse(state.episode.metadata.specification)});
}

export async function runOneJob(repository:Repository,options:{now?:()=>string;analyze?:(state:EpisodeState,asOf:string)=>Promise<JsonObject>}={}):Promise<"idle"|"published"|"superseded"|"failed"> {
  const now=options.now?.()??new Date().toISOString();
  const job=await repository.claimJob(now);
  if(!job)return "idle";
  try {
    const state=await repository.getState(job.tenantId,job.episodeId);
    if(!state || state.episode.revision!==job.inputRevision)return "superseded";
    // The explicit snapshot time is part of the hash and checks grant expiry at execution.
    const asOf=now;
    const analysis=await (options.analyze??analyzeEpisode)(state,asOf);
    const published=await repository.publish(job,{id:randomUUID(),episodeId:job.episodeId,inputRevision:job.inputRevision,
      algorithmVersion:TREATMENT_RESPONSE_ENGINE_VERSION,inputHash:digest({state,asOf}),analysis,createdAt:now});
    return published?"published":"superseded";
  } catch {
    // Error messages can contain source data; persist only a stable code.
    await repository.failJob(job,"analysis-failed");
    return "failed";
  }
}
