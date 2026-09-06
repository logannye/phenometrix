import { createHfsTreatmentResponseProtocol,createHfsTreatmentResponseSpecification } from "@phenometrix/condition-profiles";
import type { DerivedDataConsentV1,DurableObservationV1,ParticipantBindingV1,TreatmentRevisionV1 } from "@phenometrix/contracts";
import type { Principal } from "./auth.js";
import type { EncounterService } from "./service.js";

export const SYNTHETIC_SCOPE={tenantId:"synthetic-tenant",studyId:"synthetic-hfs-study",participantId:"synthetic-participant"};
export const SYNTHETIC_EPISODE_ID="synthetic-hfs-episode";
export const SYNTHETIC_PRINCIPAL:Principal={sub:"synthetic-clinician",tenantId:SYNTHETIC_SCOPE.tenantId,role:"clinician",subjectRefs:[SYNTHETIC_SCOPE.participantId],studyIds:[SYNTHETIC_SCOPE.studyId],dataClass:"synthetic",
  permissions:["episode:create","capture:write","consent:write","clinical:write","evidence:read","evidence:review","research:annotate"]};

/** All values and patients here are fabricated software fixtures, not clinical evidence. */
export async function syntheticFixture(now="2026-09-06T12:00:00.000Z"){
  const protocol=await createHfsTreatmentResponseProtocol();
  const specification=await createHfsTreatmentResponseSpecification({scope:SYNTHETIC_SCOPE,anchorTreatmentId:"synthetic-treatment"});
  const injection=new Date(Date.parse(now)-28*86_400_000).toISOString();
  const consent:DerivedDataConsentV1={consentId:"synthetic-consent",scope:SYNTHETIC_SCOPE,documentRef:{id:"synthetic-consent-document",version:"1",contentSha256:"0".repeat(64)},
    grantedAt:new Date(Date.parse(injection)-30*86_400_000).toISOString(),expiresAt:null,withdrawnAt:null,
    permissions:{capture:true,derivedAnalysis:true,derivedRetention:true,researchClips:false,modalities:["face"]}};
  const treatment:TreatmentRevisionV1={schemaVersion:"phenometric.treatment-revision.v1",treatmentId:"synthetic-treatment",revisionId:"synthetic-treatment-v1",supersedesRevisionId:null,scope:SYNTHETIC_SCOPE,
    courseId:SYNTHETIC_EPISODE_ID,cycleId:"synthetic-cycle",recordedAt:injection,effectiveTime:{earliest:injection,latest:injection,precision:"instant"},status:"administered",verification:"verified",kind:"botulinum-injection",product:"Synthetic test product",dose:null,route:null,sites:[],sourceRef:{system:"synthetic-fixture",resourceId:"treatment-1",versionId:"1"}};
  const sessions=[-7,-3,7,21].map((day,index)=>{
    const startedAt=new Date(Date.parse(injection)+day*86_400_000).toISOString();
    const endedAt=new Date(Date.parse(startedAt)+60_000).toISOString();
    const encounterId=`synthetic-encounter-${index+1}`;
    const binding:ParticipantBindingV1={bindingId:`synthetic-binding-${index+1}`,scope:SYNTHETIC_SCOPE,encounterId,platformParticipantId:"synthetic-platform-participant",status:"verified",verificationMethod:"synthetic-fixture",recordedAt:consent.grantedAt,revokedAt:null};
    const observation:DurableObservationV1={schemaVersion:"phenometric.durable-observation.v1",observationId:`synthetic-observation-${index+1}`,revisionId:`synthetic-observation-${index+1}-v1`,supersedesRevisionId:null,
      scope:SYNTHETIC_SCOPE,encounterId,consentId:consent.consentId,bindingId:binding.bindingId,recordedAt:endedAt,startedAt,endedAt,status:"available",measurementProtocolRef:protocol.measurementProtocolRef,
      capture:{adapterId:"synthetic-adapter",adapterVersion:"1.0.0",sourceKind:"synthetic",pipelineVersion:"synthetic-pipeline-1",processorFingerprint:"a".repeat(64),deviceClass:"synthetic-device",clockUncertaintyMs:0,rawMediaRetained:false},
      windows:[{windowId:`synthetic-window-${index+1}`,startMs:0,endMs:60_000,status:"eligible",reasonCodes:[]}],
      metrics:protocol.metrics.map((metric,metricIndex)=>({metricCode:metric.metricCode,modality:"face",unit:metric.unit,context:metric.context,algorithmVersion:metric.algorithmVersion,processorRef:"synthetic-processor",status:"measured",value:0.25+index*0.005+metricIndex*0.01,reasonCodes:[],usableDurationMs:60_000,technicalQualityScore:0.95,sourceWindowIds:[`synthetic-window-${index+1}`],temporalSamples:[]}))};
    return {binding,observation};
  });
  return {episode:{id:SYNTHETIC_EPISODE_ID,scope:SYNTHETIC_SCOPE,dataClass:"synthetic" as const,protocol,specification,fhirPatientReference:"Patient/synthetic-patient"},consent,treatment,sessions};
}

export async function seedSyntheticEpisode(service:EncounterService,now=new Date().toISOString()){
  const existing=await service.repository.getState(SYNTHETIC_SCOPE.tenantId,SYNTHETIC_EPISODE_ID);
  if(existing)return;
  const fixture=await syntheticFixture(now);
  await service.createEpisode(SYNTHETIC_PRINCIPAL,fixture.episode);
  await service.write(SYNTHETIC_PRINCIPAL,SYNTHETIC_EPISODE_ID,"consents",{data:fixture.consent},"synthetic-consent");
  await service.write(SYNTHETIC_PRINCIPAL,SYNTHETIC_EPISODE_ID,"clinical-events",{data:fixture.treatment},"synthetic-treatment");
  for(const {binding,observation} of fixture.sessions){
    await service.write(SYNTHETIC_PRINCIPAL,SYNTHETIC_EPISODE_ID,"bindings",{data:binding},binding.bindingId);
    await service.write(SYNTHETIC_PRINCIPAL,SYNTHETIC_EPISODE_ID,"sessions",{data:observation},observation.revisionId);
  }
}
