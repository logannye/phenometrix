import { randomUUID } from "node:crypto";
import {
  ClinicalContextRevisionV1Schema, DerivedDataConsentV1Schema, DurableObservationV1Schema,
  ParticipantBindingV1Schema, TreatmentResponseProtocolV1Schema, TreatmentResponseScopeV1Schema,
  TreatmentResponseSpecificationV1Schema, TreatmentResponseReviewV1Schema, TreatmentRevisionV1Schema, verifyTreatmentResponseArtifact,
  type DerivedDataConsentV1, type TreatmentResponseScopeV1
} from "@phenometrix/contracts";
import { z } from "zod";
import { authorize, type Permission, type Principal } from "./auth.js";
import { invariant } from "./errors.js";
import { importFhirR4, type FhirCodeMapping } from "./fhir.js";
import { digest } from "./canonical.js";
import { effectiveRecords, type AccessAudit,type AppendInput, type EpisodeState, type EvidenceSnapshot, type JsonObject, type RecordKind, type Repository } from "./types.js";

const Id = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const WriteEnvelope = z.object({ data: z.unknown(), expectedRevision: z.number().int().nonnegative().optional(), supersedesRecordId: z.string().uuid().nullable().optional() }).strict();
const CreateEpisode = z.object({
  id: Id, scope: TreatmentResponseScopeV1Schema, dataClass: z.enum(["synthetic","consented-research"]),
  protocol: TreatmentResponseProtocolV1Schema, specification: TreatmentResponseSpecificationV1Schema,
  fhirPatientReference: z.string().min(1).max(300).optional()
}).strict();
const ReviewInput = z.object({
  snapshotId: z.string().uuid(), inputRevision: z.number().int().nonnegative(),
  disposition: z.enum(["acknowledged","dismissed","correction-requested"]),
  supersedesReviewId: Id.nullable().default(null), reasonCode:z.string().min(1).max(300).nullable().default(null)
}).strict();
const Annotation = z.object({
  annotationId: Id, scope: TreatmentResponseScopeV1Schema, encounterId: Id,
  kind: z.enum(["routine-care-reference","blinded-clip-annotation"]), code: z.string().min(1).max(100),
  value: z.union([z.string().max(300),z.number().finite(),z.boolean(),z.null()]),
  sourceRef: z.object({ system: z.string().min(1),resourceId: z.string().min(1),versionId: z.string().min(1) }).strict(),
  observedAt: z.string().datetime({offset:true}), clipRef: Id.nullable().default(null)
}).strict();

export interface FhirSourceConfiguration { sourceSystem: string; treatmentCodes: FhirCodeMapping[]; dateOffset?: string }
export class EncounterService {
  constructor(readonly repository: Repository, private readonly options: {
    mode: "synthetic" | "live";
    now?: () => string;
    fhirSources?: Record<string,FhirSourceConfiguration>;
    allowedProtocolDigests?:readonly string[];
    authorizeClipAnnotation?:(input:{principal:Principal;scope:TreatmentResponseScopeV1;encounterId:string;clipRef:string})=>Promise<boolean>;
  }) {}
  private now() { return this.options.now?.() ?? new Date().toISOString(); }
  /** Deployment-owned import identity/terminology; contains no source credentials. */
  getFhirSourceConfiguration(name:string):FhirSourceConfiguration {
    const source=this.options.fhirSources?.[name];
    invariant(source,422,"fhir-source-not-configured","FHIR source and terminology mapping require server-side configuration.");
    return structuredClone(source);
  }

  private async state(principal: Principal, episodeId: string, permission: Permission): Promise<EpisodeState> {
    const state = await this.repository.getState(principal.tenantId,episodeId);
    invariant(state,404,"episode-not-found","Episode not found.");
    authorize(principal,permission,state.episode.subjectRef,state.episode.dataClass);
    invariant(principal.studyIds.includes(String(state.episode.metadata.studyId)),403,"study-out-of-scope","Study is outside the signed session scope.");
    return state;
  }
  private scope(state: EpisodeState): TreatmentResponseScopeV1 {
    return { tenantId: state.episode.tenantId,studyId: String(state.episode.metadata.studyId),participantId: state.episode.subjectRef };
  }
  private checkScope(state: EpisodeState, scope: TreatmentResponseScopeV1) {
    invariant(digest(this.scope(state)) === digest(scope),403,"scope-mismatch","Input scope does not match this episode.");
  }
  private audit(principal:Principal,state:EpisodeState,action:AccessAudit["action"],result:AccessAudit["result"],artifactId:string|null=null){
    return this.repository.appendAccessAudit({id:randomUUID(),tenantId:principal.tenantId,episodeId:state.episode.id,actorId:principal.sub,recordedAt:this.now(),action,result,inputRevision:state.episode.revision,artifactId});
  }
  private currentConsent(state: EpisodeState, consentId?: string): DerivedDataConsentV1 | undefined {
    const consents = effectiveRecords(state.records).filter(record => record.kind === "consent")
      .map(record => DerivedDataConsentV1Schema.parse(record.payload)).sort((a,b)=>Date.parse(b.grantedAt)-Date.parse(a.grantedAt));
    return consentId ? consents.find(consent => consent.consentId === consentId) : consents.find(consent => this.permits(consent,this.now(),this.now()));
  }
  private permits(consent: DerivedDataConsentV1, startedAt: string, endedAt: string): boolean {
    const now = Date.parse(this.now());
    return consent.permissions.capture && consent.permissions.derivedAnalysis && consent.permissions.derivedRetention &&
      Date.parse(consent.grantedAt) <= Date.parse(startedAt) &&
      (!consent.expiresAt || Date.parse(consent.expiresAt) > Math.max(Date.parse(endedAt),now)) &&
      (!consent.withdrawnAt || Date.parse(consent.withdrawnAt) > now);
  }

  async createEpisode(principal: Principal, raw: unknown) {
    const data = CreateEpisode.parse(raw);
    authorize(principal,"episode:create",data.scope.participantId,data.dataClass);
    invariant(data.scope.tenantId === principal.tenantId && principal.studyIds.includes(data.scope.studyId),403,"scope-mismatch","Episode is outside the signed session scope.");
    invariant(this.options.mode !== "synthetic" || data.dataClass === "synthetic",403,"synthetic-only","Development mode only accepts synthetic participants.");
    invariant(digest(data.specification.scope) === digest(data.scope),400,"specification-scope","Specification must belong to this participant and study.");
    invariant(await verifyTreatmentResponseArtifact(data.protocol) && await verifyTreatmentResponseArtifact(data.specification),422,"artifact-integrity","Protocol and specification content hashes must verify.");
    invariant(data.specification.protocolRef.contentSha256 === data.protocol.contentSha256 && data.specification.protocolRef.id===data.protocol.protocolId && data.specification.protocolRef.version===data.protocol.version,422,"protocol-mismatch","Specification must reference this exact protocol.");
    if(this.options.mode==="live")invariant(this.options.allowedProtocolDigests?.includes(data.protocol.contentSha256),403,"protocol-not-registered","Live analysis requires a server-registered protocol digest.");
    return this.repository.createEpisode({ id:data.id,tenantId:principal.tenantId,subjectRef:data.scope.participantId,dataClass:data.dataClass,createdAt:this.now(),
      metadata:{studyId:data.scope.studyId,protocol:data.protocol,specification:data.specification,...(data.fhirPatientReference ? {fhirPatientReference:data.fhirPatientReference} : {})} });
  }

  async getEpisode(principal: Principal, episodeId: string) {
    const state = await this.state(principal,episodeId,"evidence:read");
    return {episode:state.episode,inputRevision:state.episode.revision,sourceCounts:Object.fromEntries(["session","consent","binding","clinical-event","annotation"].map(kind => [kind,state.records.filter(record=>record.kind===kind).length]))};
  }

  async captureContext(principal:Principal,episodeId:string,encounterId:string) {
    const state=await this.state(principal,episodeId,"capture:write");
    const consent=this.currentConsent(state);
    if(!consent)await this.audit(principal,state,"capture-context-read","denied");
    invariant(consent,403,"consent-required","Current capture, analysis and derived-retention consent is required.");
    const binding=effectiveRecords(state.records).filter(record=>record.kind==="binding").map(record=>ParticipantBindingV1Schema.parse(record.payload))
      .find(item=>item.encounterId===encounterId && item.status==="verified" && !item.revokedAt);
    if(!binding)await this.audit(principal,state,"capture-context-read","denied");
    invariant(binding,403,"binding-required","A current verified binding for this encounter is required.");
    const protocol=TreatmentResponseProtocolV1Schema.parse(state.episode.metadata.protocol);
    await this.audit(principal,state,"capture-context-read","allowed",binding.bindingId);
    return {scope:this.scope(state),episodeId,inputRevision:state.episode.revision,consent,binding,measurementProtocolRef:protocol.measurementProtocolRef};
  }

  async write(principal: Principal, episodeId: string, route: "consents" | "bindings" | "sessions" | "clinical-events" | "annotations", raw: unknown, idempotencyKey: string) {
    const envelope = WriteEnvelope.parse(raw);
    const permission: Permission = route === "consents" ? "consent:write" : route === "sessions" || route === "bindings" ? "capture:write" : route === "annotations" ? "research:annotate" : "clinical:write";
    const state = await this.state(principal,episodeId,permission);
    let payload: JsonObject;
    let logicalId: string;
    let kind: RecordKind;
    let replacesRecordId = envelope.supersedesRecordId ?? null;
    if (route === "consents") {
      const consent = DerivedDataConsentV1Schema.parse(envelope.data);
      this.checkScope(state,consent.scope);
      invariant(Date.parse(consent.grantedAt)<=Date.parse(this.now()) && (!consent.expiresAt || Date.parse(consent.expiresAt)>Date.parse(consent.grantedAt)) && (!consent.withdrawnAt || (Date.parse(consent.withdrawnAt)>=Date.parse(consent.grantedAt) && Date.parse(consent.withdrawnAt)<=Date.parse(this.now()))),422,"consent-chronology","Consent chronology is invalid.");
      const current = effectiveRecords(state.records).find(record=>record.kind==="consent" && record.logicalId===consent.consentId);
      if (current) {
        const prior = DerivedDataConsentV1Schema.parse(current.payload);
        invariant(!prior.withdrawnAt || prior.withdrawnAt === consent.withdrawnAt,409,"consent-withdrawal-immutable","A withdrawn consent cannot be revived; record a new consent.");
        invariant(digest({...prior,withdrawnAt:null})===digest({...consent,withdrawnAt:null}),409,"consent-grant-immutable","Consent permissions and grant are immutable; record a new consent for a new grant.");
      }
      payload=consent; logicalId=consent.consentId; kind="consent";
    } else if (route === "bindings") {
      const binding=ParticipantBindingV1Schema.parse(envelope.data); this.checkScope(state,binding.scope);
      invariant(Date.parse(binding.recordedAt)<=Date.parse(this.now()),422,"binding-chronology","Binding cannot be recorded in the future.");
      invariant(binding.status!=="verified"||binding.verificationMethod!=="unverified",422,"binding-verification","Verified bindings require a verification method.");
      invariant(binding.status==="revoked"?binding.revokedAt&&Date.parse(binding.revokedAt)>=Date.parse(binding.recordedAt)&&Date.parse(binding.revokedAt)<=Date.parse(this.now()):binding.revokedAt===null,422,"binding-revocation","Binding revocation status and timestamp must agree.");
      const existing=effectiveRecords(state.records).find(record=>record.kind==="binding"&&record.logicalId===binding.bindingId);
      if(existing){
        const prior=ParticipantBindingV1Schema.parse(existing.payload);
        invariant(digest({...prior,status:null,revokedAt:null})===digest({...binding,status:null,revokedAt:null}),409,"binding-identity-immutable","Binding identity cannot change; revoke it and create a new binding.");
        invariant(prior.status===binding.status||binding.status==="revoked",409,"binding-transition","A binding can only be revoked; use a new ID for new verification.");
        invariant(!prior.revokedAt||prior.revokedAt===binding.revokedAt,409,"binding-revocation-immutable","A revoked binding cannot be revived.");
      }
      if (principal.role === "patient") invariant(binding.verificationMethod==="authenticated-patient-portal" && binding.platformParticipantId===principal.sub,403,"binding-verification","Patient bindings must derive from the authenticated portal identity.");
      if (this.options.mode === "live") invariant(binding.verificationMethod!=="synthetic-fixture",403,"synthetic-binding","Synthetic bindings are forbidden in live mode.");
      payload=binding;logicalId=binding.bindingId;kind="binding";
    } else if (route === "sessions") {
      const observation=DurableObservationV1Schema.parse(envelope.data); this.checkScope(state,observation.scope);
      invariant(Date.parse(observation.endedAt)<=Date.parse(this.now())+1000,422,"observation-future","Capture cannot end in the future.");
      invariant(Date.parse(observation.recordedAt)<=Date.parse(this.now()),422,"recorded-at-future","Observation knowledge time cannot be in the future.");
      const consent=this.currentConsent(state,observation.consentId);
      invariant(consent && this.permits(consent,observation.startedAt,observation.endedAt),403,"consent-required","Active capture, analysis and derived-retention consent is required.");
      invariant(observation.metrics.every(metric=>consent.permissions.modalities.includes(metric.modality)),403,"modality-not-consented","A captured modality is outside consent.");
      const bindingRecord=effectiveRecords(state.records).find(record=>record.kind==="binding" && record.logicalId===observation.bindingId);
      const binding=bindingRecord ? ParticipantBindingV1Schema.parse(bindingRecord.payload) : null;
      invariant(binding?.status==="verified" && binding.verificationMethod!=="unverified" && binding.encounterId===observation.encounterId && !binding.revokedAt && Date.parse(binding.recordedAt)<=Date.parse(observation.startedAt),403,"binding-required","A verified current participant-to-encounter binding is required.");
      invariant(this.options.mode!=="synthetic" || observation.capture.sourceKind==="synthetic",403,"synthetic-only","Development mode rejects non-synthetic observations.");
      invariant(this.options.mode!=="live" || observation.capture.sourceKind!=="synthetic",403,"live-source-required","Live mode rejects synthetic observations.");
      const protocol=TreatmentResponseProtocolV1Schema.parse(state.episode.metadata.protocol);
      invariant(protocol.metrics.every(metric=>!/(prompt|task|exercise)/i.test(metric.context)) && observation.metrics.every(metric=>!/(prompt|task|exercise)/i.test(metric.context)),422,"ambient-only","This pilot accepts natural observation only.");
      const current=effectiveRecords(state.records).find(record=>record.kind==="session" && record.logicalId===observation.observationId);
      const knownRevision=state.records.find(record=>record.kind==="session"&&record.payload.revisionId===observation.revisionId);
      if(knownRevision){
        invariant(knownRevision.logicalId===observation.observationId&&knownRevision.idempotencyKey===idempotencyKey,409,"revision-identity-conflict","A revision ID cannot be reused in another input.");
        invariant(envelope.supersedesRecordId===undefined||envelope.supersedesRecordId===knownRevision.replacesRecordId,409,"idempotency-conflict","A retry cannot change its predecessor.");
        replacesRecordId=knownRevision.replacesRecordId;
      }
      invariant(!state.records.some(record=>record.kind==="session" && record.logicalId!==observation.observationId && record.payload.revisionId===observation.revisionId),409,"revision-identity-conflict","Observation revision identity is already assigned to another observation.");
      invariant(current || !observation.supersedesRevisionId,409,"correction-parent-missing","A first observation cannot supersede an absent revision.");
      if (current && current.payload.revisionId!==observation.revisionId && !knownRevision) {
        invariant(current.payload.revisionId===observation.supersedesRevisionId,409,"correction-chain","Observation correction must supersede its current revision.");
        invariant(Date.parse(observation.recordedAt)>=Date.parse(String(current.payload.recordedAt)),409,"correction-chronology","A correction cannot precede knowledge of its prior revision.");
        replacesRecordId=current.id;
      }
      payload=observation;logicalId=observation.observationId;kind="session";
    } else if (route === "clinical-events") {
      invariant(principal.role==="clinician" || principal.role==="integration",403,"clinical-authority","Clinical records require a clinician or authorized integration.");
      const clinical=z.union([TreatmentRevisionV1Schema,ClinicalContextRevisionV1Schema]).parse(envelope.data);
      invariant(Date.parse(clinical.recordedAt)<=Date.parse(this.now()),422,"recorded-at-future","Clinical knowledge time cannot be in the future.");
      this.checkScope(state,clinical.scope);
      const id="treatmentId" in clinical ? clinical.treatmentId : clinical.contextId;
      const current=effectiveRecords(state.records).find(record=>record.kind==="clinical-event" && record.logicalId===id);
      invariant(!current || current.payload.schemaVersion===clinical.schemaVersion,409,"clinical-record-class-change","A correction cannot switch clinical record class; explicit reconciliation is required.");
      const knownRevision=state.records.find(record=>record.kind==="clinical-event"&&record.payload.revisionId===clinical.revisionId);
      if(knownRevision){
        invariant(knownRevision.logicalId===id&&knownRevision.idempotencyKey===idempotencyKey,409,"revision-identity-conflict","A revision ID cannot be reused in another input.");
        invariant(envelope.supersedesRecordId===undefined||envelope.supersedesRecordId===knownRevision.replacesRecordId,409,"idempotency-conflict","A retry cannot change its predecessor.");
        replacesRecordId=knownRevision.replacesRecordId;
      }
      invariant(!state.records.some(record=>record.kind==="clinical-event" && record.logicalId!==id && record.payload.revisionId===clinical.revisionId),409,"revision-identity-conflict","Clinical revision identity is already assigned to another record.");
      invariant(current || !clinical.supersedesRevisionId,409,"correction-parent-missing","A first clinical record cannot supersede an absent revision.");
      if (current && current.payload.revisionId!==clinical.revisionId && !knownRevision) {
        invariant(current.payload.revisionId===clinical.supersedesRevisionId,409,"correction-chain","Clinical correction must supersede its current revision.");
        invariant(Date.parse(clinical.recordedAt)>=Date.parse(String(current.payload.recordedAt)),409,"correction-chronology","A correction cannot precede knowledge of its prior revision.");
        replacesRecordId=current.id;
      }
      payload=clinical;logicalId=id;kind="clinical-event";
    } else {
      const annotation=Annotation.parse(envelope.data);this.checkScope(state,annotation.scope);
      if(annotation.kind==="routine-care-reference") invariant(principal.role==="clinician" || principal.role==="integration",403,"clinical-authority","Routine-care outcomes require clinical authority.");
      if(annotation.kind==="blinded-clip-annotation") {
        const researchConsent=effectiveRecords(state.records).filter(record=>record.kind==="consent").map(record=>DerivedDataConsentV1Schema.parse(record.payload))
          .some(consent=>consent.permissions.researchClips&&this.permits(consent,this.now(),this.now()));
        invariant(researchConsent && annotation.clipRef,403,"clip-consent-required","Clip annotations require separate research-clip consent and a clip reference.");
        invariant(this.options.authorizeClipAnnotation && await this.options.authorizeClipAnnotation({principal,scope:this.scope(state),encounterId:annotation.encounterId,clipRef:annotation.clipRef}),403,"clip-access-denied","A configured clip authority must verify this clip's scope, expiry and annotation permission.");
      }
      payload=annotation;logicalId=annotation.annotationId;kind="annotation";
    }
    return this.repository.append({tenantId:principal.tenantId,episodeId,kind,logicalId,replacesRecordId,
      expectedRevision:envelope.expectedRevision ?? state.episode.revision,idempotencyKey,payload,actorId:principal.sub,recordedAt:this.now()});
  }

  async importFhir(principal: Principal,episodeId:string,raw:unknown,idempotencyKey:string) {
    invariant(principal.role==="clinician" || principal.role==="integration",403,"clinical-authority","FHIR import requires clinical authority.");
    const data=z.object({source:Id,bundle:z.unknown(),expectedRevision:z.number().int().nonnegative().optional()}).strict().parse(raw);
    const state=await this.state(principal,episodeId,"clinical:write");
    const source=this.getFhirSourceConfiguration(data.source);
    invariant(typeof state.episode.metadata.fhirPatientReference==="string",422,"fhir-binding-required","A clinician-enrolled FHIR patient reference is required.");
    const current=effectiveRecords(state.records).filter(record=>record.kind==="clinical-event");
    const specification=TreatmentResponseSpecificationV1Schema.parse(state.episode.metadata.specification);
    const imported=importFhirR4(data.bundle,{scope:this.scope(state),sourceSystem:source.sourceSystem,
      patientReference:state.episode.metadata.fhirPatientReference,courseId:episodeId,receivedAt:this.now(),treatmentCodes:source.treatmentCodes,dateOffset:source.dateOffset,
      previousRevisions:new Map(current.map(record=>[record.logicalId,String(record.payload.revisionId)])),
      currentSourceVersions:new Map(current.filter(record=>record.sourceProvenance).map(record=>[record.logicalId,{versionId:String(record.sourceProvenance!.versionId),lastUpdated:record.sourceProvenance!.lastUpdated as string|null}])),
      knownRevisions:new Map(state.records.filter(record=>record.kind==="clinical-event").map(record=>[String(record.payload.revisionId),{recordedAt:String(record.payload.recordedAt),supersedesRevisionId:record.payload.supersedesRevisionId as string|null}]))});
    const records=[...imported.treatments,...imported.contexts];
    for(const excluded of imported.excluded){
      invariant(!current.some(record=>(record.payload.sourceRef as JsonObject)?.system===source.sourceSystem&&(record.payload.sourceRef as JsonObject)?.resourceId===excluded.resourceId),409,"fhir-existing-source-unresolved","A previously used clinical source is now unmapped or incomplete; reconcile it explicitly before analysis.");
    }
    if(!records.length) return {records:[],duplicate:false,excluded:imported.excluded};
    const inputs:AppendInput[]=records.map(record=>{
      const logicalId="treatmentId" in record ? record.treatmentId : record.contextId;
      const prior=current.find(item=>item.logicalId===logicalId);
      invariant(!prior || prior.payload.schemaVersion===record.schemaVersion,409,"clinical-record-class-change","A source correction changed clinical record class; explicit reconciliation is required.");
      const sameVersion=state.records.find(item=>item.kind==="clinical-event" && item.payload.revisionId===record.revisionId);
      return {tenantId:principal.tenantId,episodeId,kind:"clinical-event",logicalId,replacesRecordId:prior?.id??null,
        expectedRevision:data.expectedRevision??state.episode.revision,idempotencyKey:`fhir:${record.revisionId}`,
        ...(sameVersion?{replacesRecordId:sameVersion.replacesRecordId}:{}),
        payload:record,sourceProvenance:imported.sources.find(source=>source.revisionId===record.revisionId)??null,actorId:principal.sub,recordedAt:this.now()};
    });
    const result=await this.repository.appendBatch(inputs);
    return {...result,excluded:imported.excluded,anchorTreatmentId:specification.anchorTreatmentId};
  }

  async evidence(principal:Principal,episodeId:string,history=false) {
    const evidence=await this.repository.getEvidenceState(principal.tenantId,episodeId);
    invariant(evidence,404,"episode-not-found","Episode not found.");
    const {state}=evidence;
    authorize(principal,"evidence:read",state.episode.subjectRef,state.episode.dataClass);
    invariant(principal.studyIds.includes(String(state.episode.metadata.studyId)),403,"study-out-of-scope","Study is outside the signed session scope.");
    const hasCapturedData=state.records.some(record=>record.kind==="session");
    if(hasCapturedData&&!this.currentConsent(state))await this.audit(principal,state,history?"evidence-history-read":"evidence-read","denied");
    invariant(!hasCapturedData || this.currentConsent(state),403,"analysis-consent-inactive","Evidence access requires current analysis and derived-retention consent.");
    const accessible=evidence.snapshots.filter(snapshot=>this.snapshotAccessible(state,snapshot));
    const snapshot=accessible.find(item=>item.inputRevision===state.episode.revision)??null;
    const jobs=evidence.jobs.map(({id,inputRevision,status,errorCode})=>({id,inputRevision,status,errorCode}));
    const currentJob=jobs.find(item=>item.inputRevision===state.episode.revision);
    await this.audit(principal,state,history?"evidence-history-read":"evidence-read","allowed",snapshot?.id??null);
    return {episode:state.episode,inputRevision:state.episode.revision,
      status:snapshot?"ready":currentJob?.status==="failed"?"failed":currentJob?"pending":"empty",
      snapshot,reviews:evidence.reviews.filter(review=>accessible.some(snapshot=>snapshot.id===review.snapshotId)).map(review=>review.payload),jobs,
      ...(history?{snapshots:accessible,annotations:effectiveRecords(state.records).filter(record=>record.kind==="annotation" && record.payload.kind==="routine-care-reference")}:{})};
  }

  private snapshotAccessible(state:EpisodeState,snapshot:EvidenceSnapshot):boolean {
    const rows=snapshot.analysis.rows;
    if(!Array.isArray(rows))return false;
    const usedIds=new Set<string>();
    for(const row of rows){
      for(const point of row.points??[])usedIds.add(point.revisionId);
      for(const id of row.baseline?.observationRevisionIds??[])usedIds.add(id);
    }
    for(const id of usedIds){
      const record=state.records.find(item=>item.kind==="session" && item.payload.revisionId===id);
      if(!record)return false;
      const observation=DurableObservationV1Schema.parse(record.payload);
      const consent=this.currentConsent(state,observation.consentId);
      if(!consent || !this.permits(consent,observation.startedAt,observation.endedAt))return false;
      const binding=effectiveRecords(state.records).find(item=>item.kind==="binding"&&item.logicalId===observation.bindingId);
      if(!binding||binding.payload.status!=="verified"||binding.payload.revokedAt||binding.payload.encounterId!==observation.encounterId)return false;
    }
    return true;
  }

  async review(principal:Principal,episodeId:string,raw:unknown,idempotencyKey:string) {
    invariant(principal.role==="clinician",403,"clinician-review-required","Evidence review requires a clinician session.");
    const state=await this.state(principal,episodeId,"evidence:review");
    const data=ReviewInput.parse(raw);
    const evidence=await this.repository.getEvidence(principal.tenantId,episodeId);
    const snapshot=evidence.snapshots.find(item=>item.id===data.snapshotId && item.inputRevision===data.inputRevision);
    invariant(snapshot,404,"evidence-not-found","Evidence snapshot not found.");
    invariant(this.snapshotAccessible(state,snapshot),403,"analysis-consent-inactive","This evidence's source consent is not active.");
    if(data.supersedesReviewId) invariant(evidence.reviews.some(review=>review.payload.reviewId===data.supersedesReviewId),409,"review-predecessor-missing","Review correction must reference an existing review in this episode.");
    const id=randomUUID(),recordedAt=this.now();
    const payload=TreatmentResponseReviewV1Schema.parse({schemaVersion:"phenometric.treatment-response-review.v1",reviewId:id,scope:this.scope(state),
      runId:snapshot.analysis.runId,runSha256:snapshot.analysis.contentSha256,actorId:principal.sub,recordedAt,
      disposition:data.disposition,supersedesReviewId:data.supersedesReviewId,reasonCode:data.reasonCode});
    const saved=await this.repository.addReview(principal.tenantId,{id,episodeId,snapshotId:data.snapshotId,inputRevision:data.inputRevision,payload,recordedAt},idempotencyKey);
    return saved.payload;
  }
}
