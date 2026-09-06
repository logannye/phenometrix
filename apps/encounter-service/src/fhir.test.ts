import { describe,it,expect } from "vitest";
import { importFhirR4,parseFhirTime } from "./fhir.js";
import { SYNTHETIC_SCOPE } from "./synthetic.js";
const options={scope:SYNTHETIC_SCOPE,sourceSystem:"https://ehr.example/fhir",patientReference:"Patient/one",courseId:"course",receivedAt:"2026-09-06T00:00:00Z",treatmentCodes:[{system:"urn:synthetic-code",code:"botox",product:"Synthetic product",kind:"botulinum-injection" as const}]};
function medication(type="MedicationAdministration",status="completed"){
  return {resourceType:type,id:"one",meta:{versionId:"1",lastUpdated:"2026-08-01T00:00:00Z"},subject:{reference:"Patient/one"},status,effectiveDateTime:"2026-08-01T10:30:00-07:00",medicationCodeableConcept:{coding:[{system:"urn:synthetic-code",code:"botox"}]},dosage:{dose:{value:25,unit:"product-specific units"}}};
}
describe("FHIR R4 clinical event semantics",()=>{
  it("imports a completed administration without converting product-specific units",()=>{
    const result=importFhirR4(medication(),options);expect(result.treatments[0]).toMatchObject({status:"administered",verification:"verified",dose:{value:25,unit:"product-specific units"},sourceRef:{resourceId:"MedicationAdministration/one",versionId:"1"}});
    expect(result.treatments[0]?.effectiveTime.precision).toBe("instant");
  });
  it("does not promote orders, reports or incomplete administrations to verified injections",()=>{
    expect(importFhirR4(medication("MedicationRequest","active"),options).treatments[0]).toMatchObject({status:"planned",verification:"unverified",effectiveTime:{precision:"unknown"}});
    const report=importFhirR4(medication("MedicationStatement","active"),options);expect(report.treatments).toHaveLength(0);expect(report.contexts[0]?.valueCode).toContain("reported-medication");
    for(const status of ["not-done","unknown","in-progress","on-hold","stopped"])expect(importFhirR4(medication("MedicationAdministration",status),options).treatments).toHaveLength(0);
  });
  it("preserves partial/missing date uncertainty and only bounds known local dates",()=>{
    expect(parseFhirTime("2026").precision).toBe("unknown");expect(parseFhirTime("2026-08").precision).toBe("unknown");
    expect(parseFhirTime("2026-08-01").precision).toBe("unknown");
    expect(parseFhirTime("2026-08-01","-07:00")).toEqual({earliest:"2026-08-01T00:00:00-07:00",latest:"2026-08-01T23:59:59.999-07:00",precision:"day"});
    expect(parseFhirTime("2026-02-30","-07:00").precision).toBe("unknown");
    expect(parseFhirTime("2026-08-01T10:00:00").precision).toBe("unknown");
  });
  it("preserves source corrections and entered-in-error retractions",()=>{
    const original=importFhirR4(medication(),options).treatments[0]!;
    const resource=medication("MedicationAdministration","entered-in-error");resource.meta.versionId="2";
    const corrected=importFhirR4(resource,{...options,previousRevisions:new Map([[original.treatmentId,original.revisionId]])}).treatments[0]!;
    expect(corrected.treatmentId).toBe(original.treatmentId);expect(corrected.revisionId).not.toBe(original.revisionId);expect(corrected.supersedesRevisionId).toBe(original.revisionId);expect(corrected.status).toBe("entered-in-error");
  });
  it("uses trusted receipt time while preserving remote version ordering",()=>{
    const resource=medication();resource.meta.lastUpdated="2030-01-01T00:00:00Z";
    const result=importFhirR4(resource,options);expect(result.treatments[0]?.recordedAt).toBe(options.receivedAt);expect(result.sources[0]?.lastUpdated).toBe(resource.meta.lastUpdated);
    const original=result.treatments[0]!;
    const stale={...resource,meta:{...resource.meta,versionId:"0"}};
    expect(()=>importFhirR4(stale,{...options,previousRevisions:new Map([[original.treatmentId,original.revisionId]]),currentSourceVersions:new Map([[original.treatmentId,{versionId:"1",lastUpdated:resource.meta.lastUpdated}]])})).toThrow("older or ambiguously");
  });
  it("fails closed for wrong patient, unknown modifiers and ambiguous product names",()=>{
    expect(()=>importFhirR4({...medication(),subject:{reference:"Patient/two"}},options)).toThrow("enrolled patient");
    expect(()=>importFhirR4({...medication(),modifierExtension:[{url:"urn:unknown"}]},options)).toThrow("modifier");
    const result=importFhirR4({...medication(),medicationCodeableConcept:{text:"Botox injection done"}},options);
    expect(result.treatments).toHaveLength(0);expect(result.excluded[0]?.reason).toBe("unmapped-treatment-code");
  });
});
