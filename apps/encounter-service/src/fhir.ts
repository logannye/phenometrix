import {
  ClinicalContextRevisionV1Schema, TreatmentRevisionV1Schema,
  type ClinicalContextRevisionV1, type ClinicalEffectiveTimeV1,
  type TreatmentResponseScopeV1, type TreatmentRevisionV1
} from "@phenometrix/contracts";
import { z } from "zod";
import { digest } from "./canonical.js";
import { invariant } from "./errors.js";

type Resource = Record<string, any>;
export interface FhirCodeMapping { system: string; code: string; product: string; kind: "botulinum-injection" | "concurrent-treatment" }
export interface FhirImportOptions {
  scope: TreatmentResponseScopeV1;
  sourceSystem: string;
  patientReference: string;
  courseId: string;
  cycleId?: string | null;
  receivedAt: string;
  /** Institution-reviewed terminology mapping, never inferred from free text. */
  treatmentCodes: FhirCodeMapping[];
  previousRevisions?: Map<string, string>;
  knownRevisions?: Map<string,{recordedAt:string;supersedesRevisionId:string|null}>;
  currentSourceVersions?:Map<string,{versionId:string;lastUpdated:string|null}>;
  /** Required to turn a date without a zone into bounded times. E.g. -07:00. */
  dateOffset?: string;
}
export interface FhirImportResult {
  treatments: TreatmentRevisionV1[];
  contexts: ClinicalContextRevisionV1[];
  excluded: Array<{ resourceId: string; reason: string }>;
  sources:Array<{revisionId:string;versionId:string;lastUpdated:string|null;contentSha256:string;originalEffectiveDateTime:string|null}>;
}

const Timestamp = z.string().datetime({ offset: true });
const unknownTime = (): ClinicalEffectiveTimeV1 => ({ earliest: null, latest: null, precision: "unknown" });
export function parseFhirTime(value: unknown, dateOffset?: string): ClinicalEffectiveTimeV1 {
  if (typeof value !== "string") return unknownTime();
  if (Timestamp.safeParse(value).success) return { earliest: value, latest: value, precision: "instant" };
  if (/^\d{4}-\d{2}-\d{2}$/.test(value) && dateOffset && /^[+-](?:0\d|1[0-4]):[0-5]\d$/.test(dateOffset)) {
    const start = `${value}T00:00:00${dateOffset}`;
    const end = `${value}T23:59:59.999${dateOffset}`;
    if (Timestamp.safeParse(start).success && Timestamp.safeParse(end).success) return { earliest: start, latest: end, precision: "day" };
  }
  // Partial dates, missing zones and intervals do not silently become exact administrations.
  return unknownTime();
}

function codeLabel(concept: Resource | undefined): string | null {
  const coding = concept?.coding?.find((item: Resource) => typeof item.system === "string" && typeof item.code === "string");
  return coding ? `${coding.system}|${coding.code}` : null;
}

/** Bounded FHIR R4 normalizer. No network requests, narrative execution or text-derived clinical facts. */
export function importFhirR4(input: unknown, options: FhirImportOptions): FhirImportResult {
  const container = z.record(z.string(), z.unknown()).parse(input) as Resource;
  const resources: Resource[] = container.resourceType === "Bundle"
    ? z.array(z.object({ resource: z.record(z.string(),z.unknown()) }).passthrough()).max(1000).parse(container.entry ?? []).map(item => item.resource)
    : [container];
  const result: FhirImportResult = { treatments: [], contexts: [], excluded: [],sources:[] };
  const medications = new Map(resources.filter(item => item.resourceType === "Medication").map(item => [`Medication/${item.id}`, item]));
  for (const resource of resources) {
    const type = resource.resourceType;
    const resourceId = `${type}/${resource.id ?? "missing-id"}`;
    if (!["MedicationAdministration","MedicationRequest","MedicationStatement","Procedure"].includes(type)) {
      result.excluded.push({ resourceId, reason: "unsupported-resource-type" }); continue;
    }
    invariant(typeof resource.id === "string" && /^[A-Za-z0-9.-]{1,64}$/.test(resource.id),400,"fhir-resource-id","FHIR resource identity is required.");
    invariant(resource.subject?.reference === options.patientReference,403,"fhir-patient-mismatch","FHIR resource does not match the enrolled patient reference.");
    invariant(!resource.modifierExtension?.length,422,"fhir-unsupported-modifier","Unrecognized modifier extensions require explicit integration support.");
    const versionId = typeof resource.meta?.versionId === "string" ? resource.meta.versionId : `content-sha256:${digest(resource)}`;
    const sourceRef = { system: options.sourceSystem, resourceId, versionId };
    const stableId = `fhir:${digest([options.sourceSystem,resourceId]).slice(0,40)}`;
    const revisionId = `fhir-rev:${digest(sourceRef).slice(0,40)}`;
    const known=options.knownRevisions?.get(revisionId);
    const lastUpdated=Timestamp.safeParse(resource.meta?.lastUpdated).success?resource.meta.lastUpdated:null;
    const previousSource=options.currentSourceVersions?.get(stableId);
    if(!known && previousSource){
      const numeric=/^\d+$/.test(versionId)&&/^\d+$/.test(previousSource.versionId);
      const increasing=numeric ? BigInt(versionId)>BigInt(previousSource.versionId) : lastUpdated&&previousSource.lastUpdated&&Date.parse(lastUpdated)>Date.parse(previousSource.lastUpdated);
      invariant(increasing,409,"fhir-source-order","An older or ambiguously ordered source version requires explicit reconciliation.");
    }
    const supersedesRevisionId = known ? known.supersedesRevisionId : options.previousRevisions?.get(stableId) ?? null;
    const recordedAt = known?.recordedAt ?? options.receivedAt;
    const concept = type === "Procedure" ? resource.code : resource.medicationCodeableConcept ?? medications.get(resource.medicationReference?.reference)?.code;
    const mapping = options.treatmentCodes.find(candidate => concept?.coding?.some((code: Resource) => code.system === candidate.system && code.code === candidate.code));
    const isActualResource = type === "MedicationAdministration" || type === "Procedure";
    const isCompleted = isActualResource && resource.status === "completed";
    const isError = resource.status === "entered-in-error";
    const effective = type === "Procedure" ? resource.performedDateTime : resource.effectiveDateTime;
    result.sources.push({revisionId,versionId,lastUpdated,contentSha256:digest(resource),originalEffectiveDateTime:typeof effective==="string"?effective:null});
    if (mapping && (isCompleted || isError || type === "MedicationRequest")) {
      // An order never becomes administered, and authoredOn never becomes an administration date.
      const status = isError ? "entered-in-error" : isCompleted ? "administered" : "planned";
      const quantity = resource.dosage?.dose;
      const dose = quantity && !quantity.comparator && typeof quantity.value === "number" && quantity.value > 0 && typeof (quantity.unit ?? quantity.code) === "string"
        ? { value: quantity.value, unit: quantity.unit ?? quantity.code } : null;
      const site = codeLabel(resource.dosage?.site);
      result.treatments.push(TreatmentRevisionV1Schema.parse({
        schemaVersion: "phenometric.treatment-revision.v1", treatmentId: stableId, revisionId, supersedesRevisionId,
        scope: options.scope, courseId: options.courseId, cycleId: options.cycleId ?? null, recordedAt,
        effectiveTime: status === "planned" ? unknownTime() : parseFhirTime(effective,options.dateOffset),
        status, verification: isActualResource ? "verified" : "unverified", kind: mapping.kind,
        product: mapping.product, dose, route: codeLabel(resource.dosage?.route),
        sites: site ? [{ site, laterality: "unspecified", dose: null }] : [], sourceRef
      }));
    } else if (type === "MedicationStatement" || (type === "Procedure" && (isCompleted || isError))) {
      result.contexts.push(ClinicalContextRevisionV1Schema.parse({
        schemaVersion: "phenometric.clinical-context-revision.v1", contextId: stableId, revisionId, supersedesRevisionId,
        scope: options.scope, recordedAt, effectiveTime: parseFhirTime(effective,options.dateOffset),
        status: isError ? "entered-in-error" : "active", kind: type === "Procedure" ? "procedure" : "medication-change",
        valueCode: `${type === "MedicationStatement" ? `reported-medication:${String(resource.status??"unknown")}` : "procedure"}:${codeLabel(concept) ?? "unspecified"}`,
        sourceRef
      }));
    } else {
      result.excluded.push({ resourceId, reason: !mapping ? "unmapped-treatment-code" : "administration-not-confirmed-complete" });
    }
  }
  return result;
}
