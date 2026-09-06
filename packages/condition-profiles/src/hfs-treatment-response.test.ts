import { describe, expect, it } from "vitest";
import { AMBIENT_LOCAL_PROTOCOL_PACK, verifyTreatmentResponseArtifact } from "@phenometrix/contracts";
import { createHfsTreatmentResponseProtocol, createHfsTreatmentResponseSpecification } from "./hfs-treatment-response.js";

describe("HFS research treatment-alignment protocol", () => {
  it("uses only current source-defined measurements and excludes an unvalidated platform acquisition", async () => {
    const protocol = await createHfsTreatmentResponseProtocol();
    expect(await verifyTreatmentResponseArtifact(protocol)).toBe(true);
    expect(await createHfsTreatmentResponseProtocol()).toEqual(protocol);
    expect(protocol.permittedSourceKinds).not.toContain("platform-patient-track");
    expect(protocol.metrics).toHaveLength(4);
    for (const metric of protocol.metrics) {
      const source = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(source => source.code === metric.metricCode)!;
      expect(metric).toMatchObject({ unit: source.unit, context: source.context, modality: source.modality, algorithmVersion: source.algorithmVersion });
    }
    expect(protocol.validatedClaim).toBe("none");
    expect(protocol.prohibitedClaims).toContain("spasm-frequency");
  });

  it("seals the repeatable baseline and observation-coverage policy with models disabled", async () => {
    const specification = await createHfsTreatmentResponseSpecification({ scope: { tenantId: "clinic", studyId: "study", participantId: "participant" }, anchorTreatmentId: "injection" });
    expect(await verifyTreatmentResponseArtifact(specification)).toBe(true);
    expect(specification.baseline.minimumEncounters).toBe(2);
    expect(specification.fittedModels).toBe(false); expect(specification.clinicalClaim).toBe("descriptive-only");
    expect(specification.phases.map(phase => phase.phaseId)).toEqual(["days-0-to-14", "days-14-to-42", "days-42-to-112"]);
  });
});
