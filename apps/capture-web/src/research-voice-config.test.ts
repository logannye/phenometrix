import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { verifyResearchConfig } from "./research-voice-config.js";
import type { VoiceResearchModelConfig } from "./voice-research-hook.js";
const config = { schemaVersion: "phenometrix.research-voice-model-config.v1", qualification: "unqualified-functional-fixture",
  estimator: "phenometrix.mlp-voicing-yin-pitch.v1" } as VoiceResearchModelConfig;
const document = (text: string) => ({ text, sha256: createHash("sha256").update(text).digest("hex") });
describe("research configuration byte identity", () => {
  it("binds source bytes independently, preserving whitespace identity", async () => {
    const original = document(JSON.stringify(config, null, 2) + "\n");
    expect(await verifyResearchConfig(config, original)).toEqual({ config, configurationSha256: original.sha256 });
    await expect(verifyResearchConfig(config, { ...original, text: JSON.stringify(config) })).rejects.toThrow("digest mismatch");
  });
  it("rejects missing hybrid identity, mismatched objects and unbounded documents", async () => {
    await expect(verifyResearchConfig(config)).rejects.toThrow("document required");
    await expect(verifyResearchConfig({ ...config, modelSha256: "changed" }, document(JSON.stringify(config)))).rejects.toThrow("object/document mismatch");
    await expect(verifyResearchConfig(config, document(" ".repeat(16385)))).rejects.toThrow("Invalid");
  });
  it("labels historical candidate fixtures with explicitly unbound configuration", async () => {
    const legacy = { ...config, estimator: undefined };
    expect(await verifyResearchConfig(legacy)).toEqual({ config: legacy, configurationSha256: null });
  });
});
