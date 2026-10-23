import type { VoiceResearchConfigDocument, VoiceResearchModelConfig } from "./voice-research-hook.js";

/** Bind the actual source bytes, not a JSON reserialization. Legacy fixtures are explicitly unbound. */
export async function verifyResearchConfig(config: VoiceResearchModelConfig, document?: VoiceResearchConfigDocument):
Promise<{ config: VoiceResearchModelConfig; configurationSha256: string | null }> {
  if (!document) {
    if (config.estimator === "phenometrix.mlp-voicing-yin-pitch.v1") throw new Error("Hybrid configuration document required.");
    return { config, configurationSha256: null };
  }
  if (typeof document.text !== "string" || document.text.length > 16_384 || !/^[a-f0-9]{64}$/.test(document.sha256)) {
    throw new Error("Invalid research configuration document.");
  }
  const bytes = new TextEncoder().encode(document.text);
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    byte => byte.toString(16).padStart(2, "0")).join("");
  if (digest !== document.sha256) throw new Error("Research configuration digest mismatch.");
  const parsed = JSON.parse(document.text) as VoiceResearchModelConfig;
  if (JSON.stringify(parsed) !== JSON.stringify(config)) throw new Error("Research configuration object/document mismatch.");
  return { config: parsed, configurationSha256: digest };
}
