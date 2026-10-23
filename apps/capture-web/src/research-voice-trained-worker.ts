/// <reference lib="webworker" />
import { configureVoiceResearchHook } from "./voice-worker.js";
import { createResearchVoiceMlp } from "./research-voice-mlp.js";
import { verifyResearchConfig } from "./research-voice-config.js";
let configured = false;
self.addEventListener("message", async (event: MessageEvent) => {
  if (event.data?.type === "configure-research-model") {
    if (configured) { self.postMessage({ type: "research-model-error", reason: "Model already configured; start a new research worker." }); return; }
    configured = true;
    try {
      const began = performance.now();
      const verified = await verifyResearchConfig(event.data.config, event.data.configDocument);
      const hook = await createResearchVoiceMlp(verified.config, verified.configurationSha256);
      configureVoiceResearchHook(hook);
      self.postMessage({ type: "research-model-ready", loadMs: performance.now() - began,
        qualification: "unqualified-functional-fixture", ...hook.provenance });
    } catch (error) { self.postMessage({ type: "research-model-error", reason: String(error) }); }
  } else if (event.data?.type === "research-stall") {
    const duration = event.data.durationMs;
    if (!Number.isFinite(duration) || duration < 0 || duration > 1000) return;
    const until = performance.now() + duration;
    while (performance.now() < until) { /* Functional overload probe, not inference. */ }
  }
});
