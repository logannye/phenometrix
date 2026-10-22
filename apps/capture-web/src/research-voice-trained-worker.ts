/// <reference lib="webworker" />
import { configureVoiceResearchHook } from "./voice-worker.js";
import { createResearchVoiceMlp } from "./research-voice-mlp.js";
let configured = false;
self.addEventListener("message", async (event: MessageEvent) => {
  if (event.data?.type === "configure-research-model") {
    if (configured) { self.postMessage({ type: "research-model-error", reason: "Model already configured; start a new research worker." }); return; }
    configured = true;
    try {
      const began = performance.now();
      configureVoiceResearchHook(await createResearchVoiceMlp(event.data.config));
      self.postMessage({ type: "research-model-ready", loadMs: performance.now() - began,
        qualification: "unqualified-functional-fixture", modelSha256: event.data.config.modelSha256 });
    } catch (error) { self.postMessage({ type: "research-model-error", reason: String(error) }); }
  } else if (event.data?.type === "research-stall") {
    const duration = event.data.durationMs;
    if (!Number.isFinite(duration) || duration < 0 || duration > 1000) return;
    const until = performance.now() + duration;
    while (performance.now() < until) { /* Functional overload probe, not inference. */ }
  }
});
