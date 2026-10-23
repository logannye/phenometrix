/// <reference lib="webworker" />
import { createResearchVoiceMlp } from "./research-voice-mlp.js";
import { verifyResearchConfig } from "./research-voice-config.js";
import type { VoiceResearchPrediction } from "./voice-research-hook.js";
import type { VoiceFrameTiming } from "./voice-stream-timing.js";
let used = false;
self.onmessage = async (event: MessageEvent) => {
  if (used) return;
  used = true;
  const request = event.data;
  let hook: Awaited<ReturnType<typeof createResearchVoiceMlp>> | undefined;
  try {
    const pcm = request.pcm as ArrayBuffer;
    const ends = request.windowEndSamples as number[];
    if (!(pcm instanceof ArrayBuffer) || pcm.byteLength > 960000 * 4 || pcm.byteLength % 4 ||
        !Array.isArray(ends) || ends.length > 2000 || !ends.length) throw new Error("Invalid bounded parity PCM/window fixture.");
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", pcm)), value => value.toString(16).padStart(2, "0")).join("");
    if (digest !== request.pcmSha256) throw new Error("Parity PCM digest mismatch.");
    const samples = new Float32Array(pcm);
    if (!samples.every(Number.isFinite) || !ends.every((end, i) => Number.isSafeInteger(end) && end >= 1920 &&
        end <= samples.length && (i === 0 || end === ends[i - 1] + 480))) throw new Error("Invalid parity frame support.");
    const verified = await verifyResearchConfig(request.config, request.configDocument);
    if (verified.config.estimator !== "phenometrix.mlp-voicing-yin-pitch.v1") throw new Error("Hybrid configuration required.");
    hook = await createResearchVoiceMlp(verified.config, verified.configurationSha256);
    const frames: VoiceResearchPrediction[] = [];
    for (const end of ends) {
      const features = hook.prepare(samples.subarray(end - 1920, end), 48000);
      // Arithmetic-only hook fixture: no fabricated acquisition, lease, or latency observation.
      const support = { supportStartSample: end - 1920, supportEndSampleExclusive: end,
        lastRequiredInputSample: end - 1, sampleRateHz: 48000 } as VoiceFrameTiming;
      frames.push(await hook.predict(features, support));
    }
    self.postMessage({ status: "complete", evidence: "bounded-pcm-hook-arithmetic-parity-not-paced-audio",
      pcmSha256: digest, provenance: hook.provenance, frames });
  } catch (error) { self.postMessage({ status: "failed", reason: String(error) }); }
  finally { await hook?.dispose(); }
};
