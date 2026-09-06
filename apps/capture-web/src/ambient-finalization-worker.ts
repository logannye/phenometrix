import { finalizeAmbientMetrics } from "@phenometrix/ambient-core";

// Only the same-origin module owns this worker. No network or raw media input.
self.onmessage = event => {
  const input = event.data?.input as Parameters<typeof finalizeAmbientMetrics>[0] | undefined;
  if (event.data?.type !== "finalize" || !input) { self.postMessage({ type: "unavailable" }); return; }
  try { self.postMessage({ type: "complete", result: finalizeAmbientMetrics(input) }); }
  catch { self.postMessage({ type: "unavailable" }); }
  finally {
    // The owner terminates the worker on completion, timeout, or withdrawal.
    input.voice.frames = []; input.face.frames = [];
  }
};
