/// <reference lib="webworker" />
// Benchmark-only wrapper: the production implementation processes every PCM block.
// Explicit stalls test real MessagePort backpressure without changing production code.
import "./voice-worker.js";
self.addEventListener("message", (event: MessageEvent) => {
  if (event.data?.type !== "research-stall") return;
  const duration = event.data.durationMs;
  if (!Number.isFinite(duration) || duration < 0 || duration > 1000) return;
  const until = performance.now() + duration;
  while (performance.now() < until) { /* Controlled overload; no inference claim. */ }
});
