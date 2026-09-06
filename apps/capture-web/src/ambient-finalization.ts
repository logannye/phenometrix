import type { AmbientWindowOptions } from "@phenometrix/encounter-capture";

/** One bounded worker per completed window; no calculation on the host UI thread. */
export const finalizeMetricsInWorker: NonNullable<AmbientWindowOptions["finalizeMetrics"]> = (input, signal) => {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    let worker: Worker | null = null;
    let settled = false;
    const finish = (result?: Awaited<ReturnType<typeof finalizeMetricsInWorker>>) => {
      if (settled) return;
      settled = true; clearTimeout(timeout); signal.removeEventListener("abort", aborted);
      if (worker) { worker.onmessage = worker.onerror = worker.onmessageerror = null; worker.terminate(); }
      if (result && !signal.aborted) resolve(result); else reject(new Error("ambient-finalization-unavailable"));
    };
    const aborted = () => finish();
    const timeout = setTimeout(aborted, 8_000);
    signal.addEventListener("abort", aborted, { once: true });
    try {
      worker = new Worker(new URL("./ambient-finalization-worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = event => finish(event.data?.type === "complete" ? event.data.result : undefined);
      worker.onerror = worker.onmessageerror = () => finish();
      worker.postMessage({ type: "finalize", input });
    } catch { finish(); }
  });
};
