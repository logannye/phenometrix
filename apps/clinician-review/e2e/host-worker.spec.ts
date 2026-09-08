import { test, expect } from "@playwright/test";
import { fileURLToPath } from "node:url";

test("a real browser worker summarizes derived signals without requesting devices and aborts on withdrawal", async ({ page }) => {
  // Dynamic cross-package fixture imports can trigger Vite dependency reloads.
  // The production worker has no HMR channel; keep the test context stable too.
  await page.routeWebSocket(/.*/, socket => socket.close());
  await page.goto("/");
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const result = await page.evaluate(async base => {
    const { finalizeMetricsInWorker } = await import(`${base}capture-web/src/ambient-finalization.ts`);
    const { voiceFrame } = await import(`${base}../packages/encounter-capture/src/test-fixtures.ts`);
    const { AMBIENT_LOCAL_PROTOCOL_REF: protocol } = await import(`${base}../packages/contracts/src/ambient-protocol.ts`);
    navigator.mediaDevices.getUserMedia = async () => { throw new Error("Unexpected device request"); };
    const input = { identity: { sessionId: "synthetic-worker", protocolVersion: protocol.version,
      protocolContentSha256: protocol.contentSha256, sessionStartedAtMs: 0 },
      voice: { frames: Array.from({ length: 3000 }, (_, index) => voiceFrame(index * 10)), noiseCalibrationDurationMs: 2000 },
      face: { frames: [], calibration: null } };
    const completed = await finalizeMetricsInWorker(input, new AbortController().signal);
    const abort = new AbortController();
    const canceled = finalizeMetricsInWorker(input, abort.signal).then(() => false, () => true);
    abort.abort();
    return { pitch: completed.outcomes.find((metric: { code: string }) => metric.code === "ambient.voice.f0.median"), canceled: await canceled };
  }, `/@fs/${root}`);
  expect(result.pitch).toMatchObject({ status: "measured", value: 150 });
  expect(result.canceled).toBe(true);
  await expect(page.getByRole("heading", { name: "Observations following treatment" })).toBeVisible();
});
