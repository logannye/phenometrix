import { test, expect, type Page } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { resolve, sep } from "node:path";
import { createConnectedServiceFixture } from "./connected-service.fixture.js";
import type { startSyntheticConnectedHost } from "./connected-host.fixture.js";

declare global { interface Window { syntheticConnectedHost: Awaited<ReturnType<typeof startSyntheticConnectedHost>> } }
const appRoot = fileURLToPath(new URL("../", import.meta.url));
let fixture: Awaited<ReturnType<typeof createConnectedServiceFixture>>;
let startedAtMs: number;
let elapsedMs: number;

test.beforeEach(async () => { startedAtMs = Date.now() + 2000; elapsedMs = 0; fixture = await createConnectedServiceFixture(startedAtMs); });
test.afterEach(async () => { await fixture.close(); });

async function advance(page: Page, durationMs: number) {
  for (let remaining = durationMs; remaining > 0;) {
    await expect.poll(() => page.evaluate(() => window.syntheticConnectedHost.transportState()), { timeout: 5000 })
      .toEqual({ requests: 0, probes: 0, authorization: 0 });
    const step = Math.min(5000, remaining); elapsedMs += step; remaining -= step;
    fixture.setNow(startedAtMs + elapsedMs); await page.clock.runFor(step);
    // Drain complete sequential probe cycles, including bodies and authorization,
    // rather than predicting exactly when asynchronous renewal will be scheduled.
    await expect.poll(() => page.evaluate(() => window.syntheticConnectedHost.transportState()), { timeout: 5000 })
      .toEqual({ requests: 0, probes: 0, authorization: 0 });
    expect(await page.evaluate(() => window.syntheticConnectedHost.stats.statuses), `Capture stopped at simulated ${elapsedMs}ms`).not.toContain("stopped");
  }
}

test("synthetic normal encounter reaches the real chart across rotations, camera gaps, retry and withdrawal", async ({ page }) => {
  test.setTimeout(180000);
  const browserErrors: string[] = [];
  page.on("pageerror", error => browserErrors.push(error.message));
  page.on("response", response => { if (response.status() >= 400) browserErrors.push(`${response.status()} ${response.url()}`); });
  // Concurrent developer edits must not let Vite HMR reload the synthetic encounter.
  await page.routeWebSocket(/.*/, socket => socket.close());
  await page.clock.install({ time: startedAtMs - 1000 });
  await page.goto("/");
  await page.clock.pauseAt(startedAtMs);
  const assetsRoot = resolve(appRoot, "../capture-web/public");
  // Serve the committed bytes verbatim: Vite's /@fs JS transformation changes asset hashes.
  await page.route("**/synthetic-capture-assets/**", async route => {
    const name = new URL(route.request().url()).pathname.split("/synthetic-capture-assets/")[1];
    const path = resolve(assetsRoot, name);
    if (!path.startsWith(assetsRoot + sep)) return route.abort();
    await route.fulfill({ path });
  });
  const input = { baseUrl: fixture.baseUrl, token: fixture.token, episodeId: fixture.episodeId, encounterId: fixture.encounterId,
    platformParticipantId: fixture.platformParticipantId, scope: fixture.scope, assetBaseUrl: "/synthetic-capture-assets/" };
  await page.evaluate(async ({ module, input }) => {
    const { startSyntheticConnectedHost } = await import(module);
    window.syntheticConnectedHost = await startSyntheticConnectedHost(input);
  }, { module: `/@fs/${appRoot}e2e/connected-host.fixture.ts`, input });
  expect(await page.evaluate(() => window.syntheticConnectedHost.stats.statuses)).toContain("started");
  await expect(page.getByRole("button")).toHaveCount(0);
  // Normal host events drive the encounter. No capture UI, manual baseline, or review action.
  await advance(page, 35000);
  const active = await page.evaluate(() => window.syntheticConnectedHost.stats);
  expect(active.acceptedFrames, JSON.stringify({ active, browserErrors })).toBeGreaterThan(0);
  const beforeHidden = await page.evaluate(() => { window.syntheticConnectedHost.chartVisible(false); return window.syntheticConnectedHost.stats.acceptedFrames; });
  await advance(page, 35000);
  expect(await page.evaluate(() => window.syntheticConnectedHost.stats.acceptedFrames)).toBeGreaterThan(beforeHidden);
  await page.evaluate(() => window.syntheticConnectedHost.camera(false));
  // The client lets its bounded positive UTC uncertainty elapse before ingest.
  await advance(page, 100);
  await expect.poll(async () => (await fixture.observations()).length).toBe(1);
  expect(await fixture.analyze()).toBe("published");
  await page.evaluate(async () => { window.syntheticConnectedHost.chartVisible(true); await window.syntheticConnectedHost.refresh(); });
  await expect(page.locator("#connected-chart circle.point")).toHaveCount(4);
  const first = (await fixture.observations())[0];
  expect(Date.parse(first.endedAt) - Date.parse(first.startedAt)).toBeLessThanOrEqual(70000);
  expect(first.metrics.filter(metric => metric.status === "measured").length).toBeGreaterThan(0);
  expect(first.metrics.every(metric => metric.usableDurationMs <= 70000)).toBe(true);

  // A whole unavailable window produces explicit missing evidence, not fabricated continuity.
  await advance(page, 300000);
  await expect.poll(async () => (await fixture.observations()).length).toBe(2);
  const unavailable = (await fixture.observations())[1];
  expect(unavailable.metrics.every(metric => metric.status === "withheld" && metric.value === null)).toBe(true);
  expect(unavailable.windows).toEqual([]);
  await page.evaluate(() => window.syntheticConnectedHost.camera(true));
  await advance(page, 70000);
  let dropped = 0;
  await page.route(`${fixture.baseUrl}/v1/episodes/${fixture.episodeId}/sessions`, async route => {
    if (dropped++ === 0) await route.abort("failed"); else await route.continue();
  });
  await advance(page, 230000);
  await expect.poll(() => dropped).toBe(1);
  await advance(page, 500);
  await expect.poll(async () => (await fixture.observations()).length).toBe(3);
  expect(dropped).toBe(2);
  const observations = await fixture.observations();
  expect(observations.filter(observation => Date.parse(observation.endedAt) - Date.parse(observation.startedAt) === 300000)).toHaveLength(2);
  expect(observations.every(observation => observation.capture.clockUncertaintyMs < 1000 && observation.capture.rawMediaRetained === false)).toBe(true);
  expect(observations.every(observation => observation.scope.participantId === fixture.scope.participantId && observation.encounterId === fixture.encounterId)).toBe(true);
  for (let index = 1; index < observations.length; index++) expect(Date.parse(observations[index].startedAt)).toBeGreaterThanOrEqual(Date.parse(observations[index - 1].endedAt));
  expect(await fixture.analyze()).toBe("published");
  await page.evaluate(() => window.syntheticConnectedHost.refresh());
  await expect(page.locator("#connected-chart circle.point")).toHaveCount(8);
  await expect(page.getByRole("button")).toHaveCount(1); // Optional axis toggle only.
  expect(await page.evaluate(() => window.syntheticConnectedHost.wrongScope())).toBe("encounter-scope-mismatch");
  const state = await fixture.repository.getEvidence(fixture.scope.tenantId, fixture.episodeId);
  const run = state.snapshots.at(-1)!.analysis;
  expect(run).toMatchObject({ clinicalClaim: "descriptive-only", fittedModels: false, status: "insufficient-data" });
  for (const row of run.rows as Array<{ points: Array<{ delta: number | null }>; baseline: { encounterCount: number }; phaseCoverage: Array<{count:number}> }>) {
    expect(row.points.every(point => point.delta === null)).toBe(true);
    expect(row.phaseCoverage[0].count).toBe(1); // Three windows are still one encounter.
  }
  expect(state.reviews).toEqual([]);
  const beforeWithdraw = (await fixture.observations()).length;
  await fixture.withdraw();
  await page.evaluate(async () => { window.syntheticConnectedHost.event("consent-withdrawn"); await window.syntheticConnectedHost.refresh(); });
  await expect(page.locator("#connected-chart article")).toHaveCount(0);
  fixture.setNow(startedAtMs + elapsedMs + 300000); await page.clock.fastForward(300000);
  expect((await fixture.observations()).length).toBe(beforeWithdraw);
  const stats = await page.evaluate(() => window.syntheticConnectedHost.stats);
  expect(stats.deviceRequests).toBe(0); expect(stats.trackStops).toBe(0);
  expect(stats.bitmapsCreated).toBe(stats.bitmapsClosed);
  expect(stats.finalizationWorkersStarted).toBe(3);
  expect(stats.finalizationWorkersActive).toBe(0); expect(stats.maximumFinalizationWorkers).toBe(1);
  expect(stats.statuses).toContain("stopped"); expect(stats.faceWorkerStops).toBe(1);
  expect(stats.requests.some(request => request.includes("/capture-context/"))).toBe(true);
  expect(stats.requests.slice(0, 3).every(request => request.endsWith("/clock"))).toBe(true);
  expect(stats.clockProbeCyclesCompleted).toBeGreaterThanOrEqual(20);
  expect(stats.clockProbeCyclesFailed).toBe(0);
  expect(stats.clockResponses).toBe(3 * stats.clockProbeCyclesCompleted);
  expect(stats.maximumInFlightRequests).toBeLessThanOrEqual(3);
  await page.evaluate(() => window.syntheticConnectedHost.dispose());
});
