import { expect, test, type Page, type Request } from "@playwright/test";
import {
  ambientProbe,
  failAudioProcessor,
  failFaceWorker,
  installAmbientBrowserFixture,
  resolveLateAudio,
  resolveLateWorklet
} from "./ambient-browser-fixture.js";

const appUrl = "/phenometrix/";

async function consentAndStart(page: Page): Promise<void> {
  await page.locator("#affected-side-left").check();
  await page.locator("#consent-checkbox").check();
  await expect(page.locator("#start-button")).toBeEnabled();
  await page.locator("#start-button").click();
}

async function startAudioOnlyObservation(page: Page): Promise<void> {
  await consentAndStart(page);
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });
}

test("consent gates ambient setup and states the implemented boundary", async ({
  page
}) => {
  await installAmbientBrowserFixture(page, "deny-all");
  await page.goto(appUrl);

  await expect(page.locator("#welcome-title")).toHaveText(
    "Compare two facial movement sessions, locally."
  );
  await expect(page.getByText("Nonclinical research prototype")).toBeVisible();
  await expect(page.getByText("Not for new or sudden facial weakness")).toBeVisible();
  await expect(page.locator("#start-button")).toBeDisabled();
  await page.locator("#consent-checkbox").check();
  await expect(page.locator("#start-button")).toBeDisabled();
  await page.locator("#affected-side-left").check();
  await expect(page.locator("#start-button")).toBeEnabled();
});

test("permission denial creates no report and leaves devices off", async ({
  page
}) => {
  await installAmbientBrowserFixture(page, "deny-all");
  await page.goto(appUrl);
  await consentAndStart(page);

  await expect(page.locator("#message-title")).toHaveText("Session unavailable");
  await expect(page.locator("#message-detail")).toContainText(
    "No report was created"
  );
  await expect(page.locator("#report-view")).toBeHidden();
  await expect(page.locator("#privacy-state")).toHaveText("Devices off");
  expect((await ambientProbe(page)).trackStops).toBe(0);
});

test("audio calibration starts observation while face abstains independently", async ({
  page
}) => {
  await installAmbientBrowserFixture(page, "audio-only");
  await page.goto(appUrl);
  await startAudioOnlyObservation(page);

  await expect(page.locator("#audio-lane-state")).toHaveText("Ready");
  await expect(page.locator("#face-lane-state")).toHaveText("Not measurable");
  await expect(page.locator("#finish-button")).toBeEnabled();
  await expect(page.locator("#capture-instruction")).toContainText(
    "No exercises or scripted prompts"
  );
});

test("dual-lane capture shows the live face mesh and bounded voice dashboard", async ({
  page
}) => {
  await installAmbientBrowserFixture(page, "dual-lane");
  await page.goto(appUrl);
  await consentAndStart(page);
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });

  await expect(page.locator("#face-mesh-status")).toHaveText(
    "◆ TRACKING · 478 pts"
  );
  await expect(page.locator("#landmark-overlay")).toBeVisible();
  await expect(page.locator("#voice-live-state")).toHaveText(
    "Quiet/background"
  );
  await expect(page.locator("#voice-live-state")).toHaveText(
    "Speech/noise",
    { timeout: 1_500 }
  );
  await expect(page.locator("#voice-live-state")).toHaveText(
    "Voiced speech",
    { timeout: 1_500 }
  );
  await expect(page.locator("#voice-pitch-value")).toContainText("Hz");
  await expect(page.locator("#voice-level-value")).toContainText("dBFS");
  await expect(page.locator("#voice-energy-chart")).toHaveAttribute(
    "data-sample-count",
    "800",
    { timeout: 2_000 }
  );

  // NOTE: this fixture (ambient-browser-fixture.ts) replaces window.Worker
  // with a same-thread WorkerMock, so the real face-worker.ts (renderer
  // selection, rAF loop, drawFrame canvas resize) never runs here. This
  // poll only confirms the #landmark-overlay <canvas> element is present
  // and attached to the DOM — it does NOT exercise the WebGL2/2D renderer
  // or rAF-driven buffer sizing. Live mesh rendering is verified manually
  // via `pnpm dev`, not by this smoke test.
  const overlay = page.locator("#landmark-overlay");
  await expect
    .poll(async () => overlay.evaluate((c: HTMLCanvasElement) => c.width))
    .toBeGreaterThan(0);

  // Telemetry: the gauges + waveform canvases attach and a live readout
  // populates (replacing the "—" placeholder) during dual-lane capture.
  await expect(page.locator("#voice-level-gauge")).toBeAttached();
  await expect(page.locator("#voice-energy-chart")).toBeAttached();
  await expect(page.locator("#voice-clarity-chart")).toBeAttached();
  await expect
    .poll(async () => page.locator("#voice-level-value").textContent())
    .not.toBe("—");

  await page.locator("#discard-button").click();
  await expect(page.locator("#message-title")).toHaveText("Session discarded");
  await expect(page.locator("#landmark-overlay")).toBeHidden();
  await expect(page.locator("#face-mesh-status")).toBeHidden();
  await expect(page.locator("#voice-energy-chart")).toHaveAttribute(
    "data-sample-count",
    "0"
  );
  await expect.poll(async () => await ambientProbe(page)).toMatchObject({
    trackStops: 2,
    audioContextsClosed: 1,
    workersTerminated: 2
  });
});

test("lane processor failures release only that lane and finalize after the last lane", async ({
  page
}) => {
  await installAmbientBrowserFixture(page, "dual-lane");
  await page.goto(appUrl);
  await consentAndStart(page);
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });

  await failFaceWorker(page);
  await expect(page.locator("#face-lane-state")).toHaveText("Not measurable");
  await expect(page.locator("#audio-lane-state")).toHaveText("Ready");
  await expect(page.locator("#phase-label")).toHaveText("Ambient session");
  await expect(page.locator("#privacy-state")).toHaveText(
    "Devices active · local only"
  );
  await expect.poll(async () => await ambientProbe(page)).toMatchObject({
    audioTrackStops: 0,
    videoTrackStops: 1,
    audioContextsClosed: 0,
    voiceWorkersTerminated: 0,
    faceWorkersTerminated: 1
  });

  await failAudioProcessor(page);
  await expect(page.locator("#privacy-state")).toHaveText("Devices off");
  await expect(page.locator("#phase-label")).toHaveText("Report ready", {
    timeout: 15_000
  });
  await expect.poll(async () => await ambientProbe(page)).toMatchObject({
    audioTrackStops: 1,
    videoTrackStops: 1,
    audioContextsClosed: 1,
    voiceWorkersTerminated: 1,
    faceWorkersTerminated: 1
  });
  await expect.poll(async () => await ambientProbe(page)).toMatchObject({
    audioTrackStops: 1,
    videoTrackStops: 1,
    audioContextsClosed: 1,
    voiceWorkersTerminated: 1,
    faceWorkersTerminated: 1
  });
});

test("ambient finalization creates the bounded report without upload or persistence", async ({
  page
}) => {
  const requests: Array<Pick<Request, "url" | "method">> = [];
  page.on("request", (request) => requests.push(request));
  await installAmbientBrowserFixture(page, "audio-only");
  await page.goto(appUrl);
  await startAudioOnlyObservation(page);
  await page.locator("#finish-button").click();

  await expect(page.locator("#report-title")).toHaveText(
    "Ambient session measurement report"
  );
  await expect(page.locator(".report-section")).toHaveCount(10);
  await expect(page.locator(".metric-row")).toHaveCount(27);
  await expect(page.locator("#report-boundary")).toContainText(
    "not intended for medical decisions or longitudinal comparison"
  );
  const pitch = page.locator(
    '[data-metric-code="ambient.voice.f0.median"]'
  );
  await expect(pitch.locator(".metric-value strong")).toHaveText(
    "Not measurable"
  );
  await pitch.locator("summary").click();
  await expect(pitch.locator(".trace-grid")).toContainText("no-usable-signal");

  const origin = new URL(page.url()).origin;
  expect(requests.length).toBeGreaterThan(0);
  expect(
    requests.every(
      (request) => request.method() === "GET" && request.url().startsWith(origin)
    )
  ).toBe(true);
  expect(
    await page.evaluate(async () => ({
      localStorage: localStorage.length,
      sessionStorage: sessionStorage.length,
      indexedDatabases:
        typeof indexedDB.databases === "function"
          ? (await indexedDB.databases()).length
          : 0
    }))
  ).toEqual({ localStorage: 0, sessionStorage: 0, indexedDatabases: 0 });
  expect(await ambientProbe(page)).toMatchObject({
    trackStops: 1,
    audioContextsClosed: 1,
    workersTerminated: 1
  });
});

test("two live captures produce a page-memory-only condition comparison card", async ({
  page
}) => {
  const requests: Array<Pick<Request, "url" | "method">> = [];
  page.on("request", (request) => requests.push(request));
  await installAmbientBrowserFixture(page, "dual-lane");
  await page.goto(appUrl);

  await consentAndStart(page);
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });
  await page.locator("#finish-button").click();
  await expect(page.locator("#phase-label")).toHaveText("Report ready", {
    timeout: 15_000
  });
  await expect.poll(async () => await ambientProbe(page)).toMatchObject({
    audioTrackStops: 1,
    videoTrackStops: 1,
    audioContextsClosed: 1,
    voiceWorkersTerminated: 1,
    faceWorkersTerminated: 1
  });
  await expect(page.locator("#condition-status")).toContainText(
    "accepted as the page-memory comparison reference"
  );
  await expect(page.locator("#accept-reference-button")).toBeVisible();

  await page.locator("#accept-reference-button").click();
  await expect(page.locator("#condition-status")).toContainText(
    "Reference accepted in page memory"
  );
  await page.locator("#follow-up-button").click();
  await expect(page.locator("#welcome-view")).toBeVisible();
  await expect(page.locator("#affected-side-left")).toBeChecked();
  await expect(page.locator("#affected-side-left")).toBeDisabled();
  await expect(page.locator("#start-button")).toBeDisabled();

  await page.locator("#consent-checkbox").check();
  await expect(page.locator("#start-button")).toBeEnabled();
  await page.locator("#start-button").click();
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });
  await page.locator("#finish-button").click();
  await expect(page.locator("#phase-label")).toHaveText("Report ready", {
    timeout: 15_000
  });
  await expect.poll(async () => await ambientProbe(page)).toMatchObject({
    audioTrackStops: 2,
    videoTrackStops: 2,
    audioContextsClosed: 2,
    voiceWorkersTerminated: 2,
    faceWorkersTerminated: 2
  });

  await expect(page.locator("#condition-card")).toBeVisible();
  await expect(page.locator(".condition-row")).toHaveCount(6);
  await expect(page.locator("#condition-card-summary")).toContainText(
    "unknown minimum detectable change"
  );
  await expect(page.locator("#condition-card-summary")).toContainText(
    "0 incompatible"
  );
  await expect(
    page.locator('.condition-row[data-asserted-affected-side="true"]')
  ).toHaveCount(1);
  await expect(
    page.locator('.condition-row[data-asserted-affected-side="true"]')
  ).toContainText("Participant-asserted affected side");
  await expect(
    page.locator('.condition-row:has-text("Oculo-oral coupling difference")')
  ).toContainText("Sign: subject-left minus subject-right");
  await page.locator(".condition-row details").first().click();
  const firstTrace = page.locator(".condition-row details").first();
  const referenceObservationId = await firstTrace
    .locator(".trace-grid div")
    .filter({ hasText: "Reference observation" })
    .locator("code")
    .textContent();
  const currentObservationId = await firstTrace
    .locator(".trace-grid div")
    .filter({ hasText: "Current observation" })
    .locator("code")
    .textContent();
  const referenceSessionId = await firstTrace
    .locator(".trace-grid div")
    .filter({ hasText: "Reference session" })
    .locator("code")
    .textContent();
  const currentSessionId = await firstTrace
    .locator(".trace-grid div")
    .filter({ hasText: "Current session" })
    .locator("code")
    .textContent();
  expect(referenceObservationId).not.toBe(currentObservationId);
  expect(referenceSessionId).not.toBe(currentSessionId);
  await expect(page.locator(".condition-row details").first()).toContainText(
    "Reference technical quality"
  );
  await expect(page.locator(".condition-row details").first()).toContainText(
    "Current eligible duration"
  );
  await expect(page.locator(".condition-row details").first()).toContainText(
    "Current usable bins"
  );
  await expect(page.locator(".condition-row details").first()).toContainText(
    "Current expression events"
  );
  await expect(page.locator("#condition-panel")).not.toContainText(
    /has improved|has worsened|has recovered|diagnosed as|severity score/i
  );
  await page.locator("#condition-accept-button").click();
  await expect(page.locator("#condition-review-state")).toHaveText(
    "Accepted for this page session"
  );

  const origin = new URL(page.url()).origin;
  expect(
    requests.every(
      (request) => request.method() === "GET" && request.url().startsWith(origin)
    )
  ).toBe(true);
  expect(
    await page.evaluate(async () => ({
      localStorage: localStorage.length,
      sessionStorage: sessionStorage.length,
      indexedDatabases:
        typeof indexedDB.databases === "function"
          ? (await indexedDB.databases()).length
          : 0
    }))
  ).toEqual({ localStorage: 0, sessionStorage: 0, indexedDatabases: 0 });

  await page.locator("#reset-button").click();
  await expect(page.locator("#welcome-view")).toBeVisible();
  await expect(page.locator("#affected-side-left")).not.toBeChecked();
  await expect(page.locator("#affected-side-left")).toBeEnabled();
});

test("visibility loss clears an accepted condition reference", async ({ page }) => {
  await installAmbientBrowserFixture(page, "dual-lane");
  await page.goto(appUrl);
  await consentAndStart(page);
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });
  await page.locator("#finish-button").click();
  await expect(page.locator("#phase-label")).toHaveText("Report ready", {
    timeout: 15_000
  });
  await page.locator("#accept-reference-button").click();
  await expect(page.locator("#condition-status")).toContainText(
    "Reference accepted in page memory"
  );

  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await expect(page.locator("#welcome-view")).toBeVisible();
  await expect(page.locator("#affected-side-left")).not.toBeChecked();
  await expect(page.locator("#affected-side-left")).toBeEnabled();
  await expect(page.locator("#condition-card")).toBeHidden();
});

for (const phase of ["report", "follow-up setup"] as const) {
  test(`pagehide clears the accepted reference from ${phase} before page restoration`, async ({ page }) => {
    await installAmbientBrowserFixture(page, "dual-lane");
    await page.goto(appUrl);
    await consentAndStart(page);
    await expect(page.locator("#phase-label")).toHaveText("Ambient session");
    await page.locator("#finish-button").click();
    await expect(page.locator("#phase-label")).toHaveText("Report ready");
    await page.locator("#accept-reference-button").click();
    await expect(page.locator("#condition-status")).toContainText(
      "Reference accepted in page memory"
    );
    if (phase === "follow-up setup") {
      await page.locator("#follow-up-button").click();
      await expect(page.locator("#affected-side-left")).toBeDisabled();
    }

    // Exercise pagehide independently of visibilitychange. A cached document
    // can be restored with its JavaScript heap and rendered report intact.
    await page.evaluate(() => {
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }));
      window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true }));
    });

    await expect(page.locator("#welcome-view")).toBeVisible();
    await expect(page.locator("#affected-side-left")).not.toBeChecked();
    await expect(page.locator("#affected-side-left")).toBeEnabled();
    await expect(page.locator("#consent-checkbox")).not.toBeChecked();
    await expect(page.locator("#report-sections")).toBeEmpty();
    await expect(page.locator("#condition-card")).toBeHidden();
    await expect(page.locator("#condition-side-badge")).toHaveText("Affected side not set");
    await expect(page.locator("#follow-up-button")).toBeHidden();

    // A fresh capture must become a first observation, never an implicit
    // comparison with the reference that existed before leaving the page.
    await consentAndStart(page);
    await expect(page.locator("#phase-label")).toHaveText("Ambient session");
    await page.locator("#finish-button").click();
    await expect(page.locator("#phase-label")).toHaveText("Report ready");
    await expect(page.locator("#accept-reference-button")).toBeVisible();
    await expect(page.locator("#condition-card")).toBeHidden();
  });
}

test("withdrawing the follow-up clears its accepted reference", async ({ page }) => {
  await installAmbientBrowserFixture(page, "dual-lane");
  await page.goto(appUrl);
  await consentAndStart(page);
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });
  await page.locator("#finish-button").click();
  await expect(page.locator("#phase-label")).toHaveText("Report ready", {
    timeout: 15_000
  });
  await page.locator("#accept-reference-button").click();
  await page.locator("#follow-up-button").click();
  await page.locator("#consent-checkbox").check();
  await page.locator("#start-button").click();
  await expect(page.locator("#phase-label")).toHaveText("Ambient session", {
    timeout: 15_000
  });

  await page.locator("#discard-button").click();
  await expect(page.locator("#message-title")).toHaveText("Session discarded");
  await expect(page.locator("#message-detail")).toContainText(
    "all local session data was cleared"
  );
  await page.locator("#message-reset-button").click();
  await expect(page.locator("#welcome-view")).toBeVisible();
  await expect(page.locator("#affected-side-left")).not.toBeChecked();
  await expect(page.locator("#affected-side-left")).toBeEnabled();
});

test("discard, withdrawal, and late media resolution release resources", async ({
  page
}) => {
  await test.step("an already granted video stream is stopped while audio remains pending", async () => {
    await installAmbientBrowserFixture(page, "video-with-late-audio");
    await page.goto(appUrl);
    await consentAndStart(page);
    await expect(page.locator("#phase-label")).toHaveText("Requesting devices");

    await page.locator("#discard-button").click();
    await expect(page.locator("#message-title")).toHaveText("Session discarded");
    await expect.poll(async () => (await ambientProbe(page)).trackStops).toBe(1);

    await resolveLateAudio(page);
    await expect.poll(async () => (await ambientProbe(page)).trackStops).toBe(2);
    await expect(page.locator("#report-view")).toBeHidden();
  });

  await test.step("discard cancels voice startup while the worklet module remains pending", async () => {
    await installAmbientBrowserFixture(page, "late-worklet");
    await page.goto(appUrl);
    await consentAndStart(page);
    await expect.poll(
      async () => (await ambientProbe(page)).workletModuleRequests
    ).toBe(1);

    await page.locator("#discard-button").click();
    await expect(page.locator("#message-title")).toHaveText("Session discarded");
    await expect.poll(async () => await ambientProbe(page)).toMatchObject({
      trackStops: 1,
      audioContextsClosed: 1,
      workersTerminated: 0
    });

    await resolveLateWorklet(page);
    await page.waitForTimeout(50);
    await expect.poll(async () => await ambientProbe(page)).toMatchObject({
      audioContextsClosed: 1,
      workersTerminated: 0
    });
  });

  await test.step("a late stream is stopped after discard", async () => {
    await installAmbientBrowserFixture(page, "late-audio");
    await page.goto(appUrl);
    await consentAndStart(page);
    await expect(page.locator("#phase-label")).toHaveText("Requesting devices");
    await page.locator("#discard-button").click();
    await resolveLateAudio(page);
    await expect(page.locator("#message-title")).toHaveText("Session discarded");
    await expect.poll(async () => (await ambientProbe(page)).trackStops).toBe(1);
    await expect(page.locator("#report-view")).toBeHidden();
  });

  await test.step("withdrawing consent disposes an active session", async () => {
    await installAmbientBrowserFixture(page, "audio-only");
    await page.goto(appUrl);
    await startAudioOnlyObservation(page);
    await page.evaluate(() => {
      const checkbox = document.querySelector<HTMLInputElement>(
        "#consent-checkbox"
      );
      if (!checkbox) throw new Error("Consent checkbox is unavailable.");
      checkbox.checked = false;
      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(page.locator("#message-title")).toHaveText("Session discarded");
    await expect.poll(async () => (await ambientProbe(page)).trackStops).toBe(1);
    await expect(page.locator("#report-view")).toBeHidden();
  });
});
