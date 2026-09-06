import { test, expect } from "@playwright/test";

test("existing chart can show evidence with no capture or mandatory review controls", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Observations following treatment" })).toBeVisible();
  await expect(page.getByRole("button")).toHaveCount(1);
  await expect(page.locator("circle.point")).toHaveCount(10);
  await expect(page.locator("polyline, path")).toHaveCount(0);
  await page.getByRole("button", { name: "Days since treatment" }).click();
  await expect(page.getByRole("button", { name: "Calendar dates" })).toBeVisible();
  await page.getByText("Inspect observations and exclusions").first().click();
  await expect(page.getByText("synthetic-withheld: metric-withheld").first()).toBeVisible();
});

test("failed or mismatched host data stays silent and untrusted strings render as text", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    // @ts-expect-error Vite serves source modules in this synthetic preview only.
    const { mountTreatmentResponsePanel } = await import("/src/panel.ts");
    // @ts-expect-error Vite source import.
    const { syntheticRun, SYNTHETIC_SCOPE } = await import("/src/synthetic.ts");
    const other = document.createElement("div"); other.id = "silent"; document.body.append(other);
    const panel = mountTreatmentResponsePanel(other, { scope: SYNTHETIC_SCOPE, load: async () => { throw new Error("synthetic network failure"); } });
    await panel.refresh();
    const wrong = mountTreatmentResponsePanel(other, { scope: { ...SYNTHETIC_SCOPE, participantId: "other" }, load: async () => ({ run: await syntheticRun() }) });
    await wrong.refresh();
    const unsafe = document.createElement("div"); unsafe.id = "unsafe-label"; document.body.append(unsafe);
    const escaped = mountTreatmentResponsePanel(unsafe, { scope: SYNTHETIC_SCOPE, load: async () => ({ run: await syntheticRun('<img src=x onerror="alert(1)">') }) });
    await escaped.refresh();
  });
  await expect(page.locator("#silent article")).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.locator("#unsafe-label img")).toHaveCount(0);
  await expect(page.locator("#unsafe-label").getByText('<img src=x onerror="alert(1)">')).toBeVisible();
});

test("an unavailable treatment anchor clears cycle mode without drawing an administered-treatment marker", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    // @ts-expect-error Vite source import.
    const { mountTreatmentResponsePanel } = await import("/src/panel.ts");
    // @ts-expect-error Vite source import.
    const { syntheticRun, syntheticUnavailableAnchorRun, SYNTHETIC_SCOPE } = await import("/src/synthetic.ts");
    const host = document.createElement("div"); document.body.append(host); let ready = true;
    const panel = mountTreatmentResponsePanel(host, { scope: SYNTHETIC_SCOPE, load: async () => ({ run: await (ready ? syntheticRun() : syntheticUnavailableAnchorRun()) }) });
    await panel.refresh(); host.firstElementChild!.shadowRoot!.querySelector("button")!.click();
    ready = false; await panel.refresh();
    const shadow = host.firstElementChild!.shadowRoot!;
    return { buttons: shadow.querySelectorAll("button").length, markers: shadow.querySelectorAll(".treatment").length, points: shadow.querySelectorAll("circle").length, chartText: shadow.querySelector("svg")?.textContent };
  });
  expect(result.buttons).toBe(0); expect(result.markers).toBe(0); expect(result.points).toBe(10); expect(result.chartText).not.toContain("Day ");
});

test("late host results cannot replace a newer refresh and disposal clears evidence", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    // @ts-expect-error Vite source import.
    const { mountTreatmentResponsePanel } = await import("/src/panel.ts");
    // @ts-expect-error Vite source import.
    const { syntheticRun, SYNTHETIC_SCOPE } = await import("/src/synthetic.ts");
    let resolve: (value: unknown) => void = () => {};
    let n = 0;
    const host = document.createElement("div"); document.body.append(host);
    const panel = mountTreatmentResponsePanel(host, { scope: SYNTHETIC_SCOPE, load: async () => ++n === 1 ? new Promise(r => { resolve = r; }) : ({ run: null }) });
    const first = panel.refresh(); await panel.refresh(); resolve({ run: await syntheticRun() }); await first;
    const shown = host.firstElementChild?.shadowRoot?.querySelectorAll("article").length;
    panel.dispose(); return { shown, remaining: host.childElementCount };
  });
  expect(result).toEqual({ shown: 0, remaining: 0 });
});
