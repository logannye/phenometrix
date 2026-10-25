import { expect, test } from "@playwright/test";
import { appendFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";
import { dirname, resolve } from "node:path";
import type { ReplayResult, ReplayTelemetryBatch, NaturalPcmSource } from "../src/research-voice-replay.js";
import type { VoiceResearchModelConfig } from "../src/voice-research-hook.js";
const sha = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
const manifestPath = process.env.VOICE_NATURAL_MANIFEST;
const configPath = process.env.VOICE_RESEARCH_MODEL_CONFIG;

test("finite natural PCM keeps exact source origin, terminal fence and complete delivery evidence", async ({ browser }, testInfo) => {
  test.skip(!manifestPath || !configPath, "Set VOICE_NATURAL_MANIFEST and VOICE_RESEARCH_MODEL_CONFIG explicitly.");
  const inputPath = resolve(manifestPath!);
  const manifestBytes = readFileSync(inputPath);
  const manifest = JSON.parse(manifestBytes.toString());
  expect(manifest.schema).toBe("phenometrix.natural-browser-inputs.v1");
  expect(manifest.playlist).toEqual(["H", "H"]);
  expect(manifest.records.length).toBe(16);
  const text = readFileSync(resolve(configPath!), "utf8");
  const modelConfig = JSON.parse(text) as VoiceResearchModelConfig;
  expect(modelConfig.estimator).toBe("phenometrix.mlp-voicing-yin-pitch.v1");
  const modelConfigDocument = { text, sha256: sha(text) };
  const folder = process.env.VOICE_RESEARCH_OUTPUT_DIR ? resolve(process.env.VOICE_RESEARCH_OUTPUT_DIR) : testInfo.outputPath("natural-browser");
  mkdirSync(folder, { recursive: true });
  const root = resolve("../..");
  const sourcePaths = readdirSync(resolve("src")).filter(name => name.endsWith(".ts")).map(name => "apps/capture-web/src/" + name)
    .concat(["apps/capture-web/public/voice-capture-worklet.js", "apps/capture-web/e2e/voice-natural-research.spec.ts",
      "apps/capture-web/playwright.research.config.ts", "apps/capture-web/vite.research.config.ts", "pnpm-lock.yaml",
      "research/voice-prosody/phenometrix_prosody/prepare_natural_browser.py",
      "research/voice-prosody/phenometrix_prosody/audit_natural_browser.py", "research/voice-prosody/natural_browser_features.ts"]);
  const sourceSha256 = Object.fromEntries(sourcePaths.map(path => [path, sha(readFileSync(resolve(root, path)))]));
  for (const path of sourcePaths) {
    const target = resolve(folder, "source-snapshot", path); mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, readFileSync(resolve(root, path)), { flag: "wx" });
  }
  const assetBytes = Object.fromEntries((["model", "manifest", "wasm", "module"] as const).map(role => {
    const url = modelConfig[`${role}Url`];
    if (!url.startsWith("/.local/")) throw new Error("Expected explicitly local research assets.");
    return [role, readFileSync(resolve("public", url.slice(1))).byteLength];
  }));
  const hardware = { assetBytes, platform: platform(), osRelease: release(), cpuModel: cpus()[0]?.model, logicalCpus: cpus().length,
    totalMemoryBytes: totalmem(), browser: browser.version(), onnxRuntimeWeb: "1.23.2", wasmNumThreads: 1, wasmProxy: false,
    measurementProcessCount: 1, backgroundLoad: "Operator must exclude concurrent fitting/auditing; uncontrolled OS background load remains.",
    timingBenchmark: false, memoryScope: "Renderer JS heap sampled after each source; not a peak or total browser memory bound." };
  const cases = [{ ...manifest.indexFixture, caseId: "index-origin", role: "origin-fixture" },
    ...manifest.playlist.flatMap((model: string, index: number) => manifest.records.map((record: Record<string, unknown>) =>
      ({ ...record, caseId: `${index}-${model}-${record.id}`, role: "ordinary", model }))),
    { ...manifest.records[0], caseId: "injected-stall", role: "injected-stall", model: "H" }];
  test.setTimeout(cases.reduce((sum, record) => sum + record.sampleCount / 48 + 15000, 60000));
  const receipt = { schema: "phenometrix.natural-browser-run.v1", executionStatus: "running", startedAt: new Date().toISOString(),
    inputManifestSha256: sha(manifestBytes), configurationSha256: modelConfigDocument.sha256, hardware, sourceSha256,
    cases: [] as Array<Record<string, unknown>>, failures: [] as string[] };
  writeFileSync(resolve(folder, "started.json"), JSON.stringify(receipt) + "\n", { flag: "wx" });
  for (const record of cases) {
    const caseFolder = resolve(folder, record.caseId); mkdirSync(caseFolder);
    const rawPath = resolve(caseFolder, "raw.ndjson"); writeFileSync(rawPath, "", { flag: "wx" });
    const pcmPath = resolve(dirname(inputPath), record.pcmPath);
    if (!pcmPath.startsWith(dirname(inputPath) + "/")) throw new Error("Natural PCM path escaped input root.");
    const pcm = readFileSync(pcmPath);
    expect(sha(pcm)).toBe(record.pcmSha256);
    const context = await browser.newContext();
    const page = await context.newPage();
    let sequence = 0, records = 0;
    try {
      await page.route("**/__natural_pcm/source.f32le", route => route.fulfill({ status: 200, contentType: "application/octet-stream", body: pcm }));
      await page.exposeBinding("recordVoiceResearchBatch", (_source, batch: ReplayTelemetryBatch) => {
        if (batch.sequence !== ++sequence || batch.schemaVersion !== "phenometric.voice-replay-telemetry.v1" ||
            !batch.records.length || batch.records.length > 100) throw new Error("Natural raw telemetry order/capacity changed.");
        records += batch.records.length; appendFileSync(rawPath, JSON.stringify(batch) + "\n");
      });
      await page.goto("/research-voice.html");
      await page.waitForFunction(() => typeof window.runVoiceResearchReplay === "function");
      const naturalPcm: NaturalPcmSource = { id: record.id, pcmUrl: "/__natural_pcm/source.f32le",
        pcmSha256: record.pcmSha256, sampleCount: record.sampleCount, adapter: record.adapter };
      const result: ReplayResult = await page.evaluate(options => window.runVoiceResearchReplay(options), {
        naturalPcm, modelConfig, modelConfigDocument, streamRawTelemetry: true, clockDiagnostics: true,
        retainedFrameLimit: 16, retainedBlockLimit: 16,
        ...(record.role === "injected-stall" ? { overloadAtMs: 800, overloadDurationMs: 250 } : {}) });
      const session = await context.newCDPSession(page);
      await session.send("Performance.enable");
      const performanceMetrics = await session.send("Performance.getMetrics");
      const sampledJsHeapBytes = performanceMetrics.metrics.find(row => row.name === "JSHeapUsedSize")?.value ?? null;
      const saved = { ...result, timingBenchmark: false, inputManifestSha256: sha(manifestBytes),
        sourceSha256, case: record, rawArtifact: { file: "raw.ndjson", sha256: sha(readFileSync(rawPath)), batches: sequence, records }, sampledJsHeapBytes };
      writeFileSync(resolve(caseFolder, "result.json"), JSON.stringify(saved) + "\n", { flag: "wx" });
      receipt.cases.push({ id: record.caseId, role: record.role, resultPath: record.caseId + "/result.json",
        resultSha256: sha(readFileSync(resolve(caseFolder, "result.json"))), status: result.status });
      if (result.status !== "complete") receipt.failures.push(record.caseId + ": " + result.errors.join(","));
      if (record.role === "origin-fixture") {
        const origin = result.naturalSource?.origin as Record<string, unknown> | null;
        const expected = Array.from({ length: 16 }, (_, i) => pcm.readFloatLE(i * 4));
        if (JSON.stringify(origin?.firstSamples) !== JSON.stringify(expected) || result.status !== "complete")
          throw new Error("Index fixture failed source-zero alignment; speech execution is forbidden.");
      }
    } catch (error) {
      writeFileSync(resolve(caseFolder, "exception.json"), JSON.stringify({ error: String(error), at: new Date().toISOString() }) + "\n", { flag: "wx" });
      receipt.failures.push(record.caseId + ": " + String(error));
      if (record.role === "origin-fixture") break;
    } finally { await context.close(); }
  }
  const changedSources = sourcePaths.filter(path => sha(readFileSync(resolve(root, path))) !== sourceSha256[path]);
  if (changedSources.length) receipt.failures.push("Source changed during measurement: " + changedSources.join(","));
  receipt.executionStatus = receipt.failures.length ? "failed" : "complete";
  writeFileSync(resolve(folder, "run.json"), JSON.stringify({ ...receipt, completedAt: new Date().toISOString(), changedSources }) + "\n", { flag: "wx" });
  expect(receipt.cases.length).toBe(cases.length);
  expect(receipt.failures).toEqual([]);
});
