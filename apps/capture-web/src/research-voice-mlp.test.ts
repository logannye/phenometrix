import { afterEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { VOICE_PITCH_FEATURE_NAMES_V2 } from "./voice-pitch-features-v2.js";
import { HYBRID_ESTIMATOR_CONTRACT, ORIGINAL_MLP_WEIGHTS_SHA256 } from "./research-voice-yin.js";
import type { VoiceResearchModelConfig } from "./voice-research-hook.js";
import type { VoiceFrameTiming } from "./voice-stream-timing.js";
const ortMock = vi.hoisted(() => ({ run: vi.fn(), release: vi.fn(async () => {}) }));
vi.mock("onnxruntime-web/wasm", () => ({ env: { wasm: {} },
  Tensor: class { constructor(public type: string, public data: Float32Array | Uint8Array, public dims: number[]) {} dispose() {} },
  InferenceSession: { create: async () => ({ run: ortMock.run, release: ortMock.release }) } }));
import { createResearchVoiceMlp } from "./research-voice-mlp.js";
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
function fixture(weights = ORIGINAL_MLP_WEIGHTS_SHA256) {
  const model = new Uint8Array([1]), wasm = new Uint8Array([2]), module = new Uint8Array([3]);
  const manifest = new TextEncoder().encode(JSON.stringify({ status: "complete", training_performed: true, scaling_fitted: true,
    config: { kind: "framewise-mlp", feature_version: "v2", feature_count: 57, candidate_count: 6, hidden_size: 32, projection_size: 32 },
    training_metadata: { measurement_delay_ms: 20, weights_sha256: weights }, artifacts: { onnx: { sha256: hash(model) } },
    feature_names: VOICE_PITCH_FEATURE_NAMES_V2 }));
  const assets: Record<string, Uint8Array> = { "/manifest": manifest, "/model": model, "/wasm": wasm, "/module": module };
  vi.stubGlobal("self", { location: { href: "http://localhost/worker", origin: "http://localhost" } });
  vi.stubGlobal("fetch", async (url: URL) => ({ ok: true, arrayBuffer: async () => assets[url.pathname].slice().buffer }));
  return { schemaVersion: "phenometrix.research-voice-model-config.v1", qualification: "unqualified-functional-fixture",
    estimator: HYBRID_ESTIMATOR_CONTRACT, manifestUrl: "/manifest", manifestSha256: hash(manifest), modelUrl: "/model", modelSha256: hash(model),
    wasmUrl: "/wasm", wasmSha256: hash(wasm), moduleUrl: "/module", moduleSha256: hash(module) } satisfies VoiceResearchModelConfig;
}
const output = () => Object.fromEntries([["voicing_logit", 1], ["candidate_logits", 6], ["state_out", 32]].map(([name, count]) => [name,
  { type: "float32", dims: [1, count], data: Float32Array.from({ length: Number(count) }, () => 1), dispose() {} }]));
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); });
describe("research-only original MLP/raw YIN hook", () => {
  it("binds each raw YIN result to its prepared frame across serial inference and reset", async () => {
    const config = fixture();
    let finish!: (value: ReturnType<typeof output>) => void;
    ortMock.run.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    const hook = await createResearchVoiceMlp(config, "a".repeat(64));
    const tone = (hz: number) => Float32Array.from({ length: 1920 }, (_, i) => .075 * Math.sin(2 * Math.PI * hz * i / 48000));
    const first = hook.prepare(tone(300), 48000);
    const second = hook.prepare(tone(200), 48000);
    const firstRaw = { ...first.rawYin! };
    const predicted = hook.predict(first, { supportStartSample: 0, supportEndSampleExclusive: 1920, lastRequiredInputSample: 1919, sampleRateHz: 48000 } as VoiceFrameTiming);
    await expect(hook.predict(second, {} as VoiceFrameTiming)).rejects.toThrow("serial");
    hook.reset();
    expect(first.rawYin).toEqual(firstRaw);
    expect(second.rawYin!.rawF0Hz).not.toBe(first.rawYin!.rawF0Hz);
    finish(output());
    const result = await predicted;
    expect(result.rawYin).toEqual(firstRaw);
    expect(result.f0Hz).toBe(firstRaw.rawF0Hz);
    expect(result).toMatchObject({ state: "voiced", candidateIndex: null, correctnessProbability: null,
      candidateHeadRole: "diagnostic-only-unused-by-hybrid", estimatorContract: HYBRID_ESTIMATOR_CONTRACT,
      configurationSha256: "a".repeat(64), modelManifestSha256: config.manifestSha256, checkpointSha256: ORIGINAL_MLP_WEIGHTS_SHA256,
      referenceSample: 960, lastRequiredInputSample: 1919 });
    const afterReset = hook.prepare(tone(200), 48000);
    expect(afterReset.values.at(-1)).toBe(0); // History cannot cross the reset.
    await hook.dispose(); expect(ortMock.release).toHaveBeenCalledOnce();
    expect(() => hook.prepare(tone(200), 48000)).toThrow("active");
  });
  it("requires the frozen original checkpoint and verified configuration before model execution", async () => {
    await expect(createResearchVoiceMlp(fixture())).rejects.toThrow("verified original configuration");
    await expect(createResearchVoiceMlp(fixture("b".repeat(64)), "a".repeat(64))).rejects.toThrow("frozen original");
    expect(ortMock.run).not.toHaveBeenCalled();
  });
});
