import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  loadAndVerifyFaceStaticAssets,
  loadAndVerifyVoiceStaticAssets,
  resolveAssetUrl,
  staticAssetManifestSchema,
  verifyStaticAsset
} from "./static-assets.js";

describe("static asset manifest", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("resolves below an arbitrary deployment base", () => {
    expect(
      resolveAssetUrl(
        "models/face_landmarker.task",
        "https://example.test/tools/phenometrix/index.html"
      )
    ).toBe("https://example.test/tools/phenometrix/models/face_landmarker.task");
  });

  it("rejects root-relative and escaping paths", () => {
    expect(() =>
      staticAssetManifestSchema.shape.assets.shape.faceModel.parse({
        path: "/model",
        sha256: "a".repeat(64)
      })
    ).toThrow();
    for (const path of [
      "https://elsewhere.test/model",
      "\\\\elsewhere.test\\model",
      "%2e%2e/model"
    ]) {
      expect(() =>
        resolveAssetUrl(path, "https://example.test/app/index.html")
      ).toThrow("asset-path-outside-application-base");
    }
  });

  it("withholds an asset whose runtime bytes do not match", async () => {
    vi.stubGlobal("crypto", {
      subtle: { digest: vi.fn().mockResolvedValue(new Uint8Array(32).buffer) }
    });
    const fetcher = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([1]), { status: 200 })
    ) as unknown as typeof fetch;
    await expect(
      verifyStaticAsset(
        { path: "model.task", sha256: "f".repeat(64) },
        "https://example.test/app/",
        fetcher
      )
    ).rejects.toThrow("asset-integrity-failed:model.task");
  });

  it.each([
    [
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("network failed"))
    ],
    [
      "body read",
      vi.fn().mockResolvedValue({
        ok: true,
        arrayBuffer: vi.fn().mockRejectedValue(new TypeError("body failed"))
      })
    ]
  ])("classifies an unexpected %s failure as an unverifiable asset", async (_label, fetcher) => {
    await expect(
      verifyStaticAsset(
        { path: "model.task", sha256: "f".repeat(64) },
        "https://example.test/app/",
        fetcher as unknown as typeof fetch
      )
    ).rejects.toThrow("asset-integrity-failed:model.task");
  });

  it("keeps voice verification independent from unavailable face assets", async () => {
    const emptySha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const manifest = {
      schemaVersion: "phenometric.static-assets.v1",
      assets: Object.fromEntries(
        [
          "faceModel",
          "voiceWorklet",
          "visionWasmScript",
          "visionWasm",
          "visionWasmSimdScript",
          "visionWasmSimd",
          "visionWasmNoSimdScript",
          "visionWasmNoSimd"
        ].map((name) => [
          name,
          { path: `${name}.bin`, sha256: emptySha256 }
        ])
      )
    };
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("asset-manifest.json")) {
        return new Response(JSON.stringify(manifest), {
          status: 200,
          headers: { "content-type": "application/json" }
        });
      }
      if (url.endsWith("voiceWorklet.bin")) {
        return new Response(new Uint8Array(), { status: 200 });
      }
      return new Response(null, { status: 404 });
    }) as unknown as typeof fetch;

    await expect(
      loadAndVerifyVoiceStaticAssets("https://example.test/app/", fetcher)
    ).resolves.toMatchObject({
      voiceWorkletUrl: "https://example.test/app/voiceWorklet.bin"
    });
    await expect(
      loadAndVerifyFaceStaticAssets("https://example.test/app/", fetcher)
    ).rejects.toThrow("asset-unavailable");
  });

  it("keeps voice verification independent from malformed face metadata", async () => {
    const emptySha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const manifest = {
      schemaVersion: "phenometric.static-assets.v1",
      assets: {
        voiceWorklet: { path: "voice-worklet.js", sha256: emptySha256 },
        faceModel: { path: "../outside.task", sha256: "invalid" }
      }
    };
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return url.endsWith("asset-manifest.json")
        ? new Response(JSON.stringify(manifest), { status: 200 })
        : new Response(new Uint8Array(), { status: 200 });
    }) as unknown as typeof fetch;

    await expect(
      loadAndVerifyVoiceStaticAssets("https://example.test/app/", fetcher)
    ).resolves.toMatchObject({
      voiceWorkletUrl: "https://example.test/app/voice-worklet.js"
    });
    await expect(
      loadAndVerifyFaceStaticAssets("https://example.test/app/", fetcher)
    ).rejects.toThrow("asset-integrity-failed:asset-manifest.json");
  });

  it("keeps face verification independent from malformed voice metadata", async () => {
    const emptySha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const faceAssets = Object.fromEntries(
      [
        "faceModel",
        "visionWasmScript",
        "visionWasm",
        "visionWasmSimdScript",
        "visionWasmSimd",
        "visionWasmNoSimdScript",
        "visionWasmNoSimd"
      ].map((name) => [
        name,
        { path: `${name}.bin`, sha256: emptySha256 }
      ])
    );
    const manifest = {
      schemaVersion: "phenometric.static-assets.v1",
      assets: {
        ...faceAssets,
        voiceWorklet: { path: "../outside.js", sha256: "invalid" }
      }
    };
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      return url.endsWith("asset-manifest.json")
        ? new Response(JSON.stringify(manifest), { status: 200 })
        : new Response(new Uint8Array(), { status: 200 });
    }) as unknown as typeof fetch;

    await expect(
      loadAndVerifyFaceStaticAssets("https://example.test/app/", fetcher)
    ).resolves.toMatchObject({
      faceModelUrl: "https://example.test/app/faceModel.bin"
    });
    await expect(
      loadAndVerifyVoiceStaticAssets("https://example.test/app/", fetcher)
    ).rejects.toThrow("asset-integrity-failed:asset-manifest.json");
  });
});
