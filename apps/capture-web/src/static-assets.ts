import { z } from "zod";

const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/u);
const relativeAssetSchema = z.object({
  path: z.string().min(1).refine(
    (path) =>
      !path.startsWith("/") &&
      !path.includes("\\") &&
      !path.includes("..") &&
      !/^[a-z][a-z\d+.-]*:/iu.test(path),
    "Asset paths must remain below the application base URL."
  ),
  sha256: sha256Schema
});

const manifestEnvelopeShape = {
  schemaVersion: z.literal("phenometric.static-assets.v1"),
} as const;

const voiceStaticAssetManifestSchema = z.object({
  ...manifestEnvelopeShape,
  assets: z.object({
    voiceWorklet: relativeAssetSchema
  }).passthrough()
});

const faceStaticAssetManifestSchema = z.object({
  ...manifestEnvelopeShape,
  assets: z.object({
    faceModel: relativeAssetSchema,
    visionWasmScript: relativeAssetSchema,
    visionWasm: relativeAssetSchema,
    visionWasmSimdScript: relativeAssetSchema,
    visionWasmSimd: relativeAssetSchema,
    visionWasmNoSimdScript: relativeAssetSchema,
    visionWasmNoSimd: relativeAssetSchema
  }).passthrough()
});

export const staticAssetManifestSchema = z.object({
  ...manifestEnvelopeShape,
  assets: z.object({
    faceModel: relativeAssetSchema,
    voiceWorklet: relativeAssetSchema,
    visionWasmScript: relativeAssetSchema,
    visionWasm: relativeAssetSchema,
    visionWasmSimdScript: relativeAssetSchema,
    visionWasmSimd: relativeAssetSchema,
    visionWasmNoSimdScript: relativeAssetSchema,
    visionWasmNoSimd: relativeAssetSchema
  })
});

export type StaticAssetManifest = z.infer<typeof staticAssetManifestSchema>;
type VoiceStaticAssetManifest = z.infer<typeof voiceStaticAssetManifestSchema>;
type FaceStaticAssetManifest = z.infer<typeof faceStaticAssetManifestSchema>;

export interface ResolvedVoiceStaticAssets {
  readonly manifest: VoiceStaticAssetManifest;
  readonly voiceWorkletUrl: string;
}

export interface ResolvedFaceStaticAssets {
  readonly manifest: FaceStaticAssetManifest;
  readonly faceModelUrl: string;
  readonly mediaPipeRootUrl: string;
}

function baseUrl(documentBase: string): URL {
  return new URL("./", documentBase);
}

export function resolveAssetUrl(path: string, documentBase: string): string {
  const base = baseUrl(documentBase);
  const resolved = new URL(path, base);
  if (
    resolved.origin !== base.origin ||
    !resolved.pathname.startsWith(base.pathname)
  ) {
    throw new Error("asset-path-outside-application-base");
  }
  return resolved.href;
}

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

export async function verifyStaticAsset(
  asset: z.infer<typeof relativeAssetSchema>,
  documentBase: string,
  fetcher: typeof fetch = fetch
): Promise<string> {
  const url = resolveAssetUrl(asset.path, documentBase);
  try {
    const response = await fetcher(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`asset-unavailable:${asset.path}`);
    const actual = await sha256Hex(await response.arrayBuffer());
    if (actual !== asset.sha256) {
      throw new Error(`asset-integrity-failed:${asset.path}`);
    }
    return url;
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("asset-")) {
      throw error;
    }
    throw new Error(`asset-integrity-failed:${asset.path}`, { cause: error });
  }
}

async function fetchStaticAssetManifest(
  documentBase: string,
  fetcher: typeof fetch = fetch
): Promise<unknown> {
  const manifestUrl = resolveAssetUrl("asset-manifest.json", documentBase);
  try {
    const response = await fetcher(manifestUrl, { cache: "no-store" });
    if (!response.ok) throw new Error("asset-manifest-unavailable");
    return await response.json();
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("asset-")) {
      throw error;
    }
    throw new Error("asset-integrity-failed:asset-manifest.json", {
      cause: error
    });
  }
}

export async function loadStaticAssetManifest(
  documentBase: string,
  fetcher: typeof fetch = fetch
): Promise<StaticAssetManifest> {
  try {
    return staticAssetManifestSchema.parse(
      await fetchStaticAssetManifest(documentBase, fetcher)
    );
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("asset-")) {
      throw error;
    }
    throw new Error("asset-integrity-failed:asset-manifest.json", {
      cause: error
    });
  }
}

async function loadLaneStaticAssetManifest<T>(
  documentBase: string,
  fetcher: typeof fetch,
  schema: z.ZodType<T>
): Promise<T> {
  try {
    return schema.parse(await fetchStaticAssetManifest(documentBase, fetcher));
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("asset-")) {
      throw error;
    }
    throw new Error("asset-integrity-failed:asset-manifest.json", {
      cause: error
    });
  }
}

export async function loadAndVerifyVoiceStaticAssets(
  documentBase: string,
  fetcher: typeof fetch = fetch
): Promise<ResolvedVoiceStaticAssets> {
  const manifest = await loadLaneStaticAssetManifest(
    documentBase,
    fetcher,
    voiceStaticAssetManifestSchema
  );
  const voiceWorkletUrl = await verifyStaticAsset(
    manifest.assets.voiceWorklet,
    documentBase,
    fetcher
  );
  return Object.freeze({
    manifest,
    voiceWorkletUrl
  });
}

export async function loadAndVerifyFaceStaticAssets(
  documentBase: string,
  fetcher: typeof fetch = fetch
): Promise<ResolvedFaceStaticAssets> {
  const manifest = await loadLaneStaticAssetManifest(
    documentBase,
    fetcher,
    faceStaticAssetManifestSchema
  );
  const faceAssetNames = [
    "faceModel",
    "visionWasmScript",
    "visionWasm",
    "visionWasmSimdScript",
    "visionWasmSimd",
    "visionWasmNoSimdScript",
    "visionWasmNoSimd"
  ] as const;
  const verified = await Promise.all(
    faceAssetNames.map(async (name) => [
      name,
      await verifyStaticAsset(manifest.assets[name], documentBase, fetcher)
    ] as const)
  );
  const verifiedUrls = Object.fromEntries(verified) as Record<
    (typeof faceAssetNames)[number],
    string
  >;
  return Object.freeze({
    manifest,
    faceModelUrl: verifiedUrls.faceModel,
    mediaPipeRootUrl: resolveAssetUrl("mediapipe", documentBase)
  });
}
