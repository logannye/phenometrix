/** Input-only research kernel. Historical v2 sources remain immutable. */
import { readFileSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { createInterface } from "node:readline";
import { decimate } from "../../apps/capture-web/src/dsp-primitives.js";

export const POLICY = "shortest-admitted-full40-v1";
export const MAX_SAMPLES = 48_000 * 120;
export const MAX_ENDS = 12_000;
export interface Peak { hz: number; confidence: number; lagSamples: number; log2HzOver50: number }

/** Exact v2 full40 preprocessing, local-maximum admission, and interpolation. */
export function full40Peaks(samples: Float32Array): Peak[] {
  if (samples.length !== 1920) throw new Error("Complete 40 ms support required.");
  let mean = 0;
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new Error("Finite PCM required.");
    mean += sample;
  }
  mean /= samples.length;
  const centered = Float32Array.from(samples, sample => sample - mean);
  const input = decimate(centered, 48000, 8000);
  const minimumLag = Math.ceil(8000 / 700);
  const maximumLag = Math.min(input.length - 2, Math.floor(8000 / 50));
  const curve = new Float64Array(maximumLag + 2);
  for (let lag = minimumLag - 1; lag <= maximumLag + 1; lag += 1) {
    let cross = 0, left = 0, right = 0;
    for (let i = 0; i < input.length - lag; i += 1) {
      cross += input[i] * input[i + lag];
      left += input[i] * input[i];
      right += input[i + lag] * input[i + lag];
    }
    curve[lag] = cross / Math.max(1e-12, Math.sqrt(left * right));
  }
  const result: Peak[] = [];
  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    if (curve[lag] < 0.35 || curve[lag] < curve[lag - 1] || curve[lag] < curve[lag + 1]) continue;
    const curvature = curve[lag - 1] - 2 * curve[lag] + curve[lag + 1];
    const delta = curvature >= -1e-12 ? 0 : Math.max(-0.5, Math.min(0.5,
      0.5 * (curve[lag - 1] - curve[lag + 1]) / curvature));
    const refinedLag = lag + delta;
    const hz = 8000 / refinedLag;
    if (hz < 50 || hz > 700) continue;
    result.push({ hz, confidence: Math.min(1, Math.max(0, curve[lag])),
      lagSamples: refinedLag, log2HzOver50: Math.log2(hz / 50) });
  }
  return result;
}

export function inspectRequest(request: unknown) {
  if (request === null || typeof request !== "object" || Array.isArray(request)) throw new Error("Object request required.");
  const r = request as Record<string, unknown>;
  if (Object.keys(r).sort().join(",") !== "ends,id,pcm_path,sample_rate_hz" ||
      !Number.isSafeInteger(r.id) || (r.id as number) < 0 || r.sample_rate_hz !== 48000 ||
      typeof r.pcm_path !== "string" || !r.pcm_path.endsWith(".f32le") ||
      !Array.isArray(r.ends) || r.ends.length > MAX_ENDS) throw new Error("Invalid request fields.");
  const size = statSync(r.pcm_path).size;
  if (size === 0 || size % 4 || size / 4 > MAX_SAMPLES) throw new Error("PCM size exceeds declared bound.");
  const bytes = readFileSync(r.pcm_path);
  if (bytes.length === 0 || bytes.length % 4 || bytes.length / 4 > MAX_SAMPLES) throw new Error("PCM size exceeds declared bound.");
  const pcm = new Float32Array(bytes.length / 4);
  for (let i = 0; i < pcm.length; i++) {
    pcm[i] = bytes.readFloatLE(i * 4);
    if (!Number.isFinite(pcm[i])) throw new Error("Finite PCM required.");
  }
  let previous = -1;
  for (const end of r.ends) {
    if (!Number.isSafeInteger(end) || end < 1920 || end > pcm.length || end <= previous) throw new Error("Invalid complete-support endpoint.");
    previous = end;
  }
  const old: (Peak | null)[] = [], revised: (Peak | null)[] = [];
  for (const end of r.ends) {
    const peaks = full40Peaks(pcm.subarray(end - 1920, end));
    old.push([...peaks].sort((a, b) => b.confidence - a.confidence || a.lagSamples - b.lagSamples)[0] ?? null);
    revised.push([...peaks].sort((a, b) => a.lagSamples - b.lagSamples)[0] ?? null);
  }
  return { id: r.id, ok: true, policy: POLICY, pcm_sha256: createHash("sha256").update(bytes).digest("hex"),
    pcm_sample_count: pcm.length, ends: r.ends, old, revised, rss_bytes: process.memoryUsage().rss };
}

async function serve() {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    let id: unknown = null;
    try {
      if (Buffer.byteLength(line) > 1024 * 1024) throw new Error("Request exceeds 1 MiB.");
      const request = JSON.parse(line); id = request?.id;
      process.stdout.write(JSON.stringify(inspectRequest(request)) + "\n");
    } catch (error) {
      process.stdout.write(JSON.stringify({ id, ok: false, error: String(error) }) + "\n");
    }
  }
}

if (process.argv[1]?.replaceAll("\\", "/").endsWith("/coverage_peaks.ts")) {
  void serve().catch(error => { process.stderr.write(String(error) + "\n"); process.exitCode = 1; });
}
