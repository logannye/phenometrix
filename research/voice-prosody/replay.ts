import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { replayVoicePcmV1 } from "../../apps/capture-web/src/voice-replay.js";
import { VOICE_PITCH_FEATURE_NAMES_V2 } from "../../apps/capture-web/src/voice-pitch-features-v2.js";
import { VOICE_PITCH_FEATURE_NAMES } from "../../apps/capture-web/src/voice-pitch-features.js";
import { makeVoiceMechanismFixture } from "./fixtures.js";
import { scoreVoiceCandidateOracleV1 } from "./oracle.js";

/** Run with node --import tsx ../../research/voice-prosody/replay.ts from capture-web. */
const args = process.argv.slice(2);
const option = (name: string) => {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  if (!args[index + 1] || args[index + 1].startsWith("--")) throw new Error(`Missing ${name} value.`);
  return args[index + 1];
};
const featureVersion = option("--feature-version") ?? "v1";
if (featureVersion !== "v1" && featureVersion !== "v2") throw new Error("Unknown feature version.");
const featureNames = featureVersion === "v2" ? VOICE_PITCH_FEATURE_NAMES_V2 : VOICE_PITCH_FEATURE_NAMES;
const inputPath = option("--input-pcm-json");
const outputPath = option("--output");
const f32Path = option("--input-f32");
const rateText = option("--sample-rate");
if (inputPath && f32Path) throw new Error("Choose one PCM input format.");
const known = new Set(["--input-pcm-json", "--output", "--input-f32", "--sample-rate", "--feature-version"]);
for (let i = 0; i < args.length; i += 2) if (!known.has(args[i])) throw new Error(`Unknown option ${args[i]}.`);
if (inputPath || f32Path) {
  const bytes = readFileSync((inputPath ?? f32Path)!);
  if (f32Path && (bytes.length % 4 !== 0 || !rateText)) throw new Error("Raw PCM requires little-endian float32 bytes and --sample-rate.");
  const input = inputPath ? JSON.parse(bytes.toString("utf8")) : {
    sampleRateHz: Number(rateText),
    samples: Array.from({ length: bytes.length / 4 }, (_, i) => bytes.readFloatLE(i * 4))
  };
  if (Object.keys(input).sort().join(",") !== "sampleRateHz,samples" || !Array.isArray(input.samples) ||
      input.samples.some((value: unknown) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error("PCM input must contain only sampleRateHz and finite samples; references are scorer-only.");
  }
  const pcm = Float32Array.from(input.samples);
  const frames = replayVoicePcmV1(pcm, { sampleRateHz: input.sampleRateHz, featureVersion });
  const result = { schemaVersion: "phenometrix.voice-feature-export.v1", inputSha256:
    createHash("sha256").update(bytes).digest("hex"), inputSampleCount: pcm.length,
    featureNames, featureVersion, frames };
  if (!outputPath) throw new Error("Feature export requires a fresh --output path.");
  writeFileSync(outputPath, JSON.stringify(result) + "\n", { flag: "wx" });
  console.log(JSON.stringify({ outputPath, frameCount: frames.length, inputSampleCount: pcm.length }));
} else {
  if (rateText) throw new Error("--sample-rate is only accepted with --input-f32.");
  const conditions = ["steady", "harmonic-change", "source-step", "source-glide", "silence"] as const;
  const results = conditions.map(condition => {
    const fixture = makeVoiceMechanismFixture(condition);
    // Reference construction occurs after prediction; it cannot change inference.
    const frames = replayVoicePcmV1(fixture.pcm, { sampleRateHz: fixture.sampleRateHz, featureVersion }, [1, 127, 960, 37, 2051]);
    const references = frames.map(frame => {
      const hz = fixture.referenceF0HzAtSample(frame.lastRequiredInputSample);
      return { referenceIndex: frame.referenceIndex, valid: true, voiced: hz > 0, f0Hz: hz > 0 ? hz : null };
    });
    return { fixtureId: fixture.id, frameCount: frames.length, sampleRateHz: fixture.sampleRateHz,
      sampleCount: fixture.pcm.length, pcmSha256: createHash("sha256").update(new Uint8Array(fixture.pcm.buffer)).digest("hex"),
      ...scoreVoiceCandidateOracleV1(frames, references, fixture.pcm.length) };
  });
  const result = { schemaVersion: "phenometrix.voice-synthetic-smoke.v1", evidence:
    "Analytic synthetic diagnostics only; no natural-speech, browser timing, training or calibration claim.", results };
  if (outputPath) writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(result, null, 2));
}
