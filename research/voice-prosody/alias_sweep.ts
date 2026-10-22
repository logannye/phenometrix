import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { replayVoicePcmV1 } from "../../apps/capture-web/src/voice-replay.js";
import { VOICE_PITCH_FEATURE_NAMES_V2 } from "../../apps/capture-web/src/voice-pitch-features-v2.js";
const [planPath, outputPath] = process.argv.slice(2);
if (!planPath || !outputPath || process.argv.length !== 4) throw new Error("Expected plan and fresh output.");
const plan = JSON.parse(readFileSync(planPath, "utf8"));
if (plan.schema !== "phenometrix-subharmonic-diagnostic-plan.v1" || plan.scenarios.length !== 168) throw new Error("Unknown frozen plan.");
const recordings = plan.scenarios.map((scenario: { id: string; hz: number; phase: number; harmonics: number[] }) => {
  const pcm = new Float32Array(19200);
  for (let i = 0; i < pcm.length; i++) {
    const phase = 2 * Math.PI * scenario.hz * i / 48000 + scenario.phase;
    pcm[i] = .075 * scenario.harmonics.reduce((sum, amplitude, h) => sum + amplitude * Math.sin((h + 1) * phase), 0);
  }
  const frames = replayVoicePcmV1(pcm, { sampleRateHz: 48000, featureVersion: "v2" });
  return { id: scenario.id, pcmSha256: createHash("sha256").update(new Uint8Array(pcm.buffer)).digest("hex"),
    pcmBase64: Buffer.from(pcm.buffer).toString("base64"), frames: frames.map(frame => ({ end: frame.referenceSample,
      values: frame.features.values, candidates: frame.features.candidates, baseline: frame.features.baseline })) };
});
writeFileSync(outputPath, JSON.stringify({ schema: "phenometrix-subharmonic-acquisition.v1", featureNames: VOICE_PITCH_FEATURE_NAMES_V2, recordings }) + "\n", { flag: "wx" });
console.log(JSON.stringify({ recordings: recordings.length, frames: recordings.reduce((n: number, r: { frames: unknown[] }) => n + r.frames.length, 0) }));
