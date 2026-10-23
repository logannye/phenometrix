import { readFileSync, writeFileSync } from "node:fs";
import { replayVoicePcmV1 } from "../../apps/capture-web/src/voice-replay.js";
const [input, output] = process.argv.slice(2);
if (!input || !output || process.argv.length !== 4) throw new Error("Expected input list and fresh output.");
const records: { id: string; path: string }[] = JSON.parse(readFileSync(input, "utf8"));
const result = records.map(record => {
  const bytes = readFileSync(record.path);
  if (bytes.length % 4) throw new Error("Invalid float32 input.");
  const pcm = new Float32Array(bytes.length / 4);
  for (let i = 0; i < pcm.length; i++) pcm[i] = bytes.readFloatLE(i * 4);
  const frames = replayVoicePcmV1(pcm, { sampleRateHz: 48000, featureVersion: "v2" });
  return { id: record.id, frames: frames.map(f => ({ end: f.referenceSample, values: f.features.values,
    candidates: f.features.candidates, baseline: f.features.baseline })) };
});
writeFileSync(output, JSON.stringify(result) + "\n", { flag: "wx" });
