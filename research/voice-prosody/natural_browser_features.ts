/** Reconstruct actual browser supports and history resets outside transport/model code. */
import { readFileSync, writeFileSync } from "node:fs";
import { extractVoicePitchFeaturesV2 } from "../../apps/capture-web/src/voice-pitch-features-v2.js";
import type { VoicePitchFeatureHistoryV1 } from "../../apps/capture-web/src/voice-pitch-features.js";
const [pcmPath, requestPath, outputPath] = process.argv.slice(2);
if (!pcmPath || !requestPath || !outputPath || process.argv.length !== 5) throw new Error("Expected PCM, support requests, create-new output.");
const bytes = readFileSync(pcmPath);
if (bytes.length % 4 || bytes.length > 48000 * 120 * 4) throw new Error("Invalid finite PCM length.");
const pcm = Float32Array.from({ length: bytes.length / 4 }, (_, i) => bytes.readFloatLE(i * 4));
const requests: Array<{ segment: string; end: number }> = JSON.parse(readFileSync(requestPath, "utf8"));
let history: VoicePitchFeatureHistoryV1 | null = null, segment = "", lastEnd = 0;
const rows = requests.map(request => {
  if (!Number.isSafeInteger(request.end) || request.end < 1920 || request.end > pcm.length || request.end <= lastEnd)
    throw new Error("Invalid actual support order.");
  if (request.segment !== segment) { history = null; segment = request.segment; }
  else if (request.end !== lastEnd + 480) throw new Error("Noncontiguous history within actual segment.");
  const result = extractVoicePitchFeaturesV2(pcm.slice(request.end - 1920, request.end), 48000, history);
  history = result.nextHistory; lastEnd = request.end;
  return { segment, end: request.end, values: result.values, mask: result.candidates.map(c => c.valid),
    hz: result.candidates.map(c => c.valid ? c.hz : null) };
});
writeFileSync(outputPath, JSON.stringify(rows) + "\n", { flag: "wx" });
