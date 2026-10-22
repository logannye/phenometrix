import { readFile, mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const run = resolve(root, process.argv[2] ?? "../../research/voice-prosody/runs/delayed-mlp-export");
const destination = resolve(root, "public/.local/voice-mlp");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const manifestBytes = await readFile(resolve(run, "manifest.json"));
const manifest = JSON.parse(manifestBytes);
if (manifest.config.kind !== "framewise-mlp" || manifest.config.feature_version !== "v2" ||
    manifest.training_metadata?.measurement_delay_ms !== 20 || manifest.status !== "complete") {
  throw new Error("Expected frozen trained v2 20ms MLP export; this script does not qualify it.");
}
await mkdir(destination, { recursive: true });
const writePinned = async (name, bytes) => {
  await writeFile(resolve(destination, name), bytes);
  return { url: `/.local/voice-mlp/${name}`, sha256: digest(bytes) };
};
const pinnedManifest = await writePinned("manifest.json", manifestBytes);
const modelBytes = await readFile(resolve(run, manifest.artifacts.onnx.path));
if (digest(modelBytes) !== manifest.artifacts.onnx.sha256) throw new Error("Trained model hash mismatch.");
const model = await writePinned("model.onnx", modelBytes);
const wasm = await writePinned("ort-wasm-simd-threaded.wasm", await readFile(resolve(root,
  "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm")));
const module = await writePinned("ort-wasm-simd-threaded.mjs", await readFile(resolve(root,
  "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs")));
const config = { schemaVersion: "phenometrix.research-voice-model-config.v1",
  qualification: "unqualified-functional-fixture", manifestUrl: pinnedManifest.url, manifestSha256: pinnedManifest.sha256,
  modelUrl: model.url, modelSha256: model.sha256, wasmUrl: wasm.url, wasmSha256: wasm.sha256,
  moduleUrl: module.url, moduleSha256: module.sha256 };
await writeFile(resolve(destination, "config.json"), `${JSON.stringify(config, null, 2)}\n`);
console.log(JSON.stringify({ run, destination, config }));
