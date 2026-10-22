import { readFile, copyFile, mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
const root=resolve(dirname(fileURLToPath(import.meta.url)), "..");
const run=resolve(root,process.argv[2] ?? "../../research/voice-prosody/runs/spike");
const manifest=JSON.parse(await readFile(resolve(run,"manifest.json"),"utf8"));
const destination=resolve(root,"public/.local/voice-spike");
await mkdir(destination,{recursive:true});
for(const artifact of Object.values(manifest.artifacts)){
  const bytes=await readFile(resolve(run,artifact.path));
  if(createHash("sha256").update(bytes).digest("hex")!==artifact.sha256)throw new Error("Research artifact digest mismatch");
  if(!["model.onnx","parity-fixtures.json"].includes(artifact.path))continue;
  await writeFile(resolve(destination,artifact.path),bytes);
}
await copyFile(resolve(run,"manifest.json"),resolve(destination,"manifest.json"));
const runtime=resolve(root,"public/.local/voice-runtime");
await mkdir(runtime,{recursive:true});
const assets=[];
for(const name of ["ort-wasm-simd-threaded.wasm","ort-wasm-simd-threaded.mjs"]){
  const bytes=await readFile(resolve(root,"node_modules/onnxruntime-web/dist",name));
  await writeFile(resolve(runtime,name),bytes);
  assets.push({name,bytes:bytes.length,sha256:createHash("sha256").update(bytes).digest("hex")});
}
await writeFile(resolve(runtime,"manifest.json"),JSON.stringify({version:"1.23.2",assets},null,2)+"\n");
console.log(JSON.stringify({run,destination,runtime,assets}));
