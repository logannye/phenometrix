import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Run after both Vite builds. Research .local assets must never be release assets.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "public");
const outputs = ["dist", "dist-embedded"];
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const files = [];
function visit(relative = "") {
  for (const name of readdirSync(join(source, relative))) {
    if (relative === "" && name === ".local") continue;
    const path = join(relative, name);
    if (statSync(join(source, path)).isDirectory()) visit(path);
    else files.push(path);
  }
}
visit();
const identities = Object.fromEntries(files.sort().map(path => [path, digest(readFileSync(join(source, path)))]));
for (const output of outputs) {
  if (existsSync(join(root, output, ".local"))) throw new Error(`${output} contains local research assets`);
  for (const [path, expected] of Object.entries(identities)) {
    const target = join(root, output, path);
    if (!existsSync(target) || digest(readFileSync(target)) !== expected) {
      throw new Error(`${output}/${path} is missing or differs from the public source`);
    }
  }
}
console.log(JSON.stringify({ status: "passed", outputs, verifiedPublicFiles: files.length,
  researchAssetsExcluded: true, publicFileSha256: identities }));
