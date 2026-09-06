#!/usr/bin/env node
/** Verify or rewrite the unilateral facial movement demo profile digest. */
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PROFILE_SOURCE = resolve(
  ROOT,
  "packages/condition-profiles/src/unilateral-facial-movement.ts"
);
const PROFILE_LITERAL = /(const rawProfile = \{[\s\S]*?contentSha256:\s*\n?\s*")[a-f0-9]{64}("\s*,)/;

async function expectedDigest() {
  const [{ UNILATERAL_FACIAL_MOVEMENT_PROFILE: profile }, contracts] =
    await Promise.all([
      import(resolve(ROOT, "packages/condition-profiles/src/index.ts")),
      import(resolve(ROOT, "packages/contracts/src/index.ts"))
    ]);
  return createHash("sha256")
    .update(contracts.conditionDemoProfileDigestInput(profile), "utf8")
    .digest("hex");
}

async function main() {
  const write = process.argv.includes("--write");
  const check = process.argv.includes("--check");
  if (write === check) {
    console.error("Usage: condition-profile-digest.mjs --check | --write");
    process.exit(2);
  }

  const source = await readFile(PROFILE_SOURCE, "utf8");
  const match = PROFILE_LITERAL.exec(source);
  if (!match) throw new Error("Could not locate the condition profile digest literal.");
  const actual = /contentSha256:\s*\n?\s*"([a-f0-9]{64})"/.exec(match[0])?.[1];
  if (!actual) throw new Error("Could not read the condition profile digest literal.");
  const expected = await expectedDigest();

  if (check) {
    if (actual !== expected) {
      console.error(`Condition profile digest is stale: ${actual} != ${expected}`);
      process.exit(1);
    }
    console.log("Condition profile digest is current.");
    return;
  }

  if (actual === expected) {
    console.log(`Condition profile digest already current: ${expected}`);
    return;
  }
  await writeFile(
    PROFILE_SOURCE,
    source.replace(PROFILE_LITERAL, `$1${expected}$2`),
    "utf8"
  );
  console.log(`Condition profile digest updated: ${actual} -> ${expected}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
