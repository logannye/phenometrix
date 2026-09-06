import { readFile } from "node:fs/promises";
import { z } from "zod";
import type { FhirSourceConfiguration } from "./service.js";

const Source=z.object({
  sourceSystem:z.string().url(),
  treatmentCodes:z.array(z.object({system:z.string().min(1),code:z.string().min(1),product:z.string().min(1),kind:z.enum(["botulinum-injection","concurrent-treatment"])}).strict()),
  dateOffset:z.string().regex(/^[+-](?:0\d|1[0-3]):[0-5]\d$|^[+-]14:00$/).optional()
}).strict();
/** Configuration is loaded from a deployment-owned local file, never from an upload. */
export async function loadFhirSources(path:string|undefined):Promise<Record<string,FhirSourceConfiguration>>{
  if(!path)return {};
  return z.record(z.string().regex(/^[A-Za-z0-9._:-]{1,160}$/),Source).parse(JSON.parse(await readFile(path,"utf8")));
}
