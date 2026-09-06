/** Entry point for authorized telehealth hosts. Importing it never requests media. */
export { createEmbeddedEncounter } from "./embedded-encounter.js";
export type { EmbeddedEncounterOptions } from "./embedded-encounter.js";
export { startIntegratedEncounter } from "./integrated-encounter.js";
export type { IntegratedEncounterOptions, HostEncounterEvent } from "./integrated-encounter.js";
