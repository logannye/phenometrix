import { mountTreatmentResponsePanel } from "./panel.js";
import { syntheticRun, SYNTHETIC_SCOPE } from "./synthetic.js";
const run = await syntheticRun();
const panel = mountTreatmentResponsePanel(document.querySelector<HTMLElement>("#evidence")!, {
  scope: SYNTHETIC_SCOPE, load: async () => ({ run })
});
await panel.refresh();
