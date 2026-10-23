/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/wasm";

function exactFloatOutput(tensor: ort.Tensor | undefined, name: string, expectedLength: number): Float32Array {
  if (!tensor || tensor.type !== "float32" || !(tensor.data instanceof Float32Array) ||
      tensor.dims.length !== 2 || tensor.dims[0] !== 1 || tensor.dims[1] !== expectedLength ||
      tensor.data.length !== expectedLength) {
    throw new Error(`Output contract mismatch for ${name}: expected float32 [1,${expectedLength}].`);
  }
  return tensor.data;
}
function requireVector(value: unknown, expectedLength: number, name: string): asserts value is number[] {
  if (!Array.isArray(value) || value.length !== expectedLength || !value.every(Number.isFinite)) {
    throw new Error(`Fixture contract mismatch for ${name}.`);
  }
}
async function checkedFetch(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Model artifact HTTP ${response.status}: ${url}`);
  return response;
}
const sha256 = async (bytes: ArrayBuffer) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  value => value.toString(16).padStart(2, "0")).join("");

self.addEventListener("message", async (event: MessageEvent) => {
  if (event.data?.type !== "run-export-parity") return;
  let session: ort.InferenceSession | undefined;
  try {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = new URL("/.local/voice-runtime/", self.location.origin).href;
    const root = new URL("/.local/voice-spike/", self.location.origin).href;
    const manifest = await (await checkedFetch(root + "manifest.json")).json();
    const { feature_count: featureCount, candidate_count: candidateCount, hidden_size: hiddenSize } = manifest.config;
    if (![featureCount, candidateCount, hiddenSize].every(value => Number.isSafeInteger(value) && value > 0 && value <= 1024)) {
      throw new Error("Invalid model dimensions.");
    }
    const bytes = await (await checkedFetch(root + "model.onnx")).arrayBuffer();
    const hash = await sha256(bytes);
    if (hash !== manifest.artifacts.onnx.sha256) throw new Error("Model digest mismatch.");
    const fixtureBytes = await (await checkedFetch(root + "parity-fixtures.json")).arrayBuffer();
    const fixtureHash = await sha256(fixtureBytes);
    if (fixtureHash !== manifest.artifacts.parity_fixtures.sha256) throw new Error("Fixture digest mismatch.");
    const fixtureSet = JSON.parse(new TextDecoder().decode(fixtureBytes));
    if (!Array.isArray(fixtureSet.fixtures) || fixtureSet.fixtures.length === 0 ||
        ![fixtureSet.atol, fixtureSet.rtol].every(value => Number.isFinite(value) && value >= 0)) {
      throw new Error("Invalid parity fixture set.");
    }
    const began = performance.now();
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    const loadMs = performance.now() - began;
    let state = new Float32Array(hiddenSize);
    let maximumAbsoluteError = 0;
    let decisionMismatches = 0;
    let allMaskedSteps = 0;
    const rows = [];
    for (const fixture of fixtureSet.fixtures) {
      requireVector(fixture.features, featureCount, "features");
      requireVector(fixture.voicingLogit, 1, "voicingLogit");
      requireVector(fixture.candidateLogits, candidateCount, "candidateLogits");
      requireVector(fixture.stateOut, hiddenSize, "stateOut");
      if (!Array.isArray(fixture.candidateMask) || fixture.candidateMask.length !== candidateCount ||
          !fixture.candidateMask.every((value: unknown) => typeof value === "boolean")) throw new Error("Invalid candidate mask.");
      const inputs = {
        features: new ort.Tensor("float32", Float32Array.from(fixture.features), [1, featureCount]),
        candidate_mask: new ort.Tensor("bool", Uint8Array.from(fixture.candidateMask, (value: boolean) => Number(value)), [1, candidateCount]),
        state_in: new ort.Tensor("float32", state, [1, hiddenSize])
      };
      let output: ort.InferenceSession.OnnxValueMapType | undefined;
      try {
        output = await session.run(inputs);
        const actualVoicing = exactFloatOutput(output.voicing_logit, "voicing_logit", 1);
        const actualCandidates = exactFloatOutput(output.candidate_logits, "candidate_logits", candidateCount);
        const actualState = exactFloatOutput(output.state_out, "state_out", hiddenSize);
        for (const [name, actual, expected] of [
          ["voicing_logit", actualVoicing, fixture.voicingLogit],
          ["candidate_logits", actualCandidates, fixture.candidateLogits],
          ["state_out", actualState, fixture.stateOut]
        ] as const) {
          for (let index = 0; index < actual.length; index += 1) {
            const error = Math.abs(actual[index] - expected[index]);
            maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
            if (!Number.isFinite(actual[index]) || error > fixtureSet.atol + fixtureSet.rtol * Math.abs(expected[index])) {
              throw new Error(`Export parity failed at step ${fixture.index} ${name}[${index}].`);
            }
          }
        }
        const select = (logits: ArrayLike<number>) => {
          let selected: number | null = null;
          let best = -Infinity;
          fixture.candidateMask.forEach((valid: boolean, index: number) => {
            if (valid && logits[index] > best) { best = logits[index]; selected = index; }
          });
          return selected;
        };
        const selected = select(actualCandidates);
        decisionMismatches += Number(selected !== select(fixture.candidateLogits));
        allMaskedSteps += Number(selected === null);
        state = Float32Array.from(actualState);
        rows.push({ index: fixture.index, voicingLogit: Array.from(actualVoicing), candidateLogits: Array.from(actualCandidates),
          stateOut: Array.from(state), selectedCandidate: selected });
      } finally {
        Object.values(inputs).forEach(tensor => tensor.dispose());
        if (output) Object.values(output).forEach(tensor => tensor.dispose());
      }
    }
    self.postMessage({ status: "complete",
      evidence: manifest.training_performed ? "provided-model-browser-export-parity" : "random-weight-browser-export-spike",
      modelSha256: hash, fixtureSha256: fixtureHash, featureSchema: manifest.feature_schema,
      trainingPerformed: manifest.training_performed === true,
      runtime: "onnxruntime-web@1.23.2", executionProvider: "wasm", threads: 1, loadMs,
      outputContract: { dtype: "float32", voicingLogitShape: [1, 1], candidateLogitShape: [1, candidateCount], stateShape: [1, hiddenSize] },
      steps: rows.length, maximumAbsoluteError, decisionMismatches, allMaskedSteps, rows,
      limitation: "Synthetic parity inputs establish arithmetic equivalence only; no learned accuracy or full-pipeline timing claim." });
  } catch (error) { self.postMessage({ status: "failed", error: String(error) }); }
  finally { await session?.release(); }
});
