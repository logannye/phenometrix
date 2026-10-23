import { RawYin48k, combineMlpVoicingAndRawYin, HYBRID_ESTIMATOR_CONTRACT, ORIGINAL_MLP_WEIGHTS_SHA256 } from "./research-voice-yin.js";
import * as ort from "onnxruntime-web/wasm";
import { extractVoicePitchFeaturesV2, VOICE_PITCH_FEATURE_NAMES_V2 } from "./voice-pitch-features-v2.js";
import type { VoicePitchFeatureHistoryV1 } from "./voice-pitch-features.js";
import type { VoiceResearchHook, VoiceResearchModelConfig, VoiceResearchPrediction, VoiceResearchIdentity } from "./voice-research-hook.js";

const sha256 = async (bytes: ArrayBuffer) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  value => value.toString(16).padStart(2, "0")).join("");
async function pinnedBytes(url: string, digest: string): Promise<ArrayBuffer> {
  const resolved = new URL(url, self.location.href);
  if (resolved.origin !== self.location.origin || !/^[a-f0-9]{64}$/.test(digest)) throw new Error("Invalid local research artifact identity.");
  const response = await fetch(resolved);
  if (!response.ok) throw new Error(`Research artifact HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (await sha256(bytes) !== digest) throw new Error("Research artifact digest mismatch.");
  return bytes;
}
function output(tensor: ort.Tensor | undefined, count: number): Float32Array {
  if (!tensor || tensor.type !== "float32" || tensor.dims.length !== 2 || tensor.dims[0] !== 1 ||
      tensor.dims[1] !== count || !(tensor.data instanceof Float32Array) || tensor.data.length !== count ||
      !tensor.data.every(Number.isFinite)) throw new Error("Research model output shape/dtype/finiteness mismatch.");
  return tensor.data;
}

export async function createResearchVoiceMlp(config: VoiceResearchModelConfig, configurationSha256: string | null = null): Promise<VoiceResearchHook> {
  if (config.schemaVersion !== "phenometrix.research-voice-model-config.v1" ||
      config.qualification !== "unqualified-functional-fixture") throw new Error("Research-only model configuration required.");
  if (config.estimator !== undefined && config.estimator !== "phenometrix.candidate-selecting-mlp.v1" && config.estimator !== HYBRID_ESTIMATOR_CONTRACT) {
    throw new Error("Unknown research estimator contract.");
  }
  const hybrid = config.estimator === HYBRID_ESTIMATOR_CONTRACT;
  if (hybrid && !configurationSha256?.match(/^[a-f0-9]{64}$/)) throw new Error("Hybrid requires a verified original configuration document.");
  const manifest = JSON.parse(new TextDecoder().decode(await pinnedBytes(config.manifestUrl, config.manifestSha256)));
  if (manifest.status !== "complete" || manifest.training_performed !== true || manifest.scaling_fitted !== true ||
      manifest.config.kind !== "framewise-mlp" || manifest.config.feature_version !== "v2" ||
      manifest.config.feature_count !== 57 || manifest.config.candidate_count !== 6 || manifest.config.hidden_size !== 32 ||
      manifest.config.projection_size !== 32 ||
      manifest.training_metadata?.measurement_delay_ms !== 20 || manifest.artifacts?.onnx?.sha256 !== config.modelSha256 ||
      JSON.stringify(manifest.feature_names) !== JSON.stringify(VOICE_PITCH_FEATURE_NAMES_V2)) {
    throw new Error("Pinned export does not match the trained 57-feature/6-candidate/20ms MLP contract.");
  }
  if (hybrid && manifest.training_metadata.weights_sha256 !== ORIGINAL_MLP_WEIGHTS_SHA256) {
    throw new Error("Hybrid requires the frozen original MLP checkpoint.");
  }
  const provenance: VoiceResearchIdentity = { configurationSha256, modelManifestSha256: config.manifestSha256,
    checkpointSha256: manifest.training_metadata.weights_sha256, modelSha256: config.modelSha256,
    estimatorContract: hybrid ? HYBRID_ESTIMATOR_CONTRACT : "phenometrix.candidate-selecting-mlp.v1" };
  if (!/^[a-f0-9]{64}$/.test(provenance.checkpointSha256)) throw new Error("Missing checkpoint identity.");
  const [model, wasm, moduleBytes] = await Promise.all([
    pinnedBytes(config.modelUrl, config.modelSha256), pinnedBytes(config.wasmUrl, config.wasmSha256),
    pinnedBytes(config.moduleUrl, config.moduleSha256)
  ]);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  // Blob URLs execute exactly the verified bytes. An explicit WASM locator is
  // required because the Emscripten loader cannot resolve a relative URL against a module blob.
  ort.env.wasm.wasmBinary = undefined;
  const wasmUrl = URL.createObjectURL(new Blob([wasm], { type: "application/wasm" }));
  const moduleUrl = URL.createObjectURL(new Blob([moduleBytes], { type: "text/javascript" }));
  ort.env.wasm.wasmPaths = { mjs: moduleUrl, wasm: wasmUrl };
  let session: ort.InferenceSession;
  try { session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
  finally { URL.revokeObjectURL(moduleUrl); URL.revokeObjectURL(wasmUrl); }
  const yin = hybrid ? new RawYin48k() : null;
  let history: VoicePitchFeatureHistoryV1 | null = null;
  let state = new Float32Array(32);
  let generation = 0;
  let disposed = false;
  let inFlight: Promise<unknown> | null = null;
  return {
    provenance,
    prepare(samples, sampleRateHz) {
      if (disposed || sampleRateHz !== 48_000) throw new Error("Research MLP requires active 48kHz PCM.");
      const features = extractVoicePitchFeaturesV2(samples, sampleRateHz, history);
      history = features.nextHistory;
      if (!yin) return features;
      const began = performance.now();
      const rawYin = yin.process(samples);
      return { ...features, rawYin: { ...rawYin, serviceMs: performance.now() - began } };
    },
    async predict(features, timing) {
      if (disposed || inFlight) throw new Error("Research inference must be active and serial.");
      const predictionGeneration = generation;
      const raw = Float32Array.from(features.values); // Scaling is already embedded in the trained ONNX graph.
      const mask = features.candidates.map(candidate => candidate.valid);
      const inputs = { features: new ort.Tensor("float32", raw, [1, 57]),
        candidate_mask: new ort.Tensor("bool", Uint8Array.from(mask, Number), [1, 6]),
        state_in: new ort.Tensor("float32", state, [1, 32]) };
      let tensors: ort.InferenceSession.OnnxValueMapType | undefined;
      const started = performance.now();
      try {
        inFlight = session.run(inputs);
        tensors = await inFlight as ort.InferenceSession.OnnxValueMapType;
        const voicing = output(tensors.voicing_logit, 1)[0];
        const logits = output(tensors.candidate_logits, 6);
        const nextState = output(tensors.state_out, 32);
        if (!disposed && generation === predictionGeneration) state = Float32Array.from(nextState);
        const probability = voicing >= 0 ? 1 / (1 + Math.exp(-voicing)) : Math.exp(voicing) / (1 + Math.exp(voicing));
        let selected: number | null = null;
        for (let i = 0; i < 6; i += 1) if (mask[i] && (selected === null || logits[i] > logits[selected])) selected = i;
        const maximum = selected === null ? 0 : logits[selected];
        const weights = Array.from(logits, (logit, i) => mask[i] ? Math.exp(logit - maximum) : 0);
        const denominator = weights.reduce((total, weight) => total + weight, 0);
        const candidateProbabilities = weights.map(weight => denominator ? weight / denominator : 0);
        if (hybrid && !features.rawYin) throw new Error("Missing prepared raw YIN evidence.");
        const combined = hybrid ? combineMlpVoicingAndRawYin(probability, features.rawYin!.rawF0Hz) : null;
        const predictionState = combined?.state ?? (probability < .5 ? "unvoiced" : selected === null ? "unavailable" : "voiced");
        const candidateIndex = !hybrid && predictionState === "voiced" ? selected : null;
        return { schemaVersion: "phenometrix.research-voice-prediction.v1", qualification: "unqualified-functional-fixture",
          ...provenance,
          candidateHeadRole: hybrid ? "diagnostic-only-unused-by-hybrid" : "pitch-selection",
          ...(hybrid ? { rawYin: features.rawYin } : {}), featureSchema: "phenometrix.voice-pitch-features.v2", measurementDelayMs: 20,
          referenceSample: timing.supportEndSampleExclusive - timing.sampleRateHz * .02,
          supportStartSample: timing.supportStartSample, supportEndSampleExclusive: timing.supportEndSampleExclusive,
          lastRequiredInputSample: timing.lastRequiredInputSample, state: predictionState,
          f0Hz: combined ? combined.f0Hz : candidateIndex === null ? null : features.candidates[candidateIndex].hz,
          candidateIndex, voicingLogit: voicing, voicingProbability: probability,
          candidateLogits: Array.from(logits), candidateProbabilities, correctnessProbability: null,
          probabilitySemantics: "uncalibrated-model-probabilities-not-correctness", inferenceServiceMs: performance.now() - started,
          featureValues: Array.from(raw), candidateHz: features.candidates.map(candidate => candidate.hz), candidateMask: mask
        } satisfies VoiceResearchPrediction;
      } finally {
        inFlight = null;
        Object.values(inputs).forEach(tensor => tensor.dispose());
        if (tensors) Object.values(tensors).forEach(tensor => tensor.dispose());
      }
    },
    reset() { generation += 1; history = null; state = new Float32Array(32); yin?.clear(); },
    async dispose() {
      disposed = true; generation += 1; history = null;
      try { await inFlight; } catch { /* The worker reports inference failure separately. */ }
      state.fill(0); yin?.clear();
      await session.release();
    }
  };
}
