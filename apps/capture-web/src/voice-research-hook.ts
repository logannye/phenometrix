import type { VoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";
import type { VoiceFrameTiming } from "./voice-stream-timing.js";

/** Research outputs are separate from VoiceSignalFrameV1 and never qualify clinical measurements. */
export interface VoiceResearchPrediction {
  schemaVersion: "phenometrix.research-voice-prediction.v1";
  qualification: "unqualified-functional-fixture";
  modelSha256: string;
  featureSchema: "phenometrix.voice-pitch-features.v2";
  measurementDelayMs: 20;
  referenceSample: number;
  supportStartSample: number;
  supportEndSampleExclusive: number;
  lastRequiredInputSample: number;
  state: "voiced" | "unvoiced" | "unavailable";
  f0Hz: number | null;
  candidateIndex: number | null;
  voicingLogit: number;
  voicingProbability: number;
  candidateLogits: number[];
  candidateProbabilities: number[];
  correctnessProbability: null;
  probabilitySemantics: "uncalibrated-model-probabilities-not-correctness";
  inferenceServiceMs: number;
  modelReadyAtMs?: number;
  sampleAvailabilityToModelReadyMs?: number | null;
  rawSampleAvailabilityToModelReadyMs?: number;
  targetReferenceToModelReadyMs?: number | null;
  elapsedFromFrameStartMs?: number;
  /** Bounded acquisition-only parity evidence on the separate research route. */
  featureValues: number[];
  candidateHz: Array<number | null>;
  candidateMask: boolean[];
}

export interface VoiceResearchHook {
  prepare(samples: Float32Array, sampleRateHz: number): VoicePitchFeaturesV2;
  predict(features: VoicePitchFeaturesV2, timing: VoiceFrameTiming): Promise<VoiceResearchPrediction>;
  reset(): void;
  dispose(): Promise<void>;
}

export interface VoiceResearchModelConfig {
  schemaVersion: "phenometrix.research-voice-model-config.v1";
  qualification: "unqualified-functional-fixture";
  manifestUrl: string;
  manifestSha256: string;
  modelUrl: string;
  modelSha256: string;
  wasmUrl: string;
  wasmSha256: string;
  moduleUrl: string;
  moduleSha256: string;
}
