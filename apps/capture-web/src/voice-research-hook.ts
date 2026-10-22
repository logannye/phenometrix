import type { RawYinEvidence } from "./research-voice-yin.js";
import type { VoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";
import type { VoiceFrameTiming } from "./voice-stream-timing.js";

/** Research outputs are separate from VoiceSignalFrameV1 and never qualify clinical measurements. */
export interface VoiceResearchIdentity {
  configurationSha256: string | null;
  modelManifestSha256: string;
  checkpointSha256: string;
  estimatorContract: "phenometrix.candidate-selecting-mlp.v1" | "phenometrix.mlp-voicing-yin-pitch.v1";
  modelSha256: string;
}
export interface VoiceResearchConfigDocument { text: string; sha256: string; }

export interface VoiceResearchPrediction {
  schemaVersion: "phenometrix.research-voice-prediction.v1";
  qualification: "unqualified-functional-fixture";
  modelSha256: string;
  estimatorContract?: "phenometrix.candidate-selecting-mlp.v1" | "phenometrix.mlp-voicing-yin-pitch.v1";
  configurationSha256?: string | null;
  modelManifestSha256?: string;
  checkpointSha256?: string;
  rawYin?: RawYinEvidence;
  candidateHeadRole?: "pitch-selection" | "diagnostic-only-unused-by-hybrid";
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

export interface VoiceResearchPreparedFrame extends VoicePitchFeaturesV2 { rawYin?: RawYinEvidence; }

export interface VoiceResearchHook {
  readonly provenance?: VoiceResearchIdentity;
  prepare(samples: Float32Array, sampleRateHz: number): VoiceResearchPreparedFrame;
  predict(features: VoiceResearchPreparedFrame, timing: VoiceFrameTiming): Promise<VoiceResearchPrediction>;
  reset(): void;
  dispose(): Promise<void>;
}

export interface VoiceResearchModelConfig {
  schemaVersion: "phenometrix.research-voice-model-config.v1";
  qualification: "unqualified-functional-fixture";
  /** The absent historical setting selects the candidate MLP; production never configures this hook. */
  estimator?: "phenometrix.candidate-selecting-mlp.v1" | "phenometrix.mlp-voicing-yin-pitch.v1";
  manifestUrl: string;
  manifestSha256: string;
  modelUrl: string;
  modelSha256: string;
  wasmUrl: string;
  wasmSha256: string;
  moduleUrl: string;
  moduleSha256: string;
}
