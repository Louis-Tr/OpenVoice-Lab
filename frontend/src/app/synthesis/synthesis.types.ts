export type ModelVariant = 'fp32' | 'quantized';

export interface SynthesisRequest {
  readonly text: string;
  readonly modelId: string;
  readonly voiceId: string;
}

export interface InferenceMetrics {
  readonly modelLoadMs: number;
  readonly inferenceMs: number;
  readonly audioDurationMs: number;
  readonly realTimeFactor: number;
  readonly memoryMb: number;
  readonly warm: boolean;
  readonly modelVariant: ModelVariant;
}

export interface SynthesisResult {
  readonly status: 'mock' | 'ok';
  readonly model: string;
  readonly text: string;
  readonly audioUrl: string | null;
  readonly metrics: InferenceMetrics;
}

export interface ModelSummary {
  readonly id: string;
  readonly name: string;
  readonly precision: 'FP32' | 'INT8';
  readonly variant: ModelVariant;
  readonly voices: readonly string[];
  readonly modelVersion: string;
  readonly runtime: string;
  readonly hosting: string;
  readonly externalInferenceApis: readonly string[];
  readonly available: boolean;
}

export interface ModelSelection {
  readonly modelId: string;
}
