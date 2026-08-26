export type ModelVariant = 'fp32' | 'quantized';

export interface SynthesisRequest {
  readonly text: string;
  readonly modelId: string;
  readonly voiceId: string;
  readonly variant: ModelVariant;
}

export interface InferenceMetrics {
  readonly latencyMs: number;
  readonly realTimeFactor: number;
  readonly memoryMb: number;
  readonly coldStart: boolean;
  readonly modelId: string;
  readonly variant: ModelVariant;
}

export interface SynthesisResult {
  readonly status: 'mock' | 'ok';
  readonly model: string;
  readonly text: string;
  readonly audioUrl: string | null;
}

export interface ModelSummary {
  readonly id: string;
  readonly displayName: string;
  readonly voices: readonly string[];
  readonly variants: readonly ModelVariant[];
  readonly modelVersion: string;
  readonly runtime: string;
  readonly hosting: string;
  readonly externalInferenceApis: readonly string[];
  readonly available: boolean;
}

export interface ModelSelection {
  readonly modelId: string;
  readonly variant: ModelVariant;
}
