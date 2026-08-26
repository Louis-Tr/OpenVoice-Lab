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
  readonly audioUrl: string;
  readonly durationSeconds: number;
  readonly metrics: InferenceMetrics;
}

export interface ModelSummary {
  readonly id: string;
  readonly displayName: string;
  readonly voices: readonly string[];
  readonly variants: readonly ModelVariant[];
}

