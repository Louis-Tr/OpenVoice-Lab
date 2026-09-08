export type ModelVariant = 'fp32' | 'fp16' | 'quantized' | 'audio8' | 'pretrained';

export interface SynthesisRequest {
  readonly text: string;
  readonly modelId: string;
  readonly voiceId: string;
  readonly sanitizeText: boolean;
  readonly normalizeText: boolean;
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
  readonly normalizedText: string;
  readonly audioUrl: string | null;
  readonly metrics: InferenceMetrics;
}

export type SynthesisJobStatus =
  | 'pending' | 'loading' | 'generating' | 'saving'
  | 'completed' | 'failed' | 'rejected';

export interface SynthesisJob {
  readonly jobId: string;
  readonly status: SynthesisJobStatus;
  readonly request: SynthesisRequest;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly completedAt: string | null;
  readonly expiresAt: string | null;
  readonly result: SynthesisResult | null;
  readonly error: { readonly statusCode: number; readonly detail: string } | null;
}

export function isTerminalJob(job: SynthesisJob): boolean {
  return ['completed', 'failed', 'rejected'].includes(job.status);
}

export interface ModelSummary {
  readonly maxInputCharacters?: number;
  readonly maxInputTokens?: number | null;
  readonly id: string;
  readonly name: string;
  readonly precision: string;
  readonly variant: ModelVariant;
  readonly voices: readonly string[];
  readonly modelVersion: string;
  readonly runtime: string;
  readonly hosting: string;
  readonly externalInferenceApis: readonly string[];
  readonly available: boolean;
  readonly unavailableReason: string | null;
  readonly description: string;
}

export interface ModelSelection {
  readonly modelId: string;
}
