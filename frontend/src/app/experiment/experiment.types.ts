export type ExperimentModelId =
  | 'speecht5-pretrained'
  | 'speecht5-v1-baseline'
  | 'speecht5-v2-term-balance'
  | 'speecht5-v3-replay';

export interface ExperimentIntegrity {
  readonly status: 'passed';
  readonly datasetLockVerified: boolean;
  readonly checkpointHashesVerified: boolean;
  readonly finalArtifactHashesVerified: boolean;
  readonly allPodsTerminated: boolean;
  readonly checkpointCount: number;
  readonly stabilityGateDecision: string;
}
export interface ExperimentTrainingConfig {
  readonly precision: string;
  readonly physicalBatchSize: number;
  readonly gradientAccumulationSteps: number;
  readonly effectiveBatchSize: number;
  readonly maximumSteps: number;
  readonly nominalEpochs: number;
  readonly learningRate: number;
  readonly warmupSteps: number;
  readonly evaluationSteps: number;
  readonly maximumGradientNorm: number;
  readonly gradientCheckpointing: boolean;
  readonly earlyStoppingPatience: number;
  readonly earlyStoppingThreshold: number;
  readonly seed: number;
}

export interface ExperimentDataAudit {
  readonly status: 'passed';
  readonly uniqueAudioFiles: number;
  readonly audioVerificationFailureCount: number;
  readonly leakageIntersectionCount: number;
  readonly leakageIdentityFields: readonly string[];
  readonly sharedEvaluationManifests: boolean;
  readonly scheduleBlockSize: number;
  readonly sourceManifestSha256: Readonly<Record<string, string>>;
  readonly builderSha256: string;
  readonly variantConfigSha256: string;
}

export interface ExperimentDatasetStats {
  readonly strategy: string;
  readonly scheduledRows: number;
  readonly uniqueSourceRows: number;
  readonly repeatedExposures: number;
  readonly rowsWithTerms: number;
  readonly durationHours: number;
  readonly uniqueSpeakers: number;
  readonly maximumSpeakerShare: number;
  readonly sourcePoolCounts: Readonly<Record<string, number>>;
  readonly termCategoryOccurrences: Readonly<Record<string, number>>;
  readonly manifestSha256: Readonly<Record<string, string>>;
}

export interface ExperimentLossPoint {
  readonly step: number;
  readonly epoch: number;
  readonly evaluationLoss: number;
}

export interface ExperimentEvaluation {
  readonly caseCount: number;
  readonly failureCount: number;
  readonly domainTermsCorrect: number;
  readonly domainTermsTotal: number;
  readonly domainTermAccuracy: number;
  readonly wordErrorRate: number;
  readonly averageInferenceMs: number;
  readonly averageRealTimeFactor: number;
  readonly peakGpuMemoryMb: number;
  readonly synthesisVerified: boolean;
}

export interface ExperimentVariantReport {
  readonly id: 'v1-baseline' | 'v2-term-balance' | 'v3-replay';
  readonly name: string;
  readonly podId: string;
  readonly startedAt: string;
  readonly finishedAt: string;
  readonly finalStep: number;
  readonly bestStep: number;
  readonly bestValidationLoss: number;
  readonly stoppedEarly: boolean;
  readonly trainingSeconds: number;
  readonly trainingStepsPerSecond: number;
  readonly checkpointSteps: readonly number[];
  readonly estimatedCostUsd: number;
  readonly dataset: ExperimentDatasetStats;
  readonly validationHistory: readonly ExperimentLossPoint[];
  readonly evaluation: ExperimentEvaluation;
  readonly selectedModelSha256: string;
  readonly artifactManifestSha256: string;
}

export interface ExperimentPretrainedReport {
  readonly id: 'speecht5-pretrained';
  readonly name: string;
  readonly modelId: string;
  readonly revision: string;
  readonly trainingSteps: 0;
  readonly evaluatedAt: string;
  readonly podId: string;
  readonly hardware: string;
  readonly testManifestSha256: string;
  readonly artifactManifestSha256: string;
  readonly evaluation: ExperimentEvaluation;
}

export interface ExperimentIncident {
  readonly id: string;
  readonly variants: readonly string[];
  readonly impact: string;
  readonly resolution: string;
  readonly trainingRestart: boolean;
}

export interface ExperimentReport {
  readonly runId: string;
  readonly experimentId: string;
  readonly headline: string;
  readonly runtimeLabel: string;
  readonly integrity: ExperimentIntegrity;
  readonly training: ExperimentTrainingConfig;
  readonly dataAudit: ExperimentDataAudit;
  readonly sharedSplits: Readonly<Record<string, number>>;
  readonly datasetLockSha256: string;
  readonly configurationSha256: string;
  readonly sourceModelRevisions: Readonly<Record<string, string>>;
  readonly pretrainedControl: ExperimentPretrainedReport;
  readonly variants: readonly ExperimentVariantReport[];
  readonly incidents: readonly ExperimentIncident[];
  readonly trainingResumptions: readonly Readonly<Record<string, unknown>>[];
  readonly totalGpuHours: number;
  readonly estimatedTotalCostUsd: number;
}

export interface ExperimentFixtureTerm {
  readonly text: string;
  readonly canonical: string;
  readonly category: string;
}

export interface ExperimentFixture {
  readonly id: string;
  readonly text: string;
  readonly targetTerms: readonly ExperimentFixtureTerm[];
}

export interface ExperimentFixturePage {
  readonly items: readonly ExperimentFixture[];
  readonly total: number;
  readonly offset: number;
  readonly limit: number;
  readonly manifestSha256: string;
}

export interface ExperimentModelSummary {
  readonly id: ExperimentModelId;
  readonly name: string;
  readonly role: 'pretrained' | 'adapted';
  readonly variant: string;
  readonly runtime: 'PyTorch CPU';
  readonly hosting: 'self-hosted';
  readonly revision: string;
  readonly modelSha256: string | null;
  readonly available: boolean;
  readonly unavailableReason: string | null;
}

export interface ExperimentComparisonRequest {
  readonly mode: 'fixture' | 'custom';
  readonly fixtureId?: string;
  readonly text?: string;
  readonly targetTerms?: readonly string[];
  readonly modelIds: readonly ExperimentModelId[];
  readonly sanitizeText: boolean;
  readonly normalizeText: boolean;
}

export interface ExperimentTermScore {
  readonly correct: readonly string[];
  readonly incorrect: readonly string[];
  readonly correctCount: number;
  readonly totalCount: number;
  readonly accuracy: number;
}

export interface ExperimentRuntimeMetrics {
  readonly modelLoadMs: number;
  readonly inferenceMs: number;
  readonly audioDurationMs: number;
  readonly realTimeFactor: number;
  readonly processMemoryMb: number;
  readonly asrMs: number;
  readonly warm: boolean;
  readonly runtime: 'CPU';
}

export interface ExperimentModelResult {
  readonly modelId: ExperimentModelId;
  readonly status:
    | 'queued'
    | 'loading'
    | 'synthesizing'
    | 'audio_ready'
    | 'transcribing'
    | 'success'
    | 'failure';
  readonly originalText: string;
  readonly normalizedText: string;
  readonly audioUrl: string | null;
  readonly transcript: string | null;
  readonly targetTerms: ExperimentTermScore | null;
  readonly wordErrorRate: number | null;
  readonly metrics: ExperimentRuntimeMetrics | null;
  readonly error: string | null;
}

export interface ExperimentComparisonJob {
  readonly id: string;
  readonly mode: 'fixture' | 'custom';
  readonly stage:
    | 'queued'
    | 'preprocessing'
    | 'loading_model'
    | 'synthesizing'
    | 'audio_ready'
    | 'transcribing'
    | 'scoring'
    | 'completed'
    | 'completed_with_failures'
    | 'failed'
    | 'cancelled';
  readonly progressPercent: number;
  readonly originalText: string;
  readonly normalizedText: string | null;
  readonly targetTerms: readonly string[];
  readonly sanitizeText: boolean;
  readonly normalizeText: boolean;
  readonly modelIds: readonly ExperimentModelId[];
  readonly results: readonly ExperimentModelResult[];
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly completedAt: string | null;
  readonly error: string | null;
}
