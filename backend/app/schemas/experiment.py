"""Public contracts for the Stage 11 training experiment interface."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.base import ApiSchema

ExperimentVariantId = Literal["v1-baseline", "v2-term-balance", "v3-replay"]
ExperimentModelId = Literal[
    "speecht5-pretrained",
    "speecht5-v1-baseline",
    "speecht5-v2-term-balance",
    "speecht5-v3-replay",
]
ComparisonStage = Literal[
    "queued",
    "preprocessing",
    "loading_model",
    "synthesizing",
    "audio_ready",
    "transcribing",
    "scoring",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
]


class ExperimentIntegrity(ApiSchema):
    """Compact verification summary for the completed remote experiment."""

    status: Literal["passed"]
    dataset_lock_verified: bool
    checkpoint_hashes_verified: bool
    final_artifact_hashes_verified: bool
    all_pods_terminated: bool
    checkpoint_count: int = Field(ge=0)
    stability_gate_decision: str


class ExperimentTrainingConfig(ApiSchema):
    """Frozen hyperparameters shared by all three Stage 11 variants."""

    precision: str
    physical_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    effective_batch_size: int = Field(ge=1)
    maximum_steps: int = Field(ge=1)
    nominal_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    evaluation_steps: int = Field(ge=1)
    maximum_gradient_norm: float = Field(gt=0)
    gradient_checkpointing: bool
    early_stopping_patience: int = Field(ge=1)
    early_stopping_threshold: float = Field(ge=0)
    seed: int


class ExperimentDataAudit(ApiSchema):
    """Verified source-cleaning and split-isolation evidence."""

    status: Literal["passed"]
    unique_audio_files: int = Field(ge=0)
    audio_verification_failure_count: int = Field(ge=0)
    leakage_intersection_count: int = Field(ge=0)
    leakage_identity_fields: list[str]
    shared_evaluation_manifests: bool
    schedule_block_size: int = Field(ge=1)
    source_manifest_sha256: dict[str, str]
    builder_sha256: str
    variant_config_sha256: str


class ExperimentDatasetStats(ApiSchema):
    """Measured exposure distribution for one training schedule."""

    strategy: str
    scheduled_rows: int = Field(ge=1)
    unique_source_rows: int = Field(ge=1)
    repeated_exposures: int = Field(ge=0)
    rows_with_terms: int = Field(ge=0)
    duration_hours: float = Field(ge=0)
    unique_speakers: int = Field(ge=0)
    maximum_speaker_share: float = Field(ge=0, le=1)
    source_pool_counts: dict[str, int]
    term_category_occurrences: dict[str, int]
    manifest_sha256: dict[str, str]


class ExperimentLossPoint(ApiSchema):
    """One exact validation measurement from the trainer history."""

    step: int = Field(ge=0)
    epoch: float = Field(ge=0)
    evaluation_loss: float = Field(ge=0)


class ExperimentEvaluation(ApiSchema):
    """Historical shared-test metrics measured on an RTX 4090."""

    case_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    domain_terms_correct: int = Field(ge=0)
    domain_terms_total: int = Field(ge=0)
    domain_term_accuracy: float = Field(ge=0, le=1)
    word_error_rate: float = Field(ge=0)
    average_inference_ms: float = Field(ge=0)
    average_real_time_factor: float = Field(ge=0)
    peak_gpu_memory_mb: float = Field(ge=0)
    synthesis_verified: bool


class ExperimentVariantReport(ApiSchema):
    """Training, dataset, and evaluation evidence for one variant."""

    id: ExperimentVariantId
    name: str
    pod_id: str
    started_at: datetime
    finished_at: datetime
    final_step: int = Field(ge=0)
    best_step: int = Field(ge=0)
    best_validation_loss: float = Field(ge=0)
    stopped_early: bool
    training_seconds: float = Field(ge=0)
    training_steps_per_second: float = Field(ge=0)
    checkpoint_steps: list[int]
    estimated_cost_usd: float = Field(ge=0)
    dataset: ExperimentDatasetStats
    validation_history: list[ExperimentLossPoint]
    evaluation: ExperimentEvaluation
    selected_model_sha256: str
    artifact_manifest_sha256: str


class ExperimentPretrainedReport(ApiSchema):
    """Comparable shared-test evidence for the unadapted control."""

    id: Literal["speecht5-pretrained"] = "speecht5-pretrained"
    name: str
    model_id: str
    revision: str
    training_steps: Literal[0] = 0
    evaluated_at: datetime
    pod_id: str
    hardware: str
    test_manifest_sha256: str
    artifact_manifest_sha256: str
    evaluation: ExperimentEvaluation


class ExperimentIncident(ApiSchema):
    """Transparent record of controller interruptions during Stage 11."""

    id: str
    variants: list[str]
    impact: str
    resolution: str
    training_restart: bool


class ExperimentReport(ApiSchema):
    """Complete read-only Stage 11 evidence for the browser."""

    run_id: str
    experiment_id: str
    headline: str
    runtime_label: str
    integrity: ExperimentIntegrity
    training: ExperimentTrainingConfig
    data_audit: ExperimentDataAudit
    shared_splits: dict[str, int]
    dataset_lock_sha256: str
    configuration_sha256: str
    source_model_revisions: dict[str, str]
    pretrained_control: ExperimentPretrainedReport
    variants: list[ExperimentVariantReport]
    incidents: list[ExperimentIncident]
    training_resumptions: list[dict[str, object]]
    total_gpu_hours: float = Field(ge=0)
    estimated_total_cost_usd: float = Field(ge=0)


class ExperimentFixtureTerm(ApiSchema):
    """One domain term attached to a locked evaluation sentence."""

    text: str
    canonical: str
    category: str


class ExperimentFixture(ApiSchema):
    """Public, text-only projection of a locked Stage 11 test row."""

    id: str
    text: str
    target_terms: list[ExperimentFixtureTerm]


class ExperimentFixturePage(ApiSchema):
    """Paginated fixture response."""

    items: list[ExperimentFixture]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    manifest_sha256: str


class ExperimentModelSummary(ApiSchema):
    """Technology-neutral comparison model metadata."""

    id: ExperimentModelId
    name: str
    role: Literal["pretrained", "adapted"]
    variant: str
    runtime: Literal["PyTorch CPU"] = "PyTorch CPU"
    hosting: Literal["self-hosted"] = "self-hosted"
    revision: str
    model_sha256: str | None
    available: bool
    unavailable_reason: str | None = None


class ExperimentComparisonRequest(ApiSchema):
    """Start either a locked-fixture or custom-text comparison."""

    mode: Literal["fixture", "custom"]
    fixture_id: str | None = Field(default=None, min_length=1, max_length=160)
    text: str | None = Field(default=None, min_length=1, max_length=500)
    target_terms: list[str] | None = Field(default=None, min_length=1, max_length=20)
    model_ids: list[ExperimentModelId] = Field(min_length=2, max_length=4)
    sanitize_text: bool = True
    normalize_text: bool = True

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "ExperimentComparisonRequest":
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("Comparison model IDs must be unique.")
        if self.mode == "fixture":
            if not self.fixture_id or self.text is not None or self.target_terms is not None:
                raise ValueError(
                    "Fixture comparisons require fixtureId and do not accept text or targetTerms."
                )
            return self
        if self.fixture_id is not None or not self.text or not self.target_terms:
            raise ValueError(
                "Custom comparisons require text and targetTerms and do not accept fixtureId."
            )
        normalized_terms = [term.strip() for term in self.target_terms]
        if any(not term or len(term) > 80 for term in normalized_terms):
            raise ValueError("Target terms must contain between 1 and 80 characters.")
        reference = self.text.casefold()
        absent = [term for term in normalized_terms if term.casefold() not in reference]
        if absent:
            raise ValueError(
                "Every target term must appear in the custom text: " + ", ".join(absent)
            )
        self.target_terms = list(dict.fromkeys(normalized_terms))
        return self


class ExperimentTermScore(ApiSchema):
    """Term-level ASR proxy measurements for one generated result."""

    correct: list[str]
    incorrect: list[str]
    correct_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)


class ExperimentRuntimeMetrics(ApiSchema):
    """Live CPU runtime measurements with distinct timing boundaries."""

    model_load_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    audio_duration_ms: float = Field(ge=0)
    real_time_factor: float = Field(ge=0)
    process_memory_mb: float = Field(ge=0)
    asr_ms: float = Field(ge=0)
    warm: bool
    runtime: Literal["CPU"] = "CPU"


class ExperimentResultProvenance(ApiSchema):
    """Exact serving identity for a live result."""

    model_sha256: str
    source_revision: str
    vocoder_revision: str
    speaker_profile_sha256: str


class ExperimentModelResult(ApiSchema):
    """Progressive or terminal result for one selected model."""

    model_id: ExperimentModelId
    status: Literal[
        "queued", "loading", "synthesizing", "audio_ready", "transcribing", "success", "failure"
    ]
    original_text: str
    normalized_text: str
    audio_url: str | None = None
    transcript: str | None = None
    target_terms: ExperimentTermScore | None = None
    word_error_rate: float | None = Field(default=None, ge=0)
    metrics: ExperimentRuntimeMetrics | None = None
    provenance: ExperimentResultProvenance | None = None
    error: str | None = None


class ExperimentComparisonJob(ApiSchema):
    """Durable browser-facing snapshot of a comparison job."""

    id: str
    mode: Literal["fixture", "custom"]
    stage: ComparisonStage
    progress_percent: float = Field(ge=0, le=100)
    original_text: str
    normalized_text: str | None = None
    target_terms: list[str]
    sanitize_text: bool
    normalize_text: bool
    model_ids: list[ExperimentModelId]
    results: list[ExperimentModelResult]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
