"""Application facade for Stage 11 evidence and live comparisons."""

from importlib.util import find_spec
from pathlib import Path

from app.experiments.artifact_report import ExperimentReportService
from app.experiments.fixtures import ExperimentFixtureService
from app.experiments.jobs import ExperimentJobService
from app.experiments.model_registry import ExperimentModelRegistry
from app.experiments.store import ExperimentJobStore
from app.inference.speecht5_cpu import SpeechT5CpuRuntime
from app.schemas.experiment import (
    ExperimentComparisonJob,
    ExperimentComparisonRequest,
    ExperimentFixturePage,
    ExperimentModelSummary,
    ExperimentReport,
)
from app.text_processing.service import TextProcessingService


class ExperimentService:
    """Keep controllers independent from report, catalog, and worker details."""

    def __init__(
        self,
        *,
        reports: ExperimentReportService,
        fixtures: ExperimentFixtureService,
        models: ExperimentModelRegistry,
        jobs: ExperimentJobService | None,
        unavailable_reason: str | None = None,
    ) -> None:
        self._reports = reports
        self._fixtures = fixtures
        self._models = models
        self._jobs = jobs
        self._unavailable_reason = unavailable_reason

    def report(self) -> ExperimentReport:
        return self._reports.get()

    def fixtures(
        self,
        *,
        query: str | None,
        term: str | None,
        category: str | None,
        offset: int,
        limit: int,
    ) -> ExperimentFixturePage:
        return self._fixtures.list(
            query=query, term=term, category=category, offset=offset, limit=limit
        )

    def models(self) -> list[ExperimentModelSummary]:
        return self._models.list()

    async def start(self, request: ExperimentComparisonRequest) -> ExperimentComparisonJob:
        return await self._require_jobs().start(request)

    def get(self, job_id: str) -> ExperimentComparisonJob:
        return self._require_jobs().get(job_id)

    def cancel(self, job_id: str) -> ExperimentComparisonJob:
        return self._require_jobs().cancel(job_id)

    async def recover(self) -> None:
        if self._jobs is not None:
            await self._jobs.recover()

    def _require_jobs(self) -> ExperimentJobService:
        if self._jobs is None:
            from app.experiments.common import ExperimentEvidenceError

            raise ExperimentEvidenceError(self._unavailable_reason or (
                "Live SpeechT5 comparison is not provisioned. Run the Stage 12 model and "
                "speaker-profile setup first."
            ))
        return self._jobs


def create_experiment_service(
    *,
    artifact_root: Path,
    manifest_root: Path,
    stage12_root: Path,
    model_cache_root: Path,
    speaker_profile_root: Path,
    text_processing: TextProcessingService,
    audio_url_prefix: str,
    tts_revision: str,
    vocoder_revision: str,
    maximum_queued_jobs: int,
    maximum_cached_models: int,
    cpu_threads: int | None,
) -> ExperimentService:
    """Compose lightweight evidence services and optional live CPU execution."""
    reports = ExperimentReportService(artifact_root, manifest_root)
    fixtures = ExperimentFixtureService(manifest_root)
    models = ExperimentModelRegistry.from_artifacts(
        artifact_root,
        model_cache_root / "pretrained-speecht5",
        tts_revision,
    )
    profile_json = speaker_profile_root / "speaker-profile.json"
    profile_embedding = speaker_profile_root / "speaker-embedding.npy"
    required_runtime_paths = (
        profile_json,
        profile_embedding,
        model_cache_root / "vocoder",
        model_cache_root / "asr",
    )
    dependencies_ready = all(
        find_spec(module) is not None for module in ("soundfile", "torch", "transformers")
    )
    ready = dependencies_ready and all(path.exists() for path in required_runtime_paths) and all(
        model.available and model.model_sha256 for model in models.list()
    )
    jobs = None
    unavailable_reason = None
    if ready:
        runtime = SpeechT5CpuRuntime(
            vocoder_root=model_cache_root / "vocoder",
            asr_root=model_cache_root / "asr",
            speaker_embedding_path=profile_embedding,
            maximum_cached_models=maximum_cached_models,
            cpu_threads=cpu_threads,
        )
        jobs = ExperimentJobService(
            fixtures=fixtures,
            models=models,
            runtime=runtime,
            text_processing=text_processing,
            store=ExperimentJobStore(stage12_root / "comparisons"),
            audio_url_prefix=audio_url_prefix,
            vocoder_revision=vocoder_revision,
            speaker_profile_path=profile_json,
            maximum_queued_jobs=maximum_queued_jobs,
        )
    else:
        unavailable_reason = (
            "Live SpeechT5 comparison dependencies are not installed. Start the backend with "
            ".runtime/stage12-venv or install backend[experiment]."
            if not dependencies_ready
            else "Live SpeechT5 comparison is not provisioned. Run the Stage 12 model download "
            "and speaker-profile preparation commands, then restart FastAPI."
        )
    return ExperimentService(
        reports=reports,
        fixtures=fixtures,
        models=models,
        jobs=jobs,
        unavailable_reason=unavailable_reason,
    )
