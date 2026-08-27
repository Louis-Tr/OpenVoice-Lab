"""Durable, concurrency-limited Stage 12 comparison orchestration."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

from app.experiments.common import ExperimentQueueFullError, read_json
from app.experiments.fixtures import ExperimentFixtureService
from app.experiments.model_registry import ExperimentModelRegistry
from app.experiments.scorer import score_terms, word_error_rate
from app.experiments.store import TERMINAL_STAGES, ExperimentJobStore
from app.inference.speecht5_cpu import ExperimentInferenceRuntime
from app.schemas.experiment import (
    ExperimentComparisonJob,
    ExperimentComparisonRequest,
    ExperimentModelResult,
    ExperimentResultProvenance,
    ExperimentRuntimeMetrics,
)
from app.text_processing.service import TextProcessingService


def utc_now() -> datetime:
    return datetime.now(UTC)


class ExperimentJobService:
    """Run real model comparisons without blocking the FastAPI event loop."""

    def __init__(
        self,
        *,
        fixtures: ExperimentFixtureService,
        models: ExperimentModelRegistry,
        runtime: ExperimentInferenceRuntime,
        text_processing: TextProcessingService,
        store: ExperimentJobStore,
        audio_url_prefix: str,
        vocoder_revision: str,
        speaker_profile_path: Path,
        maximum_queued_jobs: int = 2,
    ) -> None:
        self._fixtures = fixtures
        self._models = models
        self._runtime = runtime
        self._text_processing = text_processing
        self._store = store
        self._audio_url_prefix = audio_url_prefix.rstrip("/")
        self._vocoder_revision = vocoder_revision
        profile = read_json(speaker_profile_path)
        self._speaker_profile_sha256 = str(profile["embedding_sha256"])
        self._maximum_queued_jobs = maximum_queued_jobs
        self._tasks: set[asyncio.Task[None]] = set()
        self._cancellations: dict[str, Event] = {}
        self._guard = Lock()

    async def start(self, request: ExperimentComparisonRequest) -> ExperimentComparisonJob:
        with self._guard:
            active = sum(not task.done() for task in self._tasks)
            if active >= self._maximum_queued_jobs:
                raise ExperimentQueueFullError(
                    "The CPU comparison queue is full. Wait for a running job to finish."
                )
        text, terms = self._resolve_input(request)
        for model_id in request.model_ids:
            definition = self._models.get(model_id)
            if not definition.available or not definition.model_sha256:
                raise ValueError(f"Experiment model '{model_id}' is not ready for live inference.")
        now = utc_now()
        identifier = f"comparison-{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        job = ExperimentComparisonJob(
            id=identifier,
            mode=request.mode,
            stage="queued",
            progress_percent=0,
            original_text=text,
            target_terms=terms,
            sanitize_text=request.sanitize_text,
            normalize_text=request.normalize_text,
            model_ids=request.model_ids,
            results=[
                ExperimentModelResult(
                    model_id=model_id,
                    status="queued",
                    original_text=text,
                    normalized_text=text,
                )
                for model_id in request.model_ids
            ],
            created_at=now,
            updated_at=now,
        )
        self._store.create(job)
        self._schedule(identifier)
        return job

    def get(self, job_id: str) -> ExperimentComparisonJob:
        return self._store.get(job_id)

    def cancel(self, job_id: str) -> ExperimentComparisonJob:
        job = self._store.get(job_id)
        if job.stage in TERMINAL_STAGES:
            return job
        event = self._cancellations.setdefault(job_id, Event())
        event.set()
        cancelled = self._update(
            job,
            stage="cancelled",
            error="Cancellation requested by the user.",
            completed_at=utc_now(),
        )
        self._store.write_manifest(job_id)
        return cancelled

    async def recover(self) -> None:
        """Resume incomplete jobs from their first unfinished model after restart."""
        for job in self._store.nonterminal():
            self._schedule(job.id)

    def _schedule(self, job_id: str) -> None:
        self._cancellations[job_id] = Event()
        task = asyncio.create_task(self._execute(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(self, job_id: str) -> None:
        job = self._store.get(job_id)
        cancellation = self._cancellations[job_id]
        try:
            if cancellation.is_set():
                self._store.write_manifest(job_id)
                return
            job = self._update(job, stage="preprocessing", progress_percent=2)
            processed = self._text_processing.process(
                job.original_text,
                sanitize_text=job.sanitize_text,
                normalize_text=job.normalize_text,
            )
            if cancellation.is_set():
                self._store.write_manifest(job_id)
                return
            job = self._update(job, normalized_text=processed)
            await asyncio.to_thread(self._runtime.prepare)
            if cancellation.is_set():
                self._store.write_manifest(job_id)
                return
            for index, model_id in enumerate(job.model_ids):
                if cancellation.is_set():
                    self._store.write_manifest(job_id)
                    return
                existing = job.results[index]
                if existing.status == "success":
                    continue
                definition = self._models.get(model_id)
                job = self._set_result(job, index, status="loading")
                job = self._update(
                    job,
                    stage="loading_model",
                    progress_percent=self._progress(index, 0, len(job.model_ids)),
                )
                output = self._store.directory(job_id) / "audio" / f"{model_id}.wav"
                job = self._set_result(job, index, status="synthesizing")
                job = self._update(
                    job,
                    stage="synthesizing",
                    progress_percent=self._progress(index, 1, len(job.model_ids)),
                )
                try:
                    audio = await asyncio.to_thread(
                        self._runtime.synthesize, definition, processed, output
                    )
                    if cancellation.is_set():
                        self._store.write_manifest(job_id)
                        return
                    audio_url = f"{self._audio_url_prefix}/{job_id}/audio/{output.name}"
                    provenance = ExperimentResultProvenance(
                        model_sha256=definition.model_sha256 or "",
                        source_revision=definition.revision,
                        vocoder_revision=self._vocoder_revision,
                        speaker_profile_sha256=self._speaker_profile_sha256,
                    )
                    job = self._set_result(
                        job,
                        index,
                        status="audio_ready",
                        audio_url=audio_url,
                        provenance=provenance,
                    )
                    job = self._update(
                        job,
                        stage="audio_ready",
                        progress_percent=self._progress(index, 2, len(job.model_ids)),
                    )
                    job = self._set_result(job, index, status="transcribing")
                    job = self._update(job, stage="transcribing")
                    transcript, asr_ms = await asyncio.to_thread(self._runtime.transcribe, output)
                    if cancellation.is_set():
                        self._store.write_manifest(job_id)
                        return
                    job = self._update(
                        job,
                        stage="scoring",
                        progress_percent=self._progress(index, 3, len(job.model_ids)),
                    )
                    terms = score_terms(job.target_terms, transcript)
                    metrics = ExperimentRuntimeMetrics(
                        model_load_ms=audio.model_load_ms,
                        inference_ms=audio.inference_ms,
                        audio_duration_ms=audio.audio_duration_ms,
                        real_time_factor=audio.real_time_factor,
                        process_memory_mb=audio.process_memory_mb,
                        asr_ms=asr_ms,
                        warm=audio.warm,
                    )
                    job = self._set_result(
                        job,
                        index,
                        status="success",
                        transcript=transcript,
                        target_terms=terms,
                        word_error_rate=word_error_rate(processed, transcript),
                        metrics=metrics,
                    )
                except Exception as error:  # noqa: BLE001 - preserve per-model failures.
                    job = self._set_result(
                        job,
                        index,
                        status="failure",
                        error=f"{type(error).__name__}: {error}",
                    )
                job = self._update(
                    job,
                    progress_percent=round((index + 1) / len(job.model_ids) * 100, 1),
                )
            if cancellation.is_set():
                self._store.write_manifest(job_id)
                return
            failures = sum(item.status == "failure" for item in job.results)
            terminal = "completed_with_failures" if failures else "completed"
            job = self._update(
                job,
                stage=terminal,
                progress_percent=100,
                completed_at=utc_now(),
            )
            self._store.write_manifest(job.id)
        except Exception as error:  # noqa: BLE001 - public durable failure state.
            job = self._store.get(job_id)
            self._update(
                job,
                stage="failed",
                error=f"{type(error).__name__}: {error}",
                completed_at=utc_now(),
            )
            self._store.write_manifest(job_id)

    def _resolve_input(self, request: ExperimentComparisonRequest) -> tuple[str, list[str]]:
        if request.mode == "fixture":
            fixture = self._fixtures.get(request.fixture_id or "")
            return fixture.text, [item.canonical for item in fixture.target_terms]
        assert request.text is not None
        assert request.target_terms is not None
        return request.text, request.target_terms

    def _update(self, job: ExperimentComparisonJob, **changes: object) -> ExperimentComparisonJob:
        changes["updated_at"] = utc_now()
        updated = job.model_copy(update=changes)
        self._store.save(updated)
        return updated

    def _set_result(
        self, job: ExperimentComparisonJob, index: int, **changes: object
    ) -> ExperimentComparisonJob:
        results = list(job.results)
        changes.setdefault("normalized_text", job.normalized_text or job.original_text)
        results[index] = results[index].model_copy(update=changes)
        return self._update(job, results=results)

    @staticmethod
    def _progress(model_index: int, substep: int, model_count: int) -> float:
        completed_units = model_index * 4 + substep
        total_units = max(model_count * 4, 1)
        return min(99.0, round(2 + (completed_units / total_units) * 96, 1))
