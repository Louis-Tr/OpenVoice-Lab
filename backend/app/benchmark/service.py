"""In-memory benchmark job orchestration for the browser workflow."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from app.benchmark.runner import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_RESULT_DIR,
    benchmark_id,
    load_corpus,
    resolve_benchmark_voice,
    resolve_models,
    run_isolated_benchmark,
    utc_now,
)
from app.models.registry import ModelRegistry
from app.schemas.benchmark import (
    BenchmarkConfig,
    BenchmarkJobStatus,
    BenchmarkRequest,
    BenchmarkResult,
)

BenchmarkCoordinator = Callable[..., tuple[BenchmarkResult, Path]]


class BenchmarkJobNotFoundError(LookupError):
    """Raised when a benchmark job is not present in this process."""


class BenchmarkJobService:
    """Start background benchmark jobs and expose pollable progress."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        *,
        corpus_path: Path = DEFAULT_CORPUS_PATH,
        result_dir: Path = DEFAULT_RESULT_DIR,
        coordinator: BenchmarkCoordinator = run_isolated_benchmark,
        default_voice_id: str | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._corpus_path = corpus_path
        self._result_dir = result_dir
        self._coordinator = coordinator
        self._default_voice_id = default_voice_id
        self._jobs: dict[str, BenchmarkJobStatus] = {}
        self._latest_identifier: str | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = Lock()

    def describe(self) -> BenchmarkConfig:
        """Describe the fixed workload without loading a model."""
        corpus, corpus_hash = load_corpus(self._corpus_path)
        models = resolve_models(self._model_registry, None)
        return BenchmarkConfig(
            corpus_version=corpus.version,
            corpus_sha256=corpus_hash,
            test_case_count=len(corpus.cases),
            model_count=len(models),
            total_evaluations=len(corpus.cases) * len(models),
            model_ids=[model.id for model in models],
            model_voice_ids={
                model.id: resolve_benchmark_voice(model, self._default_voice_id)
                for model in models
            },
            default_voice_id=self._default_voice_id,
        )

    async def start(self, request: BenchmarkRequest) -> BenchmarkJobStatus:
        """Validate a workload, create its job, and return immediately."""
        corpus, _corpus_hash = load_corpus(self._corpus_path)
        models = resolve_models(self._model_registry, request.model_ids)
        identifier = benchmark_id(utc_now())
        total_evaluations = len(corpus.cases) * len(models)
        job = BenchmarkJobStatus(
            benchmark_id=identifier,
            status="pending",
            test_case_count=len(corpus.cases),
            model_count=len(models),
            total_evaluations=total_evaluations,
            completed_evaluations=0,
            progress_percent=0,
        )
        with self._lock:
            self._jobs[identifier] = job
            self._latest_identifier = identifier

        task = asyncio.create_task(self._execute(identifier, request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def get(self, identifier: str) -> BenchmarkJobStatus:
        """Return the latest immutable snapshot for one job."""
        with self._lock:
            job = self._jobs.get(identifier)
        if job is None:
            raise BenchmarkJobNotFoundError(
                f"Benchmark job '{identifier}' was not found."
            )
        return job

    def latest(self) -> BenchmarkJobStatus:
        """Return the newest job so a fresh browser can recover its state."""
        with self._lock:
            identifier = self._latest_identifier
        if identifier is None:
            raise BenchmarkJobNotFoundError("No benchmark job has been started yet.")
        return self.get(identifier)

    async def _execute(self, identifier: str, request: BenchmarkRequest) -> None:
        self._update(identifier, status="running")

        def report_progress(completed: int, total: int) -> None:
            percent = round((completed / total) * 100, 1) if total else 0
            self._update(
                identifier,
                completed_evaluations=completed,
                progress_percent=percent,
            )

        try:
            result, _path = await asyncio.to_thread(
                self._coordinator,
                model_ids=request.model_ids,
                voice_id=request.voice_id,
                sanitize_text=request.sanitize_text,
                normalize_text=request.normalize_text,
                corpus_path=self._corpus_path,
                result_dir=self._result_dir,
                run_id=identifier,
                progress_callback=report_progress,
            )
        except Exception as error:  # noqa: BLE001 - job failures are public state.
            self._update(identifier, status="failed", error=str(error))
            return

        self._update(
            identifier,
            status="completed",
            completed_evaluations=len(result.raw_results),
            progress_percent=100,
            result=result,
        )

    def _update(self, identifier: str, **changes: object) -> None:
        with self._lock:
            current = self._jobs[identifier]
            self._jobs[identifier] = current.model_copy(update=changes)
