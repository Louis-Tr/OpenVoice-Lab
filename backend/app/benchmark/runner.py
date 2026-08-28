"""Reproducible benchmark execution through the synthesis workflow."""

import argparse
import asyncio
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from app.benchmark.evaluator import BenchmarkEvaluator
from app.models.registry import ModelRegistry
from app.schemas.benchmark import (
    BenchmarkCaseResult,
    BenchmarkCorpus,
    BenchmarkEnvironment,
    BenchmarkRequest,
    BenchmarkResult,
)
from app.schemas.model import ModelSummary
from app.schemas.synthesis import SynthesisRequest, SynthesisResult

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = Path(__file__).with_name("sentences.json")
DEFAULT_RESULT_DIR = BACKEND_ROOT / "benchmark-results"
UtcClock = Callable[[], datetime]
ProgressCallback = Callable[[int, int], None]


class SynthesisWorkflow(Protocol):
    """Narrow application-service contract consumed by the benchmark runner."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Execute the same measured synthesis path used by HTTP requests."""
        ...


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> tuple[BenchmarkCorpus, str]:
    """Validate the corpus and return its content hash."""
    content = path.read_bytes()
    corpus = BenchmarkCorpus.model_validate_json(content)
    case_ids = [case.id for case in corpus.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Benchmark case IDs must be unique.")
    return corpus, sha256(content).hexdigest()


def describe_environment(*, isolated: bool) -> BenchmarkEnvironment:
    """Capture enough runtime context to interpret results."""
    return BenchmarkEnvironment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor() or "unknown",
        logical_cpu_count=os.cpu_count(),
        model_process_isolation=isolated,
    )


def benchmark_id(timestamp: datetime) -> str:
    """Create a sortable, collision-resistant benchmark identifier."""
    utc_timestamp = timestamp.astimezone(UTC)
    return "benchmark-" + utc_timestamp.strftime("%Y-%m-%dT%H-%M-%S-%fZ")


def write_benchmark_result(result: BenchmarkResult, path: Path) -> BenchmarkResult:
    """Persist one immutable JSON result without overwriting evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = result.model_copy(update={"result_file": path.name})
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(persisted.model_dump_json(by_alias=True, indent=2))
        output.write("\n")
    return persisted


def resolve_models(
    model_registry: ModelRegistry,
    requested_ids: Sequence[str] | None,
) -> list[ModelSummary]:
    """Resolve requested IDs in order, defaulting to all available synthesis models."""
    summaries = model_registry.list_benchmark_models()
    if requested_ids is None:
        return summaries

    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Benchmark model IDs must be unique.")
    by_id = {model.id: model for model in summaries}
    resolved: list[ModelSummary] = []
    for model_id in requested_ids:
        model_registry.get(model_id)
        try:
            resolved.append(by_id[model_id])
        except KeyError as error:
            raise ValueError(
                f"Model '{model_id}' is not currently available for synthesis."
            ) from error
    return resolved


def resolve_benchmark_voice(
    model: ModelSummary,
    preferred_voice_id: str | None,
) -> str:
    """Choose a valid voice while preserving a caller preference when possible."""
    if preferred_voice_id and preferred_voice_id in model.voices:
        return preferred_voice_id
    if model.voices:
        return model.voices[0]
    raise ValueError(f"Model '{model.id}' does not expose a synthesis voice.")


class BenchmarkRunner:
    """Execute one corpus through the existing synthesis application service."""

    def __init__(
        self,
        synthesis_service: SynthesisWorkflow,
        model_registry: ModelRegistry,
        *,
        evaluator: BenchmarkEvaluator | None = None,
        corpus_path: Path = DEFAULT_CORPUS_PATH,
        result_dir: Path = DEFAULT_RESULT_DIR,
        clock: UtcClock = utc_now,
    ) -> None:
        self._synthesis_service = synthesis_service
        self._model_registry = model_registry
        self._evaluator = evaluator or BenchmarkEvaluator()
        self._corpus_path = corpus_path
        self._result_dir = result_dir
        self._clock = clock

    async def run(
        self,
        request: BenchmarkRequest,
        *,
        result_path: Path | None = None,
        process_isolation: bool = False,
    ) -> BenchmarkResult:
        """Run every selected model against every case and persist all outcomes."""
        started_at = self._clock()
        corpus, corpus_hash = load_corpus(self._corpus_path)
        models = resolve_models(self._model_registry, request.model_ids)
        raw_results: list[BenchmarkCaseResult] = []

        for model in models:
            model_voice_id = resolve_benchmark_voice(model, request.voice_id)
            for case in corpus.cases:
                try:
                    synthesis = await self._synthesis_service.synthesize(
                        SynthesisRequest(
                            text=case.text,
                            model_id=model.id,
                            voice_id=model_voice_id,
                            sanitize_text=request.sanitize_text,
                            normalize_text=request.normalize_text,
                        )
                    )
                    raw_results.append(
                        BenchmarkCaseResult(
                            case_id=case.id,
                            category=case.category,
                            text=case.text,
                            normalized_text=synthesis.normalized_text,
                            sanitize_text=request.sanitize_text,
                            normalize_text=request.normalize_text,
                            model_id=model.id,
                            precision=model.precision,
                            model_variant=model.variant,
                            voice_id=model_voice_id,
                            status="success",
                            audio_url=synthesis.audio_url,
                            metrics=synthesis.metrics,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - failures are benchmark data.
                    raw_results.append(
                        BenchmarkCaseResult(
                            case_id=case.id,
                            category=case.category,
                            text=case.text,
                            normalized_text=(
                                case.text
                                if not request.sanitize_text and not request.normalize_text
                                else getattr(error, "normalized_text", None)
                            ),
                            sanitize_text=request.sanitize_text,
                            normalize_text=request.normalize_text,
                            model_id=model.id,
                            precision=model.precision,
                            model_variant=model.variant,
                            voice_id=model_voice_id,
                            status="failure",
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )
                    )

        completed_at = self._clock()
        result = BenchmarkResult(
            benchmark_id=benchmark_id(started_at),
            status=(
                "completed_with_failures"
                if any(item.status == "failure" for item in raw_results)
                else "completed"
            ),
            started_at=started_at,
            completed_at=completed_at,
            corpus_version=corpus.version,
            corpus_sha256=corpus_hash,
            voice_id=request.voice_id,
            model_voice_ids={
                model.id: resolve_benchmark_voice(model, request.voice_id)
                for model in models
            },
            sanitize_text=request.sanitize_text,
            normalize_text=request.normalize_text,
            model_ids=[model.id for model in models],
            environment=describe_environment(isolated=process_isolation),
            raw_results=raw_results,
            aggregates=self._evaluator.aggregate(raw_results, models),
        )
        destination = result_path or self._result_dir / f"{result.benchmark_id}.json"
        return write_benchmark_result(result, destination)


async def run_worker(
    *,
    model_id: str,
    voice_id: str | None,
    sanitize_text: bool,
    normalize_text: bool,
    corpus_path: Path,
    result_path: Path,
) -> None:
    """Run one model in a fresh process for isolated memory measurements."""
    from app.main import create_app

    application = create_app()
    runner = BenchmarkRunner(
        application.state.synthesis_service,
        application.state.model_registry,
        corpus_path=corpus_path,
        result_dir=result_path.parent,
    )
    await runner.run(
        BenchmarkRequest(
            model_ids=[model_id],
            voice_id=voice_id,
            sanitize_text=sanitize_text,
            normalize_text=normalize_text,
        ),
        result_path=result_path,
        process_isolation=True,
    )


def run_isolated_benchmark(
    *,
    model_ids: Sequence[str] | None,
    voice_id: str | None,
    sanitize_text: bool = True,
    normalize_text: bool = True,
    corpus_path: Path,
    result_dir: Path,
    clock: UtcClock = utc_now,
    run_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[BenchmarkResult, Path]:
    """Coordinate one worker process per model and merge comparable evidence."""
    from app.main import create_app

    started_at = clock()
    application = create_app()
    registry: ModelRegistry = application.state.model_registry
    evaluator = BenchmarkEvaluator()
    models = resolve_models(registry, model_ids)
    corpus, corpus_hash = load_corpus(corpus_path)
    raw_results: list[BenchmarkCaseResult] = []
    total_evaluations = len(models) * len(corpus.cases)

    with TemporaryDirectory(prefix="openvoice-benchmark-") as temporary:
        temporary_dir = Path(temporary)
        for model in models:
            partial_path = temporary_dir / f"{model.id}.json"
            command = [
                sys.executable,
                "-m",
                "app.benchmark.runner",
                "--worker-model",
                model.id,
                "--worker-result",
                str(partial_path),
                "--sanitize-text" if sanitize_text else "--no-sanitize-text",
                "--normalize-text" if normalize_text else "--no-normalize-text",
                "--corpus",
                str(corpus_path),
            ]
            if voice_id:
                command.extend(("--voice-id", voice_id))
            subprocess.run(command, cwd=BACKEND_ROOT, check=True)
            partial = BenchmarkResult.model_validate_json(
                partial_path.read_text(encoding="utf-8")
            )
            if partial.corpus_sha256 != corpus_hash:
                raise RuntimeError("Worker corpus hash does not match the coordinator.")
            if (
                partial.sanitize_text != sanitize_text
                or partial.normalize_text != normalize_text
            ):
                raise RuntimeError(
                    "Worker text-processing configuration does not match the coordinator."
                )
            expected_voice = resolve_benchmark_voice(model, voice_id)
            if partial.model_voice_ids != {model.id: expected_voice}:
                raise RuntimeError(
                    "Worker voice selection does not match the coordinator."
                )
            raw_results.extend(partial.raw_results)
            if progress_callback:
                progress_callback(len(raw_results), total_evaluations)

    completed_at = clock()
    result = BenchmarkResult(
        benchmark_id=run_id or benchmark_id(started_at),
        status=(
            "completed_with_failures"
            if any(item.status == "failure" for item in raw_results)
            else "completed"
        ),
        started_at=started_at,
        completed_at=completed_at,
        corpus_version=corpus.version,
        corpus_sha256=corpus_hash,
        voice_id=voice_id,
        model_voice_ids={
            model.id: resolve_benchmark_voice(model, voice_id) for model in models
        },
        sanitize_text=sanitize_text,
        normalize_text=normalize_text,
        model_ids=[model.id for model in models],
        environment=describe_environment(isolated=True),
        raw_results=raw_results,
        aggregates=evaluator.aggregate(raw_results, models),
    )
    destination = result_dir / f"{result.benchmark_id}.json"
    persisted = write_benchmark_result(result, destination)
    return persisted, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark local TTS model configurations with one fixed corpus."
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Registry ID to benchmark; repeat to select multiple. Defaults to all.",
    )
    parser.add_argument(
        "--voice-id",
        default=None,
        help=(
            "Preferred voice when supported; otherwise each model uses its first "
            "advertised synthesis voice."
        ),
    )
    parser.add_argument(
        "--sanitize-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic sanitization for every benchmark case.",
    )
    parser.add_argument(
        "--normalize-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable speakable-English normalization for every benchmark case.",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def print_summary(result: BenchmarkResult, output_path: Path) -> None:
    """Print a compact human-readable result without replacing the JSON record."""
    print(f"Benchmark complete: {output_path}")
    print("Variant\tAvg latency\tP95\tAvg RTF\tPeak memory\tFailures")
    for aggregate in result.aggregates:
        print(
            f"{aggregate.precision}\t"
            f"{aggregate.average_latency_ms} ms\t"
            f"{aggregate.p95_latency_ms} ms\t"
            f"{aggregate.average_real_time_factor}\t"
            f"{aggregate.peak_memory_mb} MB\t"
            f"{aggregate.failure_count}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.worker_model) != bool(args.worker_result):
        raise SystemExit("Worker model and result path must be provided together.")

    if args.worker_model:
        asyncio.run(
            run_worker(
                model_id=args.worker_model,
                voice_id=args.voice_id,
                sanitize_text=args.sanitize_text,
                normalize_text=args.normalize_text,
                corpus_path=args.corpus.resolve(),
                result_path=args.worker_result.resolve(),
            )
        )
        return

    result, output_path = run_isolated_benchmark(
        model_ids=args.model_ids,
        voice_id=args.voice_id,
        sanitize_text=args.sanitize_text,
        normalize_text=args.normalize_text,
        corpus_path=args.corpus.resolve(),
        result_dir=args.output_dir.resolve(),
    )
    print_summary(result, output_path)


if __name__ == "__main__":
    main()
