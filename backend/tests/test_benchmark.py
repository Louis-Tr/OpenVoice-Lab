"""Contract tests for reproducible benchmark execution and evaluation."""

import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import httpx
import pytest

from app.audio.service import AudioService
from app.benchmark.evaluator import BenchmarkEvaluator
from app.benchmark.runner import BenchmarkRunner, build_parser, load_corpus
from app.benchmark.service import BenchmarkJobNotFoundError, BenchmarkJobService
from app.config.settings import Settings
from app.main import create_app
from app.models.registry import ModelDefinition, ModelRegistry
from app.schemas.benchmark import (
    BenchmarkCaseResult,
    BenchmarkEnvironment,
    BenchmarkRequest,
    BenchmarkResult,
)
from app.schemas.model import ModelSummary
from app.schemas.synthesis import SynthesisMetrics, SynthesisRequest, SynthesisResult


def model_summary(
    model_id: str,
    precision: str,
    variant: str,
) -> ModelSummary:
    return ModelSummary(
        id=model_id,
        name="Kokoro",
        precision=precision,
        variant=variant,
        voices=["af_heart"],
        model_version="1.0",
        runtime="ONNX",
        hosting="self-hosted",
        external_inference_apis=[],
        available=True,
    )


def case_result(
    *,
    model: ModelSummary,
    case_id: str,
    latency_ms: float,
    real_time_factor: float,
    memory_mb: float,
    sanitize_text: bool = True,
    normalize_text: bool = True,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=case_id,
        category="test",
        text="Benchmark text",
        normalized_text="Benchmark text",
        sanitize_text=sanitize_text,
        normalize_text=normalize_text,
        model_id=model.id,
        precision=model.precision,
        model_variant=model.variant,
        voice_id="af_heart",
        status="success",
        metrics=SynthesisMetrics(
            model_load_ms=0,
            inference_ms=latency_ms,
            audio_duration_ms=latency_ms / real_time_factor,
            real_time_factor=real_time_factor,
            memory_mb=memory_mb,
            warm=True,
            model_variant=model.variant,
        ),
    )


class RecordedSynthesisError(RuntimeError):
    """Failure double that exposes the exact text passed to fake inference."""

    def __init__(self, message: str, normalized_text: str) -> None:
        super().__init__(message)
        self.normalized_text = normalized_text


class RecordingSynthesisService:
    """Deterministic service double that retains the complete workload."""

    def __init__(self) -> None:
        self.requests: list[SynthesisRequest] = []

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        normalized_text = request.text
        if request.normalize_text:
            normalized_text = normalized_text.replace("$25", "25 dollars")
        if request.sanitize_text:
            normalized_text = normalized_text.replace("$", "").replace(
                " --- ",
                " ",
            )
        if request.model_id == "kokoro-q8" and "failure" in request.text:
            raise RecordedSynthesisError(
                "record this evaluation failure",
                normalized_text,
            )

        variant = "fp32" if request.model_id == "kokoro-fp32" else "quantized"
        inference_ms = 100.0 if variant == "fp32" else 80.0
        return SynthesisResult(
            status="ok",
            model=request.model_id,
            text=request.text,
            normalized_text=normalized_text,
            audio_url=f"/audio/{request.model_id}.wav",
            metrics=SynthesisMetrics(
                model_load_ms=0,
                inference_ms=inference_ms,
                audio_duration_ms=1_000,
                real_time_factor=inference_ms / 1_000,
                memory_mb=400 if variant == "fp32" else 300,
                warm=True,
                model_variant=variant,
            ),
        )


def registry(tmp_path: Path) -> ModelRegistry:
    voices_path = tmp_path / "voices.bin"
    voices_path.write_bytes(b"voices")
    definitions = []
    for model_id, precision, variant in (
        ("kokoro-fp32", "FP32", "fp32"),
        ("kokoro-q8", "INT8", "quantized"),
    ):
        model_path = tmp_path / f"{model_id}.onnx"
        model_path.write_bytes(b"model")
        definitions.append(
            ModelDefinition(
                model_id=model_id,
                display_name="Kokoro",
                precision=precision,
                variant=variant,
                model_version="1.0",
                model_path=model_path,
                voices_path=voices_path,
                voices=("af_heart",),
            )
        )
    return ModelRegistry(definitions)


def benchmark_coordinator(
    models: Sequence[ModelSummary],
    *,
    pause_after_first_model: tuple[Event, Event] | None = None,
) -> Callable[..., tuple[BenchmarkResult, Path]]:
    """Return a fast coordinator double with the production keyword contract."""

    def coordinate(
        *,
        model_ids: Sequence[str] | None,
        voice_id: str | None,
        sanitize_text: bool,
        normalize_text: bool,
        corpus_path: Path,
        result_dir: Path,
        run_id: str | None,
        progress_callback: Callable[[int, int], None] | None,
    ) -> tuple[BenchmarkResult, Path]:
        del corpus_path
        selected = [model for model in models if model_ids is None or model.id in model_ids]
        raw_results = [
            case_result(
                model=model,
                case_id=case_id,
                latency_ms=100 if model.precision == "FP32" else 80,
                real_time_factor=0.1 if model.precision == "FP32" else 0.08,
                memory_mb=400 if model.precision == "FP32" else 300,
                sanitize_text=sanitize_text,
                normalize_text=normalize_text,
            )
            for model in selected
            for case_id in ("case-a", "case-b")
        ]
        if progress_callback:
            progress_callback(2, 4)
            if pause_after_first_model:
                reported, release = pause_after_first_model
                reported.set()
                release.wait(timeout=2)
            progress_callback(4, 4)

        identifier = run_id or "benchmark-test"
        result = BenchmarkResult(
            benchmark_id=identifier,
            status="completed",
            started_at=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            completed_at=datetime(2026, 8, 26, 12, 1, tzinfo=UTC),
            corpus_version="test-1",
            corpus_sha256="a" * 64,
            voice_id=voice_id,
            model_voice_ids={model.id: model.voices[0] for model in selected},
            sanitize_text=sanitize_text,
            normalize_text=normalize_text,
            model_ids=[model.id for model in selected],
            environment=BenchmarkEnvironment(
                python_version="3.13",
                platform="test",
                processor="test",
                logical_cpu_count=4,
                model_process_isolation=True,
            ),
            raw_results=raw_results,
            aggregates=BenchmarkEvaluator().aggregate(raw_results, selected),
            result_file=f"{identifier}.json",
        )
        return result, result_dir / f"{identifier}.json"

    return coordinate


def test_evaluator_calculates_latency_rtf_memory_and_failures() -> None:
    model = model_summary("kokoro-fp32", "FP32", "fp32")
    results = [
        case_result(
            model=model,
            case_id=f"case-{index}",
            latency_ms=latency,
            real_time_factor=rtf,
            memory_mb=memory,
        )
        for index, (latency, rtf, memory) in enumerate(
            (
                (100.0, 0.1, 100.0),
                (200.0, 0.2, 110.0),
                (300.0, 0.3, 120.0),
                (400.0, 0.4, 130.0),
            ),
            start=1,
        )
    ]
    results.append(
        BenchmarkCaseResult(
            case_id="case-5",
            category="test",
            text="Failed text",
            normalized_text=None,
            sanitize_text=True,
            normalize_text=True,
            model_id=model.id,
            precision=model.precision,
            model_variant=model.variant,
            voice_id="af_heart",
            status="failure",
            error_type="RuntimeError",
            error_message="expected failure",
        )
    )

    aggregate = BenchmarkEvaluator().aggregate(results, [model])[0]

    assert aggregate.total_cases == 5
    assert aggregate.success_count == 4
    assert aggregate.failure_count == 1
    assert aggregate.average_latency_ms == 250.0
    assert aggregate.median_latency_ms == 250.0
    assert aggregate.p95_latency_ms == 385.0
    assert aggregate.average_real_time_factor == 0.25
    assert aggregate.average_memory_mb == 115.0
    assert aggregate.peak_memory_mb == 130.0


def test_runner_uses_identical_cases_and_persists_failures(tmp_path: Path) -> None:
    corpus_path = tmp_path / "sentences.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": "test-1",
                "cases": [
                    {"id": "case-a", "category": "short", "text": "Hello."},
                    {
                        "id": "case-b",
                        "category": "failure",
                        "text": "Record this failure.",
                    },
                    {"id": "case-c", "category": "question", "text": "Ready?"},
                ],
            }
        ),
        encoding="utf-8",
    )
    clock_values = iter(
        (
            datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 26, 12, 1, tzinfo=UTC),
        )
    )
    service = RecordingSynthesisService()
    output_dir = tmp_path / "results"
    runner = BenchmarkRunner(
        service,
        registry(tmp_path),
        corpus_path=corpus_path,
        result_dir=output_dir,
        clock=clock_values.__next__,
    )

    result = asyncio.run(runner.run(BenchmarkRequest()))

    assert result.status == "completed_with_failures"
    assert len(result.raw_results) == 6
    case_ids_by_model = {
        model_id: [
            item.case_id for item in result.raw_results if item.model_id == model_id
        ]
        for model_id in result.model_ids
    }
    assert case_ids_by_model["kokoro-fp32"] == ["case-a", "case-b", "case-c"]
    assert case_ids_by_model["kokoro-q8"] == ["case-a", "case-b", "case-c"]
    assert [request.text for request in service.requests[:3]] == [
        request.text for request in service.requests[3:]
    ]
    quantized = next(item for item in result.aggregates if item.model_id == "kokoro-q8")
    assert quantized.failure_count == 1
    failed = next(item for item in result.raw_results if item.status == "failure")
    assert failed.error_type == "RecordedSynthesisError"
    assert failed.error_message == "record this evaluation failure"
    assert failed.text == "Record this failure."
    assert failed.normalized_text == "Record this failure."
    assert failed.sanitize_text is True
    assert failed.normalize_text is True
    assert result.sanitize_text is True
    assert result.normalize_text is True

    output_path = output_dir / str(result.result_file)
    assert output_path.is_file()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["corpusVersion"] == "test-1"
    assert persisted["sanitizeText"] is True
    assert persisted["normalizeText"] is True
    assert len(persisted["rawResults"]) == 6
    assert persisted["rawResults"][0]["text"] == "Hello."
    assert persisted["rawResults"][0]["normalizedText"] == "Hello."
    assert persisted["rawResults"][0]["sanitizeText"] is True
    assert persisted["rawResults"][0]["normalizeText"] is True
    assert persisted["aggregates"][1]["failureCount"] == 1


def test_runner_includes_every_available_synthesis_model_with_valid_voice(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "all-models-corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": "all-models-1",
                "cases": [
                    {"id": "case-a", "category": "short", "text": "Hello."},
                ],
            }
        ),
        encoding="utf-8",
    )
    model_registry = registry(tmp_path)
    audio8_path = tmp_path / "audio8.onnx"
    audio8_path.write_bytes(b"model")
    definitions = [model_registry.get(model.id) for model in model_registry.list_available()]
    definitions.append(
        ModelDefinition(
            model_id="audio8-0.6b",
            display_name="Audio8 0.6B",
            precision="INT4",
            variant="audio8",
            model_version="1.0",
            model_path=audio8_path,
            voices_path=None,
            voices=("unconditioned",),
            engine="audio8-onnx",
        )
    )
    all_models_registry = ModelRegistry(definitions)
    service = RecordingSynthesisService()
    runner = BenchmarkRunner(
        service,
        all_models_registry,
        corpus_path=corpus_path,
        result_dir=tmp_path / "results",
    )

    result = asyncio.run(runner.run(BenchmarkRequest()))

    assert result.model_ids == ["kokoro-fp32", "kokoro-q8", "audio8-0.6b"]
    assert result.model_voice_ids == {
        "kokoro-fp32": "af_heart",
        "kokoro-q8": "af_heart",
        "audio8-0.6b": "unconditioned",
    }
    assert [(request.model_id, request.voice_id) for request in service.requests] == [
        ("kokoro-fp32", "af_heart"),
        ("kokoro-q8", "af_heart"),
        ("audio8-0.6b", "unconditioned"),
    ]

@pytest.mark.parametrize(
    ("sanitize_text", "normalize_text", "expected_text"),
    (
        (True, True, "Price: 25 dollars today."),
        (True, False, "Price: 25 today."),
        (False, True, "Price: 25 dollars --- today."),
        (False, False, "Price: $25 --- today."),
    ),
)
def test_runner_records_independent_processing_configuration(
    tmp_path: Path,
    sanitize_text: bool,
    normalize_text: bool,
    expected_text: str,
) -> None:
    corpus_path = tmp_path / "processing-corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": "processing-1",
                "cases": [
                    {
                        "id": "technical",
                        "category": "technical",
                        "text": "Price: $25 --- today.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = RecordingSynthesisService()
    runner = BenchmarkRunner(
        service,
        registry(tmp_path),
        corpus_path=corpus_path,
    )
    destination = tmp_path / f"result-{sanitize_text}-{normalize_text}.json"

    result = asyncio.run(
        runner.run(
            BenchmarkRequest(
                model_ids=["kokoro-fp32"],
                sanitize_text=sanitize_text,
                normalize_text=normalize_text,
            ),
            result_path=destination,
        )
    )

    assert result.sanitize_text is sanitize_text
    assert result.normalize_text is normalize_text
    assert service.requests[0].sanitize_text is sanitize_text
    assert service.requests[0].normalize_text is normalize_text
    raw = result.raw_results[0]
    assert raw.text == "Price: $25 --- today."
    assert raw.normalized_text == expected_text
    assert raw.sanitize_text is sanitize_text
    assert raw.normalize_text is normalize_text

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted["sanitizeText"] is sanitize_text
    assert persisted["normalizeText"] is normalize_text
    assert persisted["rawResults"][0]["text"] == "Price: $25 --- today."
    assert persisted["rawResults"][0]["normalizedText"] == expected_text


def test_default_corpus_covers_required_evaluation_categories() -> None:
    corpus, corpus_hash = load_corpus()

    assert {case.category for case in corpus.cases} == {
        "short conversational",
        "medium conversational",
        "long-form",
        "numbers",
        "dates",
        "punctuation",
        "questions",
        "unusual names/words",
    }
    assert len(corpus_hash) == 64


def test_empty_model_selection_is_rejected() -> None:
    with pytest.raises(ValueError):
        BenchmarkRequest(model_ids=[])


def test_benchmark_cli_processing_options_default_on_and_toggle_independently() -> None:
    parser = build_parser()

    defaults = parser.parse_args([])
    sanitizer_only = parser.parse_args(["--sanitize-text", "--no-normalize-text"])
    normalizer_only = parser.parse_args(["--no-sanitize-text", "--normalize-text"])
    neither = parser.parse_args(["--no-sanitize-text", "--no-normalize-text"])

    assert (defaults.sanitize_text, defaults.normalize_text) == (True, True)
    assert (sanitizer_only.sanitize_text, sanitizer_only.normalize_text) == (
        True,
        False,
    )
    assert (normalizer_only.sanitize_text, normalizer_only.normalize_text) == (
        False,
        True,
    )
    assert (neither.sanitize_text, neither.normalize_text) == (False, False)


def test_job_service_reports_progress_and_completed_result(tmp_path: Path) -> None:
    corpus_path = tmp_path / "job-corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": "test-1",
                "cases": [
                    {"id": "case-a", "category": "short", "text": "Hello."},
                    {"id": "case-b", "category": "question", "text": "Ready?"},
                ],
            }
        ),
        encoding="utf-8",
    )
    model_registry = registry(tmp_path)
    models = model_registry.list_available()
    first_model_reported = Event()
    release_worker = Event()
    service = BenchmarkJobService(
        model_registry,
        corpus_path=corpus_path,
        result_dir=tmp_path / "results",
        coordinator=benchmark_coordinator(
            models,
            pause_after_first_model=(first_model_reported, release_worker),
        ),
    )

    async def scenario() -> None:
        job = await service.start(BenchmarkRequest())
        assert job.status == "pending"
        assert await asyncio.to_thread(first_model_reported.wait, 1)

        progress = service.get(job.benchmark_id)
        assert progress.status == "running"
        assert progress.completed_evaluations == 2
        assert progress.total_evaluations == 4
        assert progress.progress_percent == 50

        release_worker.set()
        for _attempt in range(20):
            completed = service.get(job.benchmark_id)
            if completed.status == "completed":
                break
            await asyncio.sleep(0.01)
        assert completed.status == "completed"
        assert completed.completed_evaluations == 4
        assert completed.progress_percent == 100
        assert completed.result is not None
        assert len(completed.result.aggregates) == 2
        assert service.latest().benchmark_id == job.benchmark_id

    asyncio.run(scenario())


def test_job_service_preserves_failure_state(tmp_path: Path) -> None:
    model_registry = registry(tmp_path)

    def failing_coordinator(**_kwargs: object) -> tuple[BenchmarkResult, Path]:
        raise RuntimeError("isolated worker failed")

    service = BenchmarkJobService(
        model_registry,
        result_dir=tmp_path / "results",
        coordinator=failing_coordinator,
    )

    async def scenario() -> None:
        job = await service.start(BenchmarkRequest())
        for _attempt in range(20):
            failed = service.get(job.benchmark_id)
            if failed.status == "failed":
                break
            await asyncio.sleep(0.01)
        assert failed.status == "failed"
        assert failed.error == "isolated worker failed"
        assert failed.result is None

        with pytest.raises(BenchmarkJobNotFoundError):
            service.get("missing-job")

    asyncio.run(scenario())


def test_benchmark_http_contract_runs_and_polls_job(tmp_path: Path) -> None:
    corpus_path = tmp_path / "api-corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "version": "api-1",
                "cases": [
                    {"id": "case-a", "category": "short", "text": "Hello."},
                    {"id": "case-b", "category": "question", "text": "Ready?"},
                ],
            }
        ),
        encoding="utf-8",
    )
    model_registry = registry(tmp_path)
    service = BenchmarkJobService(
        model_registry,
        corpus_path=corpus_path,
        result_dir=tmp_path / "results",
        coordinator=benchmark_coordinator(model_registry.list_available()),
    )
    app = create_app(
        Settings(environment="test"),
        model_registry=model_registry,
        audio_service=AudioService(tmp_path / "audio"),
        benchmark_job_service=service,
    )

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            config = await client.get("/api/benchmarks/config")
            assert config.status_code == 200
            assert config.json()["testCaseCount"] == 2
            assert config.json()["totalEvaluations"] == 4

            started = await client.post(
                "/api/benchmarks",
                json={
                    "modelIds": ["kokoro-fp32", "kokoro-q8"],
                    "voiceId": "af_heart",
                    "sanitizeText": False,
                    "normalizeText": True,
                },
            )
            assert started.status_code == 202
            identifier = started.json()["benchmarkId"]

            for _attempt in range(20):
                polled = await client.get(f"/api/benchmarks/{identifier}")
                if polled.json()["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
            assert polled.status_code == 200
            assert polled.json()["completedEvaluations"] == 4
            assert polled.json()["result"]["sanitizeText"] is False
            assert polled.json()["result"]["normalizeText"] is True
            assert polled.json()["result"]["rawResults"][0]["sanitizeText"] is False
            assert polled.json()["result"]["rawResults"][0]["normalizeText"] is True
            assert len(polled.json()["result"]["aggregates"]) == 2

            latest = await client.get("/api/benchmarks/latest")
            assert latest.status_code == 200
            assert latest.json()["benchmarkId"] == identifier

            missing = await client.get("/api/benchmarks/missing-job")
            assert missing.status_code == 404

    asyncio.run(scenario())
