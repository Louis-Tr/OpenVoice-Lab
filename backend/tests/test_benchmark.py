"""Contract tests for reproducible benchmark execution and evaluation."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.benchmark.evaluator import BenchmarkEvaluator
from app.benchmark.runner import BenchmarkRunner, load_corpus
from app.models.registry import ModelDefinition, ModelRegistry
from app.schemas.benchmark import BenchmarkCaseResult, BenchmarkRequest
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
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=case_id,
        category="test",
        text="Benchmark text",
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


class RecordingSynthesisService:
    """Deterministic service double that retains the complete workload."""

    def __init__(self) -> None:
        self.requests: list[SynthesisRequest] = []

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        if request.model_id == "kokoro-q8" and "failure" in request.text:
            raise RuntimeError("record this evaluation failure")

        variant = "fp32" if request.model_id == "kokoro-fp32" else "quantized"
        inference_ms = 100.0 if variant == "fp32" else 80.0
        return SynthesisResult(
            status="ok",
            model=request.model_id,
            text=request.text,
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
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "record this evaluation failure"

    output_path = output_dir / str(result.result_file)
    assert output_path.is_file()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["corpusVersion"] == "test-1"
    assert len(persisted["rawResults"]) == 6
    assert persisted["aggregates"][1]["failureCount"] == 1


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
