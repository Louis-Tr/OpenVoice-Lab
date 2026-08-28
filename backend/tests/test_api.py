"""Executable tests for the synthesis HTTP and measurement contract."""

import asyncio
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from app.audio.service import AudioService
from app.config.settings import Settings
from app.inference.base import AudioResult, TTSInferenceEngine, UnsupportedVoiceError
from app.main import create_app
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition, ModelRegistry


class FakeInferenceEngine(TTSInferenceEngine):
    """Deterministic local engine used to exercise orchestration in unit tests."""

    def __init__(self, tone_hz: float) -> None:
        self.call_count = 0
        self.received_texts: list[str] = []
        self._tone_hz = tone_hz

    @property
    def voices(self) -> tuple[str, ...]:
        return ("af_heart",)

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> AudioResult:
        del speed, language
        if voice not in self.voices:
            raise UnsupportedVoiceError(f"Voice '{voice}' is not available.")
        self.call_count += 1
        self.received_texts.append(text)
        sample_rate = 24_000
        frames = np.arange(2_400, dtype=np.float32)
        samples = (
            0.1 * np.sin(2 * np.pi * self._tone_hz * frames / sample_rate)
        ).astype(np.float32)
        return AudioResult(samples=samples, sample_rate_hz=sample_rate)


@dataclass(frozen=True)
class ApiHarness:
    app: FastAPI
    loader: ModelLoader
    engines: dict[str, FakeInferenceEngine]
    audio_dir: Path


class StepClock:
    """Deterministic monotonic clock for duration assertions."""

    def __init__(self, step_seconds: float) -> None:
        self._step_seconds = step_seconds
        self._current = 0.0

    def __call__(self) -> float:
        current = self._current
        self._current += self._step_seconds
        return current


@pytest.fixture
def harness(tmp_path: Path) -> ApiHarness:
    fp32_path = tmp_path / "kokoro-v1.0.onnx"
    quantized_path = tmp_path / "kokoro-v1.0.int8.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    fp32_path.write_bytes(b"test-fp32-model")
    quantized_path.write_bytes(b"test-int8-model")
    voices_path.write_bytes(b"test-voices")

    definitions = (
        ModelDefinition(
            model_id="kokoro-fp32",
            display_name="Kokoro",
            precision="FP32",
            variant="fp32",
            model_version="1.0",
            model_path=fp32_path,
            voices_path=voices_path,
            voices=("af_heart",),
        ),
        ModelDefinition(
            model_id="kokoro-q8",
            display_name="Kokoro",
            precision="INT8",
            variant="quantized",
            model_version="1.0",
            model_path=quantized_path,
            voices_path=voices_path,
            voices=("af_heart",),
        ),
    )
    registry = ModelRegistry(definitions)
    engines = {
        "kokoro-fp32": FakeInferenceEngine(440),
        "kokoro-q8": FakeInferenceEngine(330),
    }
    loader = ModelLoader(engine_factory=lambda model: engines[model.model_id])
    metrics = MetricsCollector(
        clock=StepClock(0.025),
        memory_reader=lambda: 384.0,
    )
    audio_dir = tmp_path / "audio"
    audio = AudioService(audio_dir)
    app = create_app(
        Settings(environment="test"),
        model_registry=registry,
        model_loader=loader,
        audio_service=audio,
        metrics_collector=metrics,
    )
    return ApiHarness(app=app, loader=loader, engines=engines, audio_dir=audio_dir)


def request(
    app: FastAPI,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> httpx.Response:
    """Send one request directly to an isolated ASGI application."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


def test_health_endpoint(harness: ApiHarness) -> None:
    response = request(harness.app, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_listing(harness: ApiHarness) -> None:
    response = request(harness.app, "GET", "/api/models")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "kokoro-fp32",
            "name": "Kokoro",
            "precision": "FP32",
            "variant": "fp32",
            "voices": ["af_heart"],
            "modelVersion": "1.0",
            "runtime": "ONNX",
            "hosting": "self-hosted",
            "externalInferenceApis": [],
            "available": True,
            "unavailableReason": None,
            "description": "Local open-weight text-to-speech model.",
        },
        {
            "id": "kokoro-q8",
            "name": "Kokoro",
            "precision": "INT8",
            "variant": "quantized",
            "voices": ["af_heart"],
            "modelVersion": "1.0",
            "runtime": "ONNX",
            "hosting": "self-hosted",
            "externalInferenceApis": [],
            "available": True,
            "unavailableReason": None,
            "description": "Local open-weight text-to-speech model.",
        },
    ]


def test_default_catalog_exposes_all_model_options_without_claiming_missing_artifacts(
    tmp_path: Path,
) -> None:
    settings = Settings(
        environment="test",
        model_artifact_dir=tmp_path / "models",
        generated_audio_dir=tmp_path / "audio",
        benchmark_result_dir=tmp_path / "benchmarks",
        stage11_artifact_root=tmp_path / "stage11",
        stage11_approach_run_root=tmp_path / "agent-runs",
        stage11_manifest_root=tmp_path / "manifests",
        stage12_artifact_root=tmp_path / "stage12",
        experiment_model_cache_dir=tmp_path / "model-cache",
        experiment_speaker_profile_dir=tmp_path / "speaker-profile",
    )
    response = request(create_app(settings), "GET", "/api/models")

    assert response.status_code == 200
    models = response.json()
    assert [model["id"] for model in models] == [
        "kokoro-fp32",
        "kokoro-fp16",
        "kokoro-q8",
        "audio8-0.6b",
        "speecht5-pretrained",
    ]
    assert not any(model["available"] for model in models)
    audio8 = next(model for model in models if model["id"] == "audio8-0.6b")
    assert "review-required remote model code" in audio8["unavailableReason"]


def test_valid_synthesis_request_returns_playable_wav(harness: ApiHarness) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "Hello world",
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "kokoro-fp32"
    assert payload["text"] == "Hello world"
    assert payload["normalizedText"] == "Hello world"
    assert payload["audioUrl"].startswith("/audio/kokoro-fp32-af_heart-")
    assert payload["metrics"] == {
        "modelLoadMs": 25.0,
        "inferenceMs": 25.0,
        "audioDurationMs": 100.0,
        "realTimeFactor": 0.25,
        "memoryMb": 384.0,
        "warm": False,
        "modelVariant": "fp32",
    }
    assert payload["metrics"]["realTimeFactor"] == pytest.approx(
        payload["metrics"]["inferenceMs"]
        / payload["metrics"]["audioDurationMs"],
        abs=1e-6,
    )
    assert (harness.audio_dir / Path(payload["audioUrl"]).name).is_file()

    audio_response = request(harness.app, "GET", payload["audioUrl"])
    assert audio_response.status_code == 200
    assert audio_response.headers["content-type"] == "audio/wav"
    with wave.open(BytesIO(audio_response.content), "rb") as audio_file:
        assert audio_file.getframerate() == 24_000
        assert audio_file.getnframes() == 2_400


def test_invalid_synthesis_request_is_rejected(harness: ApiHarness) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "",
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "text"]


def test_sanitization_is_enabled_by_default_and_reaches_inference(
    harness: ApiHarness,
) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "Hello ./ --- world",
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
        },
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Hello ./ --- world"
    assert response.json()["normalizedText"] == "Hello world"
    assert harness.engines["kokoro-fp32"].received_texts == ["Hello world"]
    assert response.json()["metrics"]["inferenceMs"] == 25.0


def test_normalization_is_enabled_by_default_and_reaches_inference(
    harness: ApiHarness,
) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "Save 15% at https://example.com. Price: $25.",
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
        },
    )

    expected = "Save 15 percent at example dot com. Price: 25 dollars."
    assert response.status_code == 200
    assert response.json()["normalizedText"] == expected
    assert harness.engines["kokoro-fp32"].received_texts == [expected]
    assert response.json()["metrics"]["inferenceMs"] == 25.0


def test_sanitization_can_be_disabled_without_changing_inference_text(
    harness: ApiHarness,
) -> None:
    raw_text = "  Keep ./ -- $25 exactly.  "
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": raw_text,
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
            "sanitizeText": False,
            "normalizeText": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["normalizedText"] == raw_text
    assert harness.engines["kokoro-fp32"].received_texts == [raw_text]


@pytest.mark.parametrize(
    ("sanitize_text", "normalize_text", "expected"),
    (
        (True, True, "Price: 25 dollars today"),
        (True, False, "Price: 25 today"),
        (False, True, "Price: 25 dollars ,,, today"),
        (False, False, "Price: $25 ,,, today"),
    ),
)
def test_preprocessing_toggles_are_independent_api_options(
    harness: ApiHarness,
    sanitize_text: bool,
    normalize_text: bool,
    expected: str,
) -> None:
    source = "Price: $25 ,,, today"
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": source,
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
            "sanitizeText": sanitize_text,
            "normalizeText": normalize_text,
        },
    )

    assert response.status_code == 200
    assert response.json()["text"] == source
    assert response.json()["normalizedText"] == expected
    assert harness.engines["kokoro-fp32"].received_texts == [expected]
    assert response.json()["metrics"]["inferenceMs"] == 25.0


def test_noise_only_input_returns_422_before_model_loading(harness: ApiHarness) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "./ -- ,,, $ %",
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "Sanitization removed all speakable content. "
            "Enter words or disable sanitization."
        )
    }
    assert harness.loader.load_count("kokoro-fp32") == 0
    assert harness.engines["kokoro-fp32"].received_texts == []


def test_warm_requests_reuse_loaded_model(harness: ApiHarness) -> None:
    payload = {
        "text": "OpenVoice Lab is running locally.",
        "modelId": "kokoro-fp32",
        "voiceId": "af_heart",
    }

    first = request(harness.app, "POST", "/api/synthesis", payload)
    second = request(harness.app, "POST", "/api/synthesis", payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["audioUrl"] == second.json()["audioUrl"]
    assert first.json()["metrics"]["warm"] is False
    assert second.json()["metrics"]["warm"] is True
    assert first.json()["metrics"]["modelLoadMs"] == 25.0
    assert second.json()["metrics"]["modelLoadMs"] == 0.0
    assert harness.loader.load_count("kokoro-fp32") == 1
    assert harness.engines["kokoro-fp32"].call_count == 2


def test_audio_identity_uses_the_final_inference_text(harness: ApiHarness) -> None:
    common = {
        "modelId": "kokoro-fp32",
        "voiceId": "af_heart",
    }

    technical = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {**common, "text": "Price: $25."},
    )
    already_spoken = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {**common, "text": "Price: 25 dollars."},
    )

    assert technical.status_code == already_spoken.status_code == 200
    assert technical.json()["normalizedText"] == "Price: 25 dollars."
    assert already_spoken.json()["normalizedText"] == "Price: 25 dollars."
    assert technical.json()["audioUrl"] == already_spoken.json()["audioUrl"]
    assert harness.engines["kokoro-fp32"].received_texts == [
        "Price: 25 dollars.",
        "Price: 25 dollars.",
    ]


def test_same_request_runs_against_both_registered_variants(harness: ApiHarness) -> None:
    base_payload = {
        "text": "Compare the same synthesis request.",
        "voiceId": "af_heart",
    }

    fp32 = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {**base_payload, "modelId": "kokoro-fp32"},
    )
    quantized = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {**base_payload, "modelId": "kokoro-q8"},
    )

    assert fp32.status_code == quantized.status_code == 200
    assert fp32.json()["model"] == "kokoro-fp32"
    assert quantized.json()["model"] == "kokoro-q8"
    assert fp32.json()["metrics"]["modelVariant"] == "fp32"
    assert quantized.json()["metrics"]["modelVariant"] == "quantized"
    assert fp32.json()["audioUrl"] != quantized.json()["audioUrl"]
    assert harness.loader.load_count("kokoro-fp32") == 1
    assert harness.loader.load_count("kokoro-q8") == 1


def test_unknown_model_returns_useful_error(harness: ApiHarness) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "Hello world",
            "modelId": "unknown",
            "voiceId": "af_heart",
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Model 'unknown' is not registered."}
