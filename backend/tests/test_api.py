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

    def __init__(self) -> None:
        self.call_count = 0

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
        del text, speed, language
        if voice not in self.voices:
            raise UnsupportedVoiceError(f"Voice '{voice}' is not available.")
        self.call_count += 1
        sample_rate = 24_000
        frames = np.arange(2_400, dtype=np.float32)
        samples = (0.1 * np.sin(2 * np.pi * 440 * frames / sample_rate)).astype(np.float32)
        return AudioResult(samples=samples, sample_rate_hz=sample_rate)


@dataclass(frozen=True)
class ApiHarness:
    app: FastAPI
    loader: ModelLoader
    engine: FakeInferenceEngine
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
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"test-model")
    voices_path.write_bytes(b"test-voices")

    definition = ModelDefinition(
        model_id="kokoro",
        display_name="Kokoro",
        variant="fp32",
        model_version="1.0",
        model_path=model_path,
        voices_path=voices_path,
        voices=("af_heart",),
    )
    registry = ModelRegistry((definition,))
    engine = FakeInferenceEngine()
    loader = ModelLoader(engine_factory=lambda _model: engine)
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
    return ApiHarness(app=app, loader=loader, engine=engine, audio_dir=audio_dir)


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
            "id": "kokoro",
            "displayName": "Kokoro",
            "voices": ["af_heart"],
            "variants": ["fp32"],
            "modelVersion": "1.0",
            "runtime": "ONNX",
            "hosting": "self-hosted",
            "externalInferenceApis": [],
            "available": True,
        }
    ]


def test_valid_synthesis_request_returns_playable_wav(harness: ApiHarness) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "Hello world",
            "modelId": "kokoro",
            "voiceId": "af_heart",
            "variant": "fp32",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "kokoro-fp32"
    assert payload["text"] == "Hello world"
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
            "modelId": "kokoro",
            "voiceId": "af_heart",
            "variant": "fp32",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "text"]


def test_warm_requests_reuse_loaded_model(harness: ApiHarness) -> None:
    payload = {
        "text": "OpenVoice Lab is running locally.",
        "modelId": "kokoro",
        "voiceId": "af_heart",
        "variant": "fp32",
    }

    first = request(harness.app, "POST", "/api/synthesis", payload)
    second = request(harness.app, "POST", "/api/synthesis", payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["audioUrl"] == second.json()["audioUrl"]
    assert first.json()["metrics"]["warm"] is False
    assert second.json()["metrics"]["warm"] is True
    assert first.json()["metrics"]["modelLoadMs"] == 25.0
    assert second.json()["metrics"]["modelLoadMs"] == 0.0
    assert harness.loader.load_count("kokoro", "fp32") == 1
    assert harness.engine.call_count == 2


def test_unknown_model_returns_useful_error(harness: ApiHarness) -> None:
    response = request(
        harness.app,
        "POST",
        "/api/synthesis",
        {
            "text": "Hello world",
            "modelId": "unknown",
            "voiceId": "af_heart",
            "variant": "fp32",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Model 'unknown' with variant 'fp32' is not registered."
    }
