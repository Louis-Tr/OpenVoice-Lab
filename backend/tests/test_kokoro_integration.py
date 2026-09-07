"""Acceptance tests for real, locally hosted Kokoro ONNX variants."""

import asyncio
import wave
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.config.settings import Settings
from app.main import create_app, resolve_backend_path


def request(
    app: FastAPI,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


@pytest.mark.integration
def test_real_kokoro_audio_is_playable_reproducible_and_warm() -> None:
    settings = Settings(environment="test")
    artifact_dir = resolve_backend_path(settings.model_artifact_dir)
    required = (
        artifact_dir / settings.kokoro_model_filename,
        artifact_dir / settings.kokoro_fp16_model_filename,
        artifact_dir / settings.kokoro_voices_filename,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Run scripts/download_models.py to provision Kokoro artifacts.")

    app = create_app(settings)
    base_payload = {
        "text": "OpenVoice Lab is running locally.",
        "voiceId": "af_heart",
    }

    fp32_payload = {**base_payload, "modelId": "kokoro-fp32"}
    fp16_payload = {**base_payload, "modelId": "kokoro-fp16"}
    first = request(app, "POST", "/api/synthesis", fp32_payload)
    second = request(app, "POST", "/api/synthesis", fp32_payload)
    fp16_first = request(app, "POST", "/api/synthesis", fp16_payload)
    fp16_second = request(app, "POST", "/api/synthesis", fp16_payload)

    assert (
        first.status_code
        == second.status_code
        == fp16_first.status_code
        == fp16_second.status_code
        == 200
    )
    first_payload = first.json()
    second_payload = second.json()
    fp16_first_payload = fp16_first.json()
    fp16_second_payload = fp16_second.json()
    first_url = first_payload["audioUrl"]
    assert first_url == second_payload["audioUrl"]
    assert fp16_first_payload["audioUrl"] == fp16_second_payload["audioUrl"]
    assert first_url != fp16_first_payload["audioUrl"]
    assert app.state.model_loader.load_count("kokoro-fp32") == 1
    assert app.state.model_loader.load_count("kokoro-fp16") == 1
    assert first_payload["metrics"]["warm"] is False
    assert second_payload["metrics"]["warm"] is True
    assert fp16_first_payload["metrics"]["warm"] is False
    assert fp16_second_payload["metrics"]["warm"] is True
    assert first_payload["metrics"]["modelVariant"] == "fp32"
    assert fp16_first_payload["metrics"]["modelVariant"] == "fp16"
    assert second_payload["metrics"]["modelLoadMs"] == 0.0
    assert fp16_second_payload["metrics"]["modelLoadMs"] == 0.0
    for measured in (
        first_payload["metrics"],
        second_payload["metrics"],
        fp16_first_payload["metrics"],
        fp16_second_payload["metrics"],
    ):
        assert measured["inferenceMs"] > 0
        assert measured["audioDurationMs"] > 0
        assert measured["memoryMb"] > 0
        assert measured["realTimeFactor"] == pytest.approx(
            measured["inferenceMs"] / measured["audioDurationMs"],
            abs=1e-6,
        )

    first_audio = request(app, "GET", first_url)
    second_audio = request(app, "GET", first_url)
    assert first_audio.status_code == second_audio.status_code == 200
    assert sha256(first_audio.content).digest() == sha256(second_audio.content).digest()

    with wave.open(BytesIO(first_audio.content), "rb") as audio_file:
        assert audio_file.getframerate() == 24_000
        assert audio_file.getnframes() > 0

    fp16_audio = request(app, "GET", fp16_first_payload["audioUrl"])
    assert fp16_audio.status_code == 200
    with wave.open(BytesIO(fp16_audio.content), "rb") as audio_file:
        assert audio_file.getframerate() == 24_000
        assert audio_file.getnframes() > 0

    synthetic_text = (
        "Email dev.team@example.com -- open ./docs/api-guide.md.\n"
        "The price is $25, with a 15% discount."
    )
    synthetic = request(
        app,
        "POST",
        "/api/synthesis",
        {
            "text": synthetic_text,
            "modelId": "kokoro-fp32",
            "voiceId": "af_heart",
        },
    )
    assert synthetic.status_code == 200
    synthetic_payload = synthetic.json()
    assert synthetic_payload["normalizedText"] == (
        "Email dev dot team at example dot com—open docs slash api guide dot M D. "
        "The price is 25 dollars, with a 15 percent discount."
    )
    synthetic_audio = request(app, "GET", synthetic_payload["audioUrl"])
    assert synthetic_audio.status_code == 200
    with wave.open(BytesIO(synthetic_audio.content), "rb") as audio_file:
        assert audio_file.getframerate() == 24_000
        assert audio_file.getnframes() > 0

    output_file = resolve_backend_path(settings.generated_audio_dir) / Path(first_url).name
    assert output_file.is_file()
