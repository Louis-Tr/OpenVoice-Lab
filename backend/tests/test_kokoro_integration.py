"""Acceptance test for real, locally hosted Kokoro ONNX inference."""

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
        artifact_dir / settings.kokoro_voices_filename,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Run scripts/download_models.py to provision Kokoro artifacts.")

    app = create_app(settings)
    payload = {
        "text": "OpenVoice Lab is running locally.",
        "modelId": "kokoro",
        "voiceId": "af_heart",
        "variant": "fp32",
    }

    first = request(app, "POST", "/api/synthesis", payload)
    second = request(app, "POST", "/api/synthesis", payload)

    assert first.status_code == second.status_code == 200
    first_url = first.json()["audioUrl"]
    assert first_url == second.json()["audioUrl"]
    assert app.state.model_loader.load_count("kokoro", "fp32") == 1

    first_audio = request(app, "GET", first_url)
    second_audio = request(app, "GET", first_url)
    assert first_audio.status_code == second_audio.status_code == 200
    assert sha256(first_audio.content).digest() == sha256(second_audio.content).digest()

    with wave.open(BytesIO(first_audio.content), "rb") as audio_file:
        assert audio_file.getframerate() == 24_000
        assert audio_file.getnframes() > 0

    output_file = resolve_backend_path(settings.generated_audio_dir) / Path(first_url).name
    assert output_file.is_file()
