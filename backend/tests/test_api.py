"""Executable tests for the Stage 1 HTTP contract."""

import asyncio

import httpx

from app.config.settings import Settings
from app.main import create_app


def request(
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> httpx.Response:
    """Send one request directly to an isolated ASGI application."""

    async def send() -> httpx.Response:
        app = create_app(Settings(environment="test"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


def test_health_endpoint() -> None:
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_listing() -> None:
    response = request("GET", "/api/models")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "kokoro",
            "displayName": "Kokoro",
            "voices": ["af_heart"],
            "variants": ["fp32", "quantized"],
        }
    ]


def test_valid_synthesis_request_returns_deterministic_mock() -> None:
    response = request(
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
    assert response.json() == {
        "status": "mock",
        "model": "kokoro-fp32",
        "text": "Hello world",
        "audioUrl": None,
    }


def test_invalid_synthesis_request_is_rejected() -> None:
    response = request(
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
