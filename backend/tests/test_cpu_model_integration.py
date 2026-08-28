"""Real integration acceptance for CPU-compatible Audio8 and SpeechT5."""

import asyncio
import importlib.util
import wave
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.main import create_app, resolve_backend_path
from app.schemas.synthesis import SynthesisRequest


@pytest.mark.integration
def test_audio8_int4_and_speecht5_generate_playable_cpu_audio(tmp_path: Path) -> None:
    required_modules = ("onnxruntime", "tokenizers", "torch", "transformers", "sentencepiece")
    if not all(importlib.util.find_spec(module) for module in required_modules):
        pytest.skip("Install the serving dependency profile and CPU PyTorch.")

    defaults = Settings(environment="test")
    artifact_root = resolve_backend_path(defaults.model_artifact_dir)
    required_artifacts = (
        artifact_root / defaults.audio8_model_dirname / "runtime_manifest.json",
        artifact_root / defaults.audio8_model_dirname / "slow_ar_int4.onnx.data",
        artifact_root / defaults.speecht5_model_dirname / "pytorch_model.bin",
        artifact_root / defaults.speecht5_vocoder_dirname / "pytorch_model.bin",
        artifact_root / defaults.speecht5_speaker_filename,
    )
    if not all(path.is_file() for path in required_artifacts):
        pytest.skip("Run scripts/download_cpu_models.py to provision CPU models.")

    output = tmp_path / "audio"
    app = create_app(
        Settings(
            environment="test",
            model_artifact_dir=artifact_root,
            generated_audio_dir=output,
            benchmark_result_dir=tmp_path / "benchmarks",
            stage11_artifact_root=tmp_path / "stage11",
            stage11_approach_run_root=tmp_path / "agent-runs",
            stage11_manifest_root=tmp_path / "manifests",
            stage12_artifact_root=tmp_path / "stage12",
            product_maximum_cached_models=1,
            product_cpu_threads=2,
        )
    )

    async def synthesize(request: SynthesisRequest):
        return await app.state.synthesis_service.synthesize(request)

    audio8 = asyncio.run(
        synthesize(
            SynthesisRequest(
                text="Hello from Audio eight.",
                model_id="audio8-0.6b",
                voice_id="unconditioned",
            )
        )
    )
    speecht5 = asyncio.run(
        synthesize(
            SynthesisRequest(
                text="OpenVoice Lab runs Speech T five on CPU.",
                model_id="speecht5-pretrained",
                voice_id="cmu-slt",
            )
        )
    )

    assert audio8.status == speecht5.status == "ok"
    assert audio8.metrics.model_variant == "audio8"
    assert speecht5.metrics.model_variant == "pretrained"
    for result, expected_rate in ((audio8, 44_100), (speecht5, 16_000)):
        assert result.audio_url is not None
        audio_path = output / Path(result.audio_url).name
        assert audio_path.is_file()
        with wave.open(str(audio_path), "rb") as audio_file:
            assert audio_file.getframerate() == expected_rate
            assert audio_file.getnframes() > 0
    assert app.state.model_loader.load_count("audio8-0.6b") == 1
    assert app.state.model_loader.load_count("speecht5-pretrained") == 1
