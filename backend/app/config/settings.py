"""Typed process configuration."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings without inference-specific options."""

    model_config = SettingsConfigDict(
        env_prefix="OPENVOICE_",
        env_file=".env",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    model_artifact_dir: Path = Path("model-artifacts")
    generated_audio_dir: Path = Path("artifacts/audio")
    default_model_id: str = "kokoro-fp32"
    quantized_model_id: str = "kokoro-q8"
    default_model_display_name: str = "Kokoro"
    default_voice_id: str = "af_heart"
    kokoro_model_filename: str = "kokoro-v1.0.onnx"
    kokoro_quantized_model_filename: str = "kokoro-v1.0.int8.onnx"
    kokoro_voices_filename: str = "voices-v1.0.bin"
    kokoro_model_version: str = "1.0"
    kokoro_language: str = "en-us"
    kokoro_speed: float = 1.0
    audio_url_prefix: str = "/audio"
