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
    default_model_id: str = "kokoro"
    default_model_display_name: str = "Kokoro"
    default_voice_id: str = "af_heart"
