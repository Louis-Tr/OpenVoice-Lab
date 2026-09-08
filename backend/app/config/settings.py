"""Typed process configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    benchmark_result_dir: Path = Path("benchmark-results")
    frontend_dist_dir: Path | None = None
    default_model_id: str = "kokoro-fp32"
    fp16_model_id: str = "kokoro-fp16"
    quantized_model_id: str = "kokoro-q8"
    default_model_display_name: str = "Kokoro"
    default_voice_id: str = "af_heart"
    kokoro_model_filename: str = "kokoro-v1.0.onnx"
    kokoro_fp16_model_filename: str = "kokoro-v1.0.fp16.onnx"
    kokoro_quantized_model_filename: str = "kokoro-v1.0.int8.onnx"
    kokoro_voices_filename: str = "voices-v1.0.bin"
    kokoro_model_version: str = "1.0"
    kokoro_language: str = "en-us"
    kokoro_speed: float = 1.0
    speecht5_model_id: str = "speecht5-pretrained"
    speecht5_model_dirname: str = "speecht5-tts"
    speecht5_vocoder_dirname: str = "speecht5-hifigan"
    speecht5_speaker_filename: str = "speecht5-speakers/cmu-slt.npy"
    speecht5_voice_id: str = "cmu-slt"
    product_maximum_cached_models: int | None = None
    product_cpu_threads: int | None = None
    product_cpu_units: int | None = None
    product_memory_limit_mb: float | None = None
    product_memory_headroom_mb: float = 256
    synthesis_job_maximum_records: int = Field(default=1000, ge=1)
    synthesis_job_retention_seconds: int = Field(default=86400, ge=1)
    audio_url_prefix: str = "/audio"
    stage11_artifact_root: Path = Path("../artifacts/stage11/full-training")
    stage11_approach_run_root: Path = Path("../artifacts/stage11/agent-runs")
    stage11_manifest_root: Path = Path("../data-processing/manifests/stage11")
    stage12_artifact_root: Path = Path("../artifacts/stage12")
    experiment_audio_url_prefix: str = "/experiment-audio"
    experiment_model_cache_dir: Path = Path("../artifacts/stage12/model-cache")
    experiment_speaker_profile_dir: Path = Path("../artifacts/stage12/serving-profile")
    experiment_maximum_queued_jobs: int = 2
    experiment_maximum_cached_models: int = 2
    experiment_cpu_threads: int | None = None
    experiment_model_base_url: str | None = None
    experiment_model_access_token: SecretStr | None = None
    experiment_remote_manifest_path: Path = Path(
        "app/experiments/snapshots/remote_models.json"
    )
    experiment_asr_dirname: str = "whisper-small-en"
    speecht5_revision: str = "30fcde30f19b87502b8435427b5f5068e401d5f6"
    speecht5_vocoder_revision: str = "bb6f429406e86a9992357a972c0698b22043307d"
