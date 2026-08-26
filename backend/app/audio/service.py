"""Deterministic WAV encoding and local artifact storage."""

import re
import wave
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from app.inference.base import AudioResult


class AudioStorageError(RuntimeError):
    """Raised when generated audio cannot be encoded or persisted."""


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    """Stored, encoded audio metadata."""

    url: str
    filename: str
    duration_seconds: float


class AudioService:
    """Encode and persist request-addressed audio outside inference engines."""

    def __init__(self, output_dir: Path, public_url_prefix: str = "/audio") -> None:
        self._output_dir = output_dir
        self._public_url_prefix = public_url_prefix.rstrip("/")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def create_artifact(
        self,
        audio: AudioResult,
        *,
        model: str,
        voice: str,
        artifact_key: str,
    ) -> AudioArtifact:
        """Write a stable request-addressed WAV and return its public location."""
        try:
            wav_bytes = self._encode_wav(audio)
            digest = sha256(artifact_key.encode("utf-8")).hexdigest()[:16]
            safe_model = self._safe_name(model)
            safe_voice = self._safe_name(voice)
            filename = f"{safe_model}-{safe_voice}-{digest}.wav"
            destination = self._output_dir / filename
            if not destination.exists():
                try:
                    with destination.open("xb") as output:
                        output.write(wav_bytes)
                except FileExistsError:
                    pass
        except (OSError, ValueError) as error:
            raise AudioStorageError(f"Failed to store generated audio: {error}") from error

        return AudioArtifact(
            url=f"{self._public_url_prefix}/{filename}",
            filename=filename,
            duration_seconds=audio.duration_seconds,
        )

    @staticmethod
    def _encode_wav(audio: AudioResult) -> bytes:
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("Audio samples must be a non-empty mono buffer.")
        pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio.sample_rate_hz)
            wav_file.writeframes(pcm.tobytes())
        return output.getvalue()

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
