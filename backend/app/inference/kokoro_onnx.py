"""Self-hosted Kokoro ONNX inference engine."""

from pathlib import Path
from threading import Lock

import numpy as np
from kokoro_onnx import Kokoro

from app.inference.base import (
    AudioResult,
    InferenceError,
    TTSInferenceEngine,
    UnsupportedVoiceError,
)


class KokoroONNXEngine(TTSInferenceEngine):
    """Run Kokoro locally through one long-lived ONNX Runtime session."""

    def __init__(self, model_path: Path, voices_path: Path) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        try:
            self._runtime = Kokoro(str(model_path), str(voices_path))
            self._voices = tuple(self._runtime.get_voices())
        except Exception as error:
            raise InferenceError(f"Failed to initialize Kokoro ONNX: {error}") from error
        self._inference_lock = Lock()

    @property
    def voices(self) -> tuple[str, ...]:
        return self._voices

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> AudioResult:
        """Generate validated mono samples with the loaded local session."""
        if voice not in self._voices:
            raise UnsupportedVoiceError(f"Kokoro voice '{voice}' is not available.")

        try:
            with self._inference_lock:
                samples, sample_rate = self._runtime.create(
                    text,
                    voice=voice,
                    speed=speed,
                    lang=language,
                )
        except Exception as error:
            raise InferenceError(f"Kokoro ONNX synthesis failed: {error}") from error

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or sample_rate <= 0:
            raise InferenceError("Kokoro ONNX returned an invalid audio buffer.")
        return AudioResult(samples=audio, sample_rate_hz=int(sample_rate))


# Existing imports can migrate without coupling callers to a new filename.
KokoroOnnxAdapter = KokoroONNXEngine
