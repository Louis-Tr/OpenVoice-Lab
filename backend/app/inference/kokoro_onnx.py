"""Self-hosted Kokoro ONNX inference engine."""

from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from app.inference.base import (
    AudioResult,
    InferenceError,
    TTSInferenceEngine,
    UnsupportedVoiceError,
)

# eSpeak uses process-global native state, including across model variants.
_phonemization_lock = Lock()


class KokoroONNXEngine(TTSInferenceEngine):
    """Run Kokoro locally through one long-lived ONNX Runtime session."""

    def __init__(self, model_path: Path, voices_path: Path, *, cpu_threads: int = 1) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            with _phonemization_lock:
                self._runtime = Kokoro.from_session(session, str(voices_path))
            self._voices = tuple(self._runtime.get_voices())
        except Exception as error:
            raise InferenceError(f"Failed to initialize Kokoro ONNX: {error}") from error

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
            with _phonemization_lock:
                phonemes = self._runtime.tokenizer.phonemize(text, language)
            samples, sample_rate = self._runtime.create(
                phonemes,
                voice=voice,
                speed=speed,
                lang=language,
                is_phonemes=True,
            )
        except Exception as error:
            raise InferenceError(f"Kokoro ONNX synthesis failed: {error}") from error

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or sample_rate <= 0:
            raise InferenceError("Kokoro ONNX returned an invalid audio buffer.")
        return AudioResult(samples=audio, sample_rate_hz=int(sample_rate))


# Existing imports can migrate without coupling callers to a new filename.
KokoroOnnxAdapter = KokoroONNXEngine
