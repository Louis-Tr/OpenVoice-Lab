"""SpeechT5 adapter for the product synthesis abstraction."""

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.inference.base import (
    AudioResult,
    InferenceError,
    TTSInferenceEngine,
    UnsupportedVoiceError,
)


class SpeechT5InferenceEngine(TTSInferenceEngine):
    """Serve one pinned SpeechT5 model and verified speaker profile on CPU."""

    def __init__(
        self,
        *,
        model_root: Path,
        vocoder_root: Path,
        speaker_embedding_path: Path,
        voice_id: str,
    ) -> None:
        try:
            import torch
            from transformers import (
                SpeechT5ForTextToSpeech,
                SpeechT5HifiGan,
                SpeechT5Processor,
            )

            self._torch = torch
            self._processor = SpeechT5Processor.from_pretrained(
                model_root, local_files_only=True
            )
            self._model = SpeechT5ForTextToSpeech.from_pretrained(
                model_root, local_files_only=True
            ).to("cpu").eval()
            self._vocoder = SpeechT5HifiGan.from_pretrained(
                vocoder_root, local_files_only=True
            ).to("cpu").eval()
            values = np.load(speaker_embedding_path, allow_pickle=False).astype(np.float32)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("speaker embedding must be a finite one-dimensional vector")
            self._speaker = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        except Exception as error:
            raise InferenceError(f"Failed to initialize SpeechT5: {error}") from error
        self._voices = (voice_id,)
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
        del language
        if voice not in self._voices:
            raise UnsupportedVoiceError(f"Voice '{voice}' is not available.")
        if speed != 1.0:
            raise InferenceError("SpeechT5 does not expose speed control in this profile.")
        try:
            with self._inference_lock:
                self._torch.manual_seed(42)
                inputs = self._processor(text=text, return_tensors="pt")
                with self._torch.inference_mode():
                    waveform: Any = self._model.generate_speech(
                        inputs["input_ids"],
                        self._speaker,
                        vocoder=self._vocoder,
                        attention_mask=inputs.get("attention_mask"),
                    )
            samples = waveform.detach().float().cpu().numpy().astype(np.float32)
        except Exception as error:
            raise InferenceError(f"SpeechT5 synthesis failed: {error}") from error
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise InferenceError("SpeechT5 returned an invalid audio buffer.")
        return AudioResult(samples=samples, sample_rate_hz=16_000)
