"""SpeechT5 adapter for the product synthesis abstraction."""

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.inference.base import (
    AudioResult,
    InferenceError,
    InputTooLongError,
    TTSInferenceEngine,
    UnsupportedVoiceError,
)
from app.inference.cpu import configure_torch_threads
from app.inference.speecht5_random import _generator, install_request_dropout
from app.inference.text_chunks import split_text


class SpeechT5InferenceEngine(TTSInferenceEngine):
    """Serve one pinned SpeechT5 model and verified speaker profile on CPU."""

    def __init__(
        self,
        *,
        model_root: Path,
        vocoder_root: Path,
        speaker_embedding_path: Path,
        voice_id: str,
        cpu_threads: int = 1,
    ) -> None:
        try:
            import torch

            configure_torch_threads(torch, cpu_threads)
            from transformers import (
                SpeechT5ForTextToSpeech,
                SpeechT5HifiGan,
                SpeechT5Processor,
            )

            self._torch = torch
            self._processor = SpeechT5Processor.from_pretrained(model_root, local_files_only=True)
            self._model = (
                SpeechT5ForTextToSpeech.from_pretrained(model_root, local_files_only=True)
                .to("cpu")
                .eval()
            )
            install_request_dropout(self._model, torch)
            self._vocoder = (
                SpeechT5HifiGan.from_pretrained(vocoder_root, local_files_only=True)
                .to("cpu")
                .eval()
            )
            values = np.load(speaker_embedding_path, allow_pickle=False).astype(np.float32)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("speaker embedding must be a finite one-dimensional vector")
            self._speaker = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        except Exception as error:
            raise InferenceError(f"Failed to initialize SpeechT5: {error}") from error
        self._voices = (voice_id,)
        self._processor_lock = Lock()

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
            with self._processor_lock:
                token_limit = min(
                    self._processor.tokenizer.model_max_length,
                    self._model.config.max_text_positions,
                )
                chunks = split_text(
                    text,
                    lambda chunk: len(self._processor.tokenizer(chunk, verbose=False)["input_ids"]),
                    token_limit,
                )
                prepared = [
                    self._processor(text=chunk, return_tensors="pt", verbose=False)
                    for chunk in chunks
                ]
                # Keep the model boundary checked even if processor behavior changes.
                if any(inputs["input_ids"].shape[-1] > token_limit for inputs in prepared):
                    raise InputTooLongError(f"SpeechT5 chunk exceeds {token_limit} tokens.")
            generator = self._torch.Generator(device="cpu").manual_seed(42)
            token = _generator.set(generator)
            try:
                parts = []
                with self._torch.inference_mode():
                    for inputs in prepared:
                        waveform: Any = self._model.generate_speech(
                            inputs["input_ids"],
                            self._speaker,
                            vocoder=self._vocoder,
                            attention_mask=inputs.get("attention_mask"),
                        )
                        samples = waveform.detach().float().cpu().numpy().astype(np.float32)
                        del waveform
                        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
                            raise InferenceError("SpeechT5 returned an invalid audio buffer.")
                        parts.append(samples)
            finally:
                _generator.reset(token)
            samples = parts[0] if len(parts) == 1 else np.concatenate(parts)
        except InputTooLongError:
            raise
        except Exception as error:
            raise InferenceError(f"SpeechT5 synthesis failed: {error}") from error
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise InferenceError("SpeechT5 returned an invalid audio buffer.")
        return AudioResult(samples=samples, sample_rate_hz=16_000)
