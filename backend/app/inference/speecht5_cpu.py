"""Lazy, bounded CPU runtime for pinned SpeechT5 comparison models."""

from __future__ import annotations

import gc
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import numpy as np
import psutil

from app.experiments.common import ExperimentEvidenceError
from app.experiments.model_registry import ExperimentModelDefinition


@dataclass(frozen=True, slots=True)
class SpeechT5SynthesisOutput:
    """Measured audio output before ASR scoring."""

    audio_path: Path
    model_load_ms: float
    inference_ms: float
    audio_duration_ms: float
    real_time_factor: float
    process_memory_mb: float
    warm: bool


class ExperimentInferenceRuntime(Protocol):
    """Replaceable boundary used by the experiment job service."""

    def prepare(self) -> None: ...

    def synthesize(
        self, definition: ExperimentModelDefinition, text: str, output_path: Path
    ) -> SpeechT5SynthesisOutput: ...

    def transcribe(self, audio_path: Path) -> tuple[str, float]: ...


class SpeechT5CpuRuntime:
    """Run SpeechT5 and Whisper locally while bounding cached model memory."""

    def __init__(
        self,
        *,
        vocoder_root: Path,
        asr_root: Path,
        speaker_embedding_path: Path,
        maximum_cached_models: int = 2,
        cpu_threads: int | None = None,
    ) -> None:
        if maximum_cached_models < 1:
            raise ValueError("maximum_cached_models must be positive")
        self._vocoder_root = vocoder_root.resolve()
        self._asr_root = asr_root.resolve()
        self._speaker_embedding_path = speaker_embedding_path.resolve()
        self._maximum_cached_models = maximum_cached_models
        self._cpu_threads = cpu_threads
        self._models: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._vocoder: Any | None = None
        self._asr: tuple[Any, Any] | None = None
        self._lock = Lock()

    def prepare(self) -> None:
        """Load shared evaluation dependencies outside per-model measurements."""
        torch, _soundfile, model_types = self._dependencies()
        with self._lock:
            if self._cpu_threads:
                torch.set_num_threads(self._cpu_threads)
            self._load_vocoder(model_types)
            self._load_asr(model_types)

    def synthesize(
        self, definition: ExperimentModelDefinition, text: str, output_path: Path
    ) -> SpeechT5SynthesisOutput:
        torch, soundfile, model_types = self._dependencies()
        with self._lock:
            if self._cpu_threads:
                torch.set_num_threads(self._cpu_threads)
            torch.manual_seed(42)
            model_started = time.perf_counter()
            warm = definition.id in self._models
            processor, model = self._load_model(definition, model_types)
            vocoder = self._load_vocoder(model_types)
            load_ms = 0.0 if warm else (time.perf_counter() - model_started) * 1000
            speaker = self._speaker_tensor(torch)
            inputs = processor(text=text, return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                waveform = model.generate_speech(
                    inputs["input_ids"],
                    speaker,
                    vocoder=vocoder,
                    attention_mask=inputs.get("attention_mask"),
                )
            inference_ms = (time.perf_counter() - started) * 1000
            values = waveform.detach().float().cpu().numpy()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(f".{output_path.name}.tmp.wav")
            soundfile.write(temporary, values, 16_000, subtype="PCM_16")
            temporary.replace(output_path)
            duration_ms = len(values) / 16_000 * 1000
            return SpeechT5SynthesisOutput(
                audio_path=output_path,
                model_load_ms=load_ms,
                inference_ms=inference_ms,
                audio_duration_ms=duration_ms,
                real_time_factor=inference_ms / duration_ms if duration_ms else 0.0,
                process_memory_mb=psutil.Process().memory_info().rss / 1024**2,
                warm=warm,
            )

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        torch, soundfile, model_types = self._dependencies()
        with self._lock:
            processor, model = self._load_asr(model_types)
            audio, sample_rate = soundfile.read(audio_path, dtype="float32", always_2d=False)
            if sample_rate != 16_000 or audio.ndim != 1:
                raise ExperimentEvidenceError("Experiment audio must be mono 16 kHz WAV.")
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(inputs.input_features)
            transcript = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            return transcript, (time.perf_counter() - started) * 1000

    def _load_model(
        self, definition: ExperimentModelDefinition, model_types: dict[str, Any]
    ) -> tuple[Any, Any]:
        cached = self._models.pop(definition.id, None)
        if cached is not None:
            self._models[definition.id] = cached
            return cached
        if not definition.available:
            raise ExperimentEvidenceError(f"Model artifacts are unavailable for {definition.id}.")
        processor = model_types["SpeechT5Processor"].from_pretrained(
            definition.source, local_files_only=True
        )
        model = model_types["SpeechT5ForTextToSpeech"].from_pretrained(
            definition.source, local_files_only=True
        ).to("cpu").eval()
        self._models[definition.id] = (processor, model)
        while len(self._models) > self._maximum_cached_models:
            self._models.popitem(last=False)
            gc.collect()
        return processor, model

    def _load_vocoder(self, model_types: dict[str, Any]) -> Any:
        if self._vocoder is None:
            if not self._vocoder_root.is_dir():
                raise ExperimentEvidenceError("The pinned SpeechT5 vocoder is unavailable.")
            self._vocoder = model_types["SpeechT5HifiGan"].from_pretrained(
                self._vocoder_root, local_files_only=True
            ).to("cpu").eval()
        return self._vocoder

    def _load_asr(self, model_types: dict[str, Any]) -> tuple[Any, Any]:
        if self._asr is None:
            if not self._asr_root.is_dir():
                raise ExperimentEvidenceError("The pinned Whisper evaluator is unavailable.")
            processor = model_types["WhisperProcessor"].from_pretrained(
                self._asr_root, local_files_only=True
            )
            model = model_types["WhisperForConditionalGeneration"].from_pretrained(
                self._asr_root, local_files_only=True
            ).to("cpu").eval()
            self._asr = (processor, model)
        return self._asr

    def _speaker_tensor(self, torch: Any) -> Any:
        if not self._speaker_embedding_path.is_file():
            raise ExperimentEvidenceError("The verified Stage 12 speaker profile is unavailable.")
        values = np.load(self._speaker_embedding_path, allow_pickle=False).astype(np.float32)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ExperimentEvidenceError("The Stage 12 speaker embedding is invalid.")
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _dependencies() -> tuple[Any, Any, dict[str, Any]]:
        try:
            import soundfile
            import torch
            from transformers import (
                SpeechT5ForTextToSpeech,
                SpeechT5HifiGan,
                SpeechT5Processor,
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except ImportError as error:
            raise ExperimentEvidenceError(
                "SpeechT5 CPU dependencies are not installed. Install backend[experiment]."
            ) from error
        return torch, soundfile, {
            "SpeechT5ForTextToSpeech": SpeechT5ForTextToSpeech,
            "SpeechT5HifiGan": SpeechT5HifiGan,
            "SpeechT5Processor": SpeechT5Processor,
            "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
            "WhisperProcessor": WhisperProcessor,
        }
