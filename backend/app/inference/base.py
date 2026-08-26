"""Replaceable text-to-speech inference contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class InferenceError(RuntimeError):
    """Base error for a failed local inference operation."""


class UnsupportedVoiceError(InferenceError):
    """Raised when an engine does not expose the requested voice."""


@dataclass(frozen=True, slots=True)
class AudioResult:
    """Raw mono audio returned by an inference engine."""

    samples: NDArray[np.float32]
    sample_rate_hz: int

    @property
    def duration_seconds(self) -> float:
        """Return the exact duration represented by the sample buffer."""
        return len(self.samples) / self.sample_rate_hz


class TTSInferenceEngine(ABC):
    """Replaceable boundary for locally hosted TTS inference."""

    @property
    @abstractmethod
    def voices(self) -> tuple[str, ...]:
        """Return voices supported by the loaded engine."""
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> AudioResult:
        """Generate raw audio without HTTP or storage concerns."""
        raise NotImplementedError


# Backward-compatible architecture name retained for existing documentation.
TTSInferenceAdapter = TTSInferenceEngine
