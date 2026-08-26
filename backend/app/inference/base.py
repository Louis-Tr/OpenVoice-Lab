"""Abstract text-to-speech inference contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.synthesis import ModelVariant


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Backend-internal input shared by inference adapters."""

    text: str
    model_id: str
    voice_id: str
    variant: ModelVariant


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Unencoded audio returned by an inference adapter."""

    samples: bytes
    sample_rate_hz: int
    channels: int = 1


class TTSInferenceAdapter(ABC):
    """Replaceable boundary for a concrete TTS inference implementation."""

    @abstractmethod
    async def synthesize(self, request: InferenceRequest) -> AudioBuffer:
        """Generate an audio buffer for a technology-neutral request."""
        raise NotImplementedError

