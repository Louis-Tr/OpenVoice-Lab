"""Audio output service boundary."""

from dataclasses import dataclass

from app.inference.base import AudioBuffer


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    """Stored, encoded audio metadata."""

    url: str
    duration_seconds: float


class AudioService:
    """Encode, measure, and store generated audio outside inference adapters."""

    def create_artifact(self, _buffer: AudioBuffer) -> AudioArtifact:
        """Create an externally addressable audio artifact."""
        raise NotImplementedError("Audio handling is not implemented yet.")

