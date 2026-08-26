"""Application service for optional synthesis text preprocessing."""

from app.text_processing.normalizer import TextNormalizer
from app.text_processing.sanitizer import TextSanitizer


class TextProcessingError(ValueError):
    """Raised when preprocessing cannot produce valid synthesis input."""


class TextProcessingService:
    """Apply configured text processing without knowing any inference runtime."""

    def __init__(
        self,
        sanitizer: TextSanitizer | None = None,
        normalizer: TextNormalizer | None = None,
        *,
        max_output_length: int = 5_000,
    ) -> None:
        self._sanitizer = sanitizer or TextSanitizer()
        self._normalizer = normalizer or TextNormalizer()
        self._max_output_length = max_output_length

    def process(
        self,
        text: str,
        *,
        sanitize_text: bool,
        normalize_text: bool,
    ) -> str:
        """Return the exact string that the inference engine should receive."""
        processed = self._normalizer.normalize(text) if normalize_text else text
        processed = self._sanitizer.sanitize(processed) if sanitize_text else processed

        if sanitize_text and not any(character.isalnum() for character in processed):
            raise TextProcessingError(
                "Sanitization removed all speakable content. "
                "Enter words or disable sanitization."
            )
        if len(processed) > self._max_output_length:
            raise TextProcessingError(
                f"Processed text exceeds {self._max_output_length} characters."
            )
        return processed
