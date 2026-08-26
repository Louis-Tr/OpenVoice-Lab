"""Focused tests for deterministic synthesis text sanitization."""

import pytest

from app.text_processing.sanitizer import TextSanitizer
from app.text_processing.service import TextProcessingError, TextProcessingService


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("Hello ./ --- world", "Hello world"),
        ("Price: $25 ,,, today", "Price: 25 today"),
        ("Well-written, natural speech.", "Well-written, natural speech."),
        ("Wait.... what?", "Wait... what?"),
        ("Keep https://example.com /// docs", "Keep https://example.com docs"),
    ),
)
def test_sanitizer_removes_noise_without_damaging_useful_punctuation(
    source: str,
    expected: str,
) -> None:
    sanitizer = TextSanitizer()

    assert sanitizer.sanitize(source) == expected
    assert sanitizer.sanitize(expected) == expected


def test_sanitizer_normalizes_unicode_controls_and_whitespace() -> None:
    sanitizer = TextSanitizer()

    assert sanitizer.sanitize("Ａ clean\u200b\tline\nnow\x00") == "A clean line now"


def test_processing_rejects_input_without_speakable_content() -> None:
    service = TextProcessingService()

    with pytest.raises(TextProcessingError, match="removed all speakable content"):
        service.process("./ -- ,,, $ %", sanitize_text=True, normalize_text=True)


def test_processing_can_be_disabled_without_changing_the_input() -> None:
    service = TextProcessingService()
    source = "  Keep ./ -- $25 exactly.  "

    assert service.process(source, sanitize_text=False, normalize_text=False) == source


def test_processing_enforces_its_output_limit() -> None:
    service = TextProcessingService(max_output_length=10)

    with pytest.raises(TextProcessingError, match="exceeds 10 characters"):
        service.process("eleven chars", sanitize_text=True, normalize_text=True)
