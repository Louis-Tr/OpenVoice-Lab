"""Focused tests for deterministic speakable-English normalization."""

import pytest

from app.text_processing.normalizer import TextNormalizer
from app.text_processing.service import TextProcessingError, TextProcessingService


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("Price: $1.", "Price: 1 dollar."),
        ("Price: $25.", "Price: 25 dollars."),
        ("Save 15% today.", "Save 15 percent today."),
        ("Email dev.team@example.com.", "Email dev dot team at example dot com."),
        ("Read https://example.com/docs.", "Read example dot com slash docs."),
        ("Load ./models/kokoro.onnx.", "Load models slash kokoro dot onnx."),
        ("**important** and *clear*", "important and clear"),
        ("Use [API guide](https://example.com/docs).", "Use API guide."),
        (
            "Save 15% at [https://example.com](https://example.com). Price: $25.",
            "Save 15 percent at example dot com. Price: 25 dollars.",
        ),
        ("Set `model_id`.", "Set model id."),
        ("x == 5", "x equals 5"),
        ("model_registry uses audioDuration", "model registry uses audio duration"),
    ),
)
def test_normalizer_converts_supported_notation(source: str, expected: str) -> None:
    normalizer = TextNormalizer()

    assert normalizer.normalize(source) == expected


def test_synthetic_sample_matches_the_speakable_contract() -> None:
    normalizer = TextNormalizer()
    source = (
        "Email dev.team@example.com -- open ./docs/api-guide.md.\n"
        "The price is $25, with a 15% discount."
    )

    assert normalizer.normalize(source) == (
        "Email dev dot team at example dot com—open docs slash api guide dot M D.\n"
        "The price is 25 dollars, with a 15 percent discount."
    )


def test_normal_sentences_remain_stable_and_normalization_is_idempotent() -> None:
    normalizer = TextNormalizer()
    source = "Well-written, natural speech asks useful questions."

    assert normalizer.normalize(source) == source
    normalized = normalizer.normalize(
        "Save 15% at https://example.com. Price: $25."
    )
    assert normalizer.normalize(normalized) == normalized


@pytest.mark.parametrize(
    ("sanitize_text", "normalize_text", "expected"),
    (
        (True, True, "Price: 25 dollars today"),
        (True, False, "Price: 25 today"),
        (False, True, "Price: 25 dollars ,,, today"),
        (False, False, "Price: $25 ,,, today"),
    ),
)
def test_processing_options_are_independent(
    sanitize_text: bool,
    normalize_text: bool,
    expected: str,
) -> None:
    service = TextProcessingService()

    assert service.process(
        "Price: $25 ,,, today",
        sanitize_text=sanitize_text,
        normalize_text=normalize_text,
    ) == expected


def test_normalization_runs_before_sanitization() -> None:
    service = TextProcessingService()

    assert service.process(
        "Save 15% and pay $1.",
        sanitize_text=True,
        normalize_text=True,
    ) == "Save 15 percent and pay 1 dollar."
    assert service.process(
        "Hello ./ --- world",
        sanitize_text=True,
        normalize_text=True,
    ) == "Hello world"


def test_processing_limit_applies_to_normalization_expansion() -> None:
    service = TextProcessingService(max_output_length=9)

    with pytest.raises(TextProcessingError, match="exceeds 9 characters"):
        service.process("$25", sanitize_text=False, normalize_text=True)
