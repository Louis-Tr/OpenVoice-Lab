"""Review-driven holdouts: mixed contexts, independent toggles and resource bounds."""

import itertools

import pytest

from app.text_processing.normalizer import TextNormalizationError, TextNormalizer
from app.text_processing.service import TextProcessingError, TextProcessingService


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**Visit https://example.com**.", "Visit example dot com."),
        ("[**API** guide](https://example.com/docs)", "API guide"),
        ("Use **$25** and **15%**.", "Use 25 dollars and 15 percent."),
        ("- ***important***", "important."),
        ("Read `D:\\My Folder\\main.py`.", "Read D drive slash My Folder slash main dot P Y."),
        ("Compute `a * b` without changing words.", "Compute a * b without changing words."),
        ("-5 is negative.\n- 5 is a list item.", "-5 is negative.\n5 is a list item."),
        ("Do not alter version v1.2.3 or 3.14.", "Do not alter version v1.2.3 or 3.14."),
        ("Read https://example.com/a%2Fb.", "Read example dot com slash a slash b."),
        ("Please use foo_bar@example.com.", "Please use foo underscore bar at example dot com."),
    ],
)
def test_review_holdout_outputs(source, expected):
    service = TextProcessingService()
    result = service.process(source, sanitize_text=True, normalize_text=True)
    assert result == expected
    assert service.process(result, sanitize_text=True, normalize_text=True) == result


@pytest.mark.parametrize(("sanitize", "normalize"), itertools.product((False, True), repeat=2))
@pytest.mark.parametrize(
    "source",
    [
        "  Keep ./ -- $25 exactly.\n\nNext paragraph.  ",
        "./ -- ,,, $ %",
        "First\u0000Second\u200bThird.",
        "NaturalSoft uses `model_id` and `user_name`.\n- Save ＄２５.",
        "**Read https://example.com/docs.**\nThen email dev_team@example.com.",
    ],
)
def test_repeated_processing_with_all_toggle_combinations(source, sanitize, normalize):
    service = TextProcessingService()
    if sanitize and source == "./ -- ,,, $ %":
        with pytest.raises(TextProcessingError, match="speakable"):
            service.process(source, sanitize_text=sanitize, normalize_text=normalize)
        return
    first = service.process(source, sanitize_text=sanitize, normalize_text=normalize)
    assert service.process(first, sanitize_text=sanitize, normalize_text=normalize) == first
    if not sanitize and not normalize:
        assert first == source


@pytest.mark.parametrize("source", ["https://example.com:no/", "https://[broken/"])
def test_invalid_url_only_parsed_when_normalizer_enabled(source):
    service = TextProcessingService()
    assert service.process(source, sanitize_text=False, normalize_text=False) == source
    assert service.process(source, sanitize_text=True, normalize_text=False) == source
    with pytest.raises(TextProcessingError, match="Invalid URL"):
        service.process(source, sanitize_text=False, normalize_text=True)


def test_expansion_and_intermediate_limits():
    with pytest.raises(TextProcessingError, match="exceeds 5000"):
        TextProcessingService().process("$1 " * 1000, sanitize_text=True, normalize_text=True)
    with pytest.raises(TextNormalizationError, match="intermediate"):
        TextNormalizer().normalize("＄" * 50_001)


def test_plain_text_at_output_limit_is_preserved():
    text = "a" * 5000
    assert TextProcessingService().process(text, sanitize_text=True, normalize_text=True) == text


@pytest.mark.parametrize("text", ["[" * 4999, "*" * 4999, "_" * 4999])
def test_long_unmatched_formatting_is_rejected_without_a_parser_crash(text):
    with pytest.raises(TextProcessingError, match="speakable"):
        TextProcessingService().process(text, sanitize_text=True, normalize_text=True)
