from training.data_pipeline.text import (
    audit_model_text,
    integer_to_words,
    model_input_text,
    normalize_transcript,
)


def test_normalization_preserves_medical_notation_and_removes_controls() -> None:
    original = "  BP\u0000 120/80\u00a0mmHg — SpO2 97% at 37°C  "
    normalized = normalize_transcript(original)

    assert normalized == "BP 120/80 mmHg - SpO2 97% at 37°C"
    assert "120/80" in normalized
    assert "97%" in normalized
    assert "37°C" in normalized


def test_model_text_verbalizes_digits_and_measurements_deterministically() -> None:
    text = model_input_text("BP 120/80 mmHg, SpO2 97%, take 5 mg at 37°C.")

    assert (
        "blood pressure one hundred twenty over eighty millimeters of mercury" in text
    )
    assert "oxygen saturation ninety-seven percent" in text
    assert "five milligrams" in text
    assert "thirty-seven degrees celsius" in text
    assert audit_model_text(text)["safe"]


def test_embedded_medical_identifier_digits_are_verbalized_not_dropped() -> None:
    text = model_input_text("Check B12 after COVID-19")

    assert text == "check b twelve after covid-nineteen"
    assert audit_model_text(text)["safe"]


def test_number_words_cover_decimal_and_large_values() -> None:
    assert integer_to_words(0) == "zero"
    assert integer_to_words(120) == "one hundred twenty"
    assert integer_to_words(2005) == "two thousand five"
