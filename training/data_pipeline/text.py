from __future__ import annotations

import re
import unicodedata
from typing import Any

from .config import PipelineConfig


_CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}
_WHITESPACE = re.compile(r"\s+")
_SAFE_MODEL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz'.,!?- ")

_SMALL = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def integer_to_words(value: int) -> str:
    if value < 0:
        return "minus " + integer_to_words(-value)
    if value < 20:
        return _SMALL[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return _TENS[tens] + (f"-{_SMALL[remainder]}" if remainder else "")
    if value < 1_000:
        hundreds, remainder = divmod(value, 100)
        return (
            _SMALL[hundreds]
            + " hundred"
            + (f" {integer_to_words(remainder)}" if remainder else "")
        )
    for scale, label in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ):
        if value >= scale:
            quotient, remainder = divmod(value, scale)
            return (
                integer_to_words(quotient)
                + f" {label}"
                + (f" {integer_to_words(remainder)}" if remainder else "")
            )
    return " ".join(_SMALL[int(digit)] for digit in str(value))


def number_to_words(value: str) -> str:
    value = value.strip()
    sign = ""
    if value.startswith("-"):
        sign, value = "minus ", value[1:]
    if "." in value:
        whole, fractional = value.split(".", 1)
        return (
            sign
            + integer_to_words(int(whole or "0"))
            + " point "
            + " ".join(_SMALL[int(digit)] for digit in fractional)
        )
    return sign + integer_to_words(int(value))


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    translations = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u00a0": " ",
        }
    )
    normalized = normalized.translate(translations)
    normalized = "".join(
        " " if unicodedata.category(character) in _CONTROL_CATEGORIES else character
        for character in normalized
    )
    return _WHITESPACE.sub(" ", normalized).strip()


def _verbalize_measurements(text: str) -> str:
    text = re.sub(
        r"(?<!\d)(-?\d{2,3})\s*/\s*(\d{2,3})(?:\s*mm\s*hg|\s*mmhg)?(?!\w)",
        lambda match: (
            f"{number_to_words(match.group(1))} over {number_to_words(match.group(2))}"
            + (" millimeters of mercury" if "mm" in match.group(0).casefold() else "")
        ),
        text,
        flags=re.IGNORECASE,
    )
    units = {
        "mcg": "micrograms",
        "mg": "milligrams",
        "kg": "kilograms",
        "ml": "milliliters",
        "mmhg": "millimeters of mercury",
        "bpm": "beats per minute",
        "cm": "centimeters",
        "mm": "millimeters",
        "g": "grams",
        "l": "liters",
    }
    unit_pattern = "|".join(sorted(units, key=len, reverse=True))
    text = re.sub(
        rf"(?<!\w)(-?\d+(?:\.\d+)?)\s*({unit_pattern})(?!\w)",
        lambda match: (
            f"{number_to_words(match.group(1))} {units[match.group(2).casefold()]}"
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)(-?\d+(?:\.\d+)?)\s*°?\s*([CF])(?!\w)",
        lambda match: (
            f"{number_to_words(match.group(1))} degrees "
            + ("Celsius" if match.group(2).casefold() == "c" else "Fahrenheit")
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)(-?\d+(?:\.\d+)?)\s*%",
        lambda match: f"{number_to_words(match.group(1))} percent",
        text,
    )
    return text


def model_input_text(normalized: str) -> str:
    text = normalized
    abbreviations = {
        r"\bSpO2\b": "oxygen saturation",
        r"\bBP\b": "blood pressure",
        r"\bECG\b": "E C G",
        r"\bEKG\b": "E K G",
        r"\bMRI\b": "M R I",
        r"\bCT\b": "C T",
        r"\bIV\b": "I V",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _verbalize_measurements(text)
    text = re.sub(
        r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)",
        lambda match: number_to_words(match.group(0)),
        text,
    )
    # Verbalize digits embedded in medical identifiers (for example B12 or
    # COVID-19) instead of deleting them as unsupported tokenizer input.
    text = re.sub(r"\d+", lambda match: f" {number_to_words(match.group(0))} ", text)
    text = text.replace("&", " and ").replace("+", " plus ").replace("/", " slash ")
    text = (
        text.replace("=", " equals ")
        .replace("<", " less than ")
        .replace(">", " greater than ")
    )
    text = text.casefold()
    text = "".join(
        character if character in _SAFE_MODEL_CHARS else " " for character in text
    )
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def process_transcripts(
    records: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    del config
    for record in records:
        original = record.get("original_transcript", "")
        normalized = normalize_transcript(original) if isinstance(original, str) else ""
        model_text = model_input_text(normalized)
        record["normalized_transcript"] = normalized
        record["model_input_text"] = model_text
        if normalized and not model_text:
            if "MODEL_INPUT_TEXT_EMPTY" not in record["exclusion_reasons"]:
                record["exclusion_reasons"].append("MODEL_INPUT_TEXT_EMPTY")
        record["text_audit"] = {
            "normalization_changed": original != normalized,
            "model_input_changed": normalized != model_text,
            "unsupported_model_characters": sorted(set(model_text) - _SAFE_MODEL_CHARS),
            "contains_digit_after_verbalization": bool(re.search(r"\d", model_text)),
        }
        record["exclusion_reasons"].sort()
    return records


def audit_model_text(text: str) -> dict[str, Any]:
    unsupported = sorted(set(text) - _SAFE_MODEL_CHARS)
    return {
        "unsupported_characters": unsupported,
        "contains_digit": bool(re.search(r"\d", text)),
        "safe": not unsupported and not re.search(r"\d", text),
    }
