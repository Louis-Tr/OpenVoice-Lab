"""Deterministic ASR proxy metrics for live SpeechT5 comparisons."""

import re
from collections.abc import Sequence

from app.schemas.experiment import ExperimentTermScore

WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(value: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(value)]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Levenshtein word error rate without external packages."""
    expected = tokenize(reference)
    observed = tokenize(hypothesis)
    if not expected:
        return 0.0 if not observed else float(len(observed))
    previous = list(range(len(observed) + 1))
    for reference_index, reference_word in enumerate(expected, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(observed, start=1):
            substitution = previous[hypothesis_index - 1] + (
                reference_word != hypothesis_word
            )
            deletion = previous[hypothesis_index] + 1
            insertion = current[hypothesis_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1] / len(expected)


def score_terms(target_terms: Sequence[str], transcript: str) -> ExperimentTermScore:
    """Match complete normalized word sequences, preserving requested order."""
    transcript_tokens = tokenize(transcript)
    correct: list[str] = []
    incorrect: list[str] = []
    for term in target_terms:
        term_tokens = tokenize(term)
        matched = bool(term_tokens) and any(
            transcript_tokens[index : index + len(term_tokens)] == term_tokens
            for index in range(len(transcript_tokens) - len(term_tokens) + 1)
        )
        (correct if matched else incorrect).append(term)
    total = len(target_terms)
    return ExperimentTermScore(
        correct=correct,
        incorrect=incorrect,
        correct_count=len(correct),
        total_count=total,
        accuracy=len(correct) / total if total else 0.0,
    )
