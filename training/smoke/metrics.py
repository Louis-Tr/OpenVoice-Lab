from __future__ import annotations

import re
import statistics
from typing import Any


def normal_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def contains_phrase(transcript: str, phrase: str) -> bool:
    words = normal_words(transcript)
    target = normal_words(phrase)
    if not target:
        return False
    size = len(target)
    return any(
        words[index : index + size] == target for index in range(len(words) - size + 1)
    )


def medical_term_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    total = 0
    details: list[dict[str, Any]] = []
    for row in rows:
        for annotation in row.get("medical_terms", []):
            term = annotation.get("canonical") or annotation.get("text")
            recognized = contains_phrase(row.get("transcript", ""), str(term))
            total += 1
            correct += int(recognized)
            details.append(
                {
                    "sample_id": row["sample_id"],
                    "term": term,
                    "category": annotation.get("category"),
                    "recognized": recognized,
                }
            )
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "status": "measured" if total else "not_available_no_annotated_terms",
        "details": details,
    }


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    reference_words = [word for value in references for word in normal_words(value)]
    hypothesis_words = [word for value in hypotheses for word in normal_words(value)]
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + int(reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    length_ratios = [
        len(normal_words(row.get("transcript", "")))
        / max(1, len(normal_words(row["reference"])))
        for row in successful
    ]
    short_outputs = [
        row for row in successful if len(normal_words(row.get("transcript", ""))) <= 2
    ]
    exact = [
        row
        for row in successful
        if normal_words(row.get("transcript", "")) == normal_words(row["reference"])
    ]
    return {
        "case_count": len(rows),
        "successful_cases": len(successful),
        "failure_count": len(rows) - len(successful),
        "wer": (
            word_error_rate(
                [row["reference"] for row in successful],
                [row["transcript"] for row in successful],
            )
            if successful
            else None
        ),
        "domain_term_accuracy": medical_term_score(successful),
        "average_inference_ms": (
            statistics.fmean(row["inference_ms"] for row in successful)
            if successful
            else None
        ),
        "average_rtf": (
            statistics.fmean(row["rtf"] for row in successful) if successful else None
        ),
        "peak_gpu_memory_mb": (
            max(row["peak_gpu_memory_mb"] for row in successful) if successful else None
        ),
        "average_audio_duration_seconds": (
            statistics.fmean(row["audio_duration_seconds"] for row in successful)
            if successful and all("audio_duration_seconds" in row for row in successful)
            else None
        ),
        "peak_process_memory_mb": (
            max(row["process_memory_mb"] for row in successful)
            if successful and all("process_memory_mb" in row for row in successful)
            else None
        ),
        "short_output_count": len(short_outputs),
        "short_output_rate": len(short_outputs) / len(successful) if successful else None,
        "median_transcript_reference_length_ratio": (
            statistics.median(length_ratios) if length_ratios else None
        ),
        "exact_sentence_count": len(exact),
        "exact_sentence_rate": len(exact) / len(successful) if successful else None,
    }
