"""Lossless, token-bounded text chunks for sequential speech generation."""

import re
from collections.abc import Callable

_SENTENCE_END = re.compile(r"[.!?][\"'\u2019\u201d)]*\s+")
_WORD_END = re.compile(r"\s+")


def split_text(text: str, token_count: Callable[[str], int], limit: int) -> list[str]:
    """Prefer sentence/word boundaries; count special tokens without truncation.

    Every returned chunk is checked with the actual tokenizer. Concatenating the
    chunks exactly reconstructs the input, including punctuation and whitespace.
    Overlong words fall back to character boundaries.
    """
    if limit < 1:
        raise ValueError("The text token limit must be positive.")
    chunks: list[str] = []
    remaining = text
    while remaining:
        if token_count(remaining) <= limit:
            chunks.append(remaining)
            break
        low, high, end = 1, len(remaining) - 1, 0
        while low <= high:
            middle = (low + high) // 2
            if token_count(remaining[:middle]) <= limit:
                end = middle
                low = middle + 1
            else:
                high = middle - 1
        if not end:
            raise ValueError("A single text character exceeds the model's token limit.")
        # A shorter prefix can tokenize differently, so validate boundaries too.
        for pattern in (_SENTENCE_END, _WORD_END):
            boundaries = list(pattern.finditer(remaining[:end]))
            boundary = next(
                (
                    match.end()
                    for match in reversed(boundaries)
                    if remaining[: match.end()].strip()
                    and token_count(remaining[: match.end()]) <= limit
                ),
                None,
            )
            if boundary is not None:
                end = boundary
                break
        chunks.append(remaining[:end])
        remaining = remaining[end:]
    return chunks
