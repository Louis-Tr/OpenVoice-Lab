"""Deterministic cleanup of text noise before speech synthesis."""

import re
import unicodedata

ISOLATED_CURRENT_PATH = re.compile(r"(?<!\S)\./(?!\S)")
REPEATED_HYPHENS = re.compile(r"-{2,}")
REPEATED_COMMAS = re.compile(r",{2,}")
ISOLATED_SLASH_RUN = re.compile(r"(?<!\S)/{2,}(?!\S)")
LONG_DOT_RUN = re.compile(r"\.{4,}")
ISOLATED_HYPHEN = re.compile(r"(?<!\S)-(?!\S)")
SPACE_BEFORE_PUNCTUATION = re.compile(r"[^\S\r\n]+([,.;:!?])(?=\s|$)")
HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")


class TextSanitizer:
    """Remove non-semantic noise while retaining useful English punctuation."""

    def sanitize(self, text: str) -> str:
        """Return one deterministic, whitespace-normalized synthesis string."""
        value = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
        value = "".join(self._clean_character(character) for character in value)
        value = ISOLATED_CURRENT_PATH.sub(" ", value)
        value = REPEATED_HYPHENS.sub(" ", value)
        value = REPEATED_COMMAS.sub(" ", value)
        value = ISOLATED_SLASH_RUN.sub(" ", value)
        value = LONG_DOT_RUN.sub("...", value)
        value = ISOLATED_HYPHEN.sub(" ", value)
        value = value.replace("$", "").replace("%", "")
        value = SPACE_BEFORE_PUNCTUATION.sub(r"\1", value)
        value = "\n".join(
            HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in value.split("\n")
        )
        return re.sub(r"\n{3,}", "\n\n", value).strip()

    @staticmethod
    def _clean_character(character: str) -> str:
        if character in {"\n", "\u2028"}:
            return "\n"
        if character == "\u2029":
            return "\n\n"
        if character.isspace():
            return " "
        if character == "\u200b" or unicodedata.category(character) == "Cc":
            # Controls/word-break hints must not fuse surrounding English tokens.
            return " "
        if unicodedata.category(character) == "Cf":
            return ""
        return character
