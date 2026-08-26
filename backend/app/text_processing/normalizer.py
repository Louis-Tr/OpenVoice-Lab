"""Deterministic conversion of technical notation into speakable English."""

import re
from decimal import Decimal, InvalidOperation
from urllib.parse import unquote, urlsplit

MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
MARKDOWN_STRONG = re.compile(r"(?P<marker>\*\*|__)(?P<content>\S(?:.*?\S)?)\1")
MARKDOWN_EMPHASIS = re.compile(r"(?P<marker>[*_])(?P<content>\S(?:.*?\S)?)\1")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
EMAIL = re.compile(
    r"(?<![\w.+-])"
    r"(?P<local>[A-Za-z0-9][A-Za-z0-9._%+-]*)"
    r"@"
    r"(?P<domain>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+)"
)
BARE_URL = re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>()\[\]]+")
RELATIVE_PATH = re.compile(r"(?<![\w./])\./[^\s<>()\[\]]+")
CURRENCY = re.compile(r"(?<!\w)\$(?P<amount>\d(?:[\d,]*\d)?(?:\.\d+)?)")
PERCENTAGE = re.compile(r"(?P<amount>\d(?:[\d,]*\d)?(?:\.\d+)?)\s*%")
SPACED_DOUBLE_HYPHEN = re.compile(r"\s+--\s+")
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
TRAILING_SENTENCE_PUNCTUATION = ".,!?;:"
INITIALISM_EXTENSIONS = frozenset({"md"})

OPERATORS = (
    (re.compile(r"\s*!=\s*"), " not equals "),
    (re.compile(r"\s*>=\s*"), " greater than or equal to "),
    (re.compile(r"\s*<=\s*"), " less than or equal to "),
    (re.compile(r"\s*==\s*"), " equals "),
)


class TextNormalizer:
    """Convert supported notation without depending on an inference engine."""

    def normalize(self, text: str) -> str:
        """Return text with supported notation converted in a fixed rule order."""
        value = MARKDOWN_LINK.sub(r"\1", text)
        value = MARKDOWN_STRONG.sub(r"\g<content>", value)
        value = MARKDOWN_EMPHASIS.sub(r"\g<content>", value)
        value = INLINE_CODE.sub(r"\1", value)
        value = EMAIL.sub(self._speak_email, value)
        value = BARE_URL.sub(self._speak_url, value)
        value = RELATIVE_PATH.sub(self._speak_relative_path, value)
        value = CURRENCY.sub(self._speak_currency, value)
        value = PERCENTAGE.sub(r"\g<amount> percent", value)
        for pattern, replacement in OPERATORS:
            value = pattern.sub(replacement, value)
        value = SPACED_DOUBLE_HYPHEN.sub("—", value)
        return IDENTIFIER.sub(self._speak_identifier, value)

    @staticmethod
    def _speak_currency(match: re.Match[str]) -> str:
        amount = match.group("amount")
        try:
            singular = Decimal(amount.replace(",", "")) == Decimal(1)
        except InvalidOperation:
            singular = False
        unit = "dollar" if singular else "dollars"
        return f"{amount} {unit}"

    @classmethod
    def _speak_email(cls, match: re.Match[str]) -> str:
        local = cls._speak_email_part(match.group("local"), local=True)
        domain = cls._speak_email_part(match.group("domain"), local=False)
        return f"{local} at {domain}"

    @staticmethod
    def _speak_email_part(value: str, *, local: bool) -> str:
        replacements = {
            ".": " dot ",
            "-": " dash ",
            "_": " underscore ",
        }
        if local:
            replacements.update({"+": " plus ", "%": " percent "})
        for symbol, spoken in replacements.items():
            value = value.replace(symbol, spoken)
        return " ".join(value.split())

    @classmethod
    def _speak_url(cls, match: re.Match[str]) -> str:
        raw_url, punctuation = cls._split_trailing_punctuation(match.group(0))
        parseable_url = raw_url if "://" in raw_url else f"https://{raw_url}"
        parsed = urlsplit(parseable_url)

        host = parsed.hostname or parsed.netloc
        if host.lower().startswith("www."):
            host = host[4:]
        spoken = host.replace(".", " dot ")
        if parsed.port is not None:
            spoken = f"{spoken} port {parsed.port}"

        path = unquote(parsed.path).strip("/")
        if path:
            spoken = f"{spoken} slash {cls._speak_path(path)}"
        if parsed.query:
            spoken = f"{spoken} {cls._speak_query(parsed.query)}"
        if parsed.fragment:
            spoken = f"{spoken} fragment {cls._speak_path(unquote(parsed.fragment))}"

        return spoken + punctuation

    @classmethod
    def _speak_relative_path(cls, match: re.Match[str]) -> str:
        raw_path, punctuation = cls._split_trailing_punctuation(match.group(0))
        return cls._speak_path(raw_path.removeprefix("./")) + punctuation

    @classmethod
    def _speak_path(cls, path: str) -> str:
        return " slash ".join(
            cls._speak_path_segment(segment)
            for segment in path.split("/")
            if segment
        )

    @classmethod
    def _speak_path_segment(cls, segment: str) -> str:
        parts = segment.split(".")
        spoken = cls._split_identifier(parts[0].replace("-", " "))
        for extension in parts[1:]:
            extension_words = cls._split_identifier(extension.replace("-", " "))
            if extension.lower() in INITIALISM_EXTENSIONS:
                extension_words = " ".join(extension.upper())
            spoken = f"{spoken} dot {extension_words}"
        return spoken

    @classmethod
    def _speak_query(cls, query: str) -> str:
        pairs: list[str] = []
        for item in query.split("&"):
            key, separator, value = item.partition("=")
            spoken_key = cls._split_identifier(unquote(key))
            if separator:
                pairs.append(f"{spoken_key} equals {cls._split_identifier(unquote(value))}")
            else:
                pairs.append(spoken_key)
        return "question " + " and ".join(pairs)

    @classmethod
    def _speak_identifier(cls, match: re.Match[str]) -> str:
        value = match.group(0)
        if "_" not in value and not re.search(r"[a-z0-9][A-Z]", value):
            return value
        return cls._split_identifier(value)

    @staticmethod
    def _split_identifier(value: str) -> str:
        value = value.replace("_", " ")
        value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        words = value.split()
        if not words:
            return value
        return " ".join((words[0], *(word.lower() for word in words[1:])))

    @staticmethod
    def _split_trailing_punctuation(value: str) -> tuple[str, str]:
        punctuation = ""
        while value and value[-1] in TRAILING_SENTENCE_PUNCTUATION:
            punctuation = value[-1] + punctuation
            value = value[:-1]
        return value, punctuation
