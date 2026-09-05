"""Deterministic conversion of technical notation into speakable English."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from urllib.parse import unquote, urlsplit

from app.text_processing.lexicon import speak_known_names
from app.text_processing.structure import normalize_document

MARKDOWN_LINK = re.compile(r"\[([^\[\]\n]+)\]\(([^)\n]+)\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
EMAIL = re.compile(
    r"(?<![\w.+-])"
    r"(?P<local>[A-Za-z0-9][A-Za-z0-9._%+-]*)"
    r"@"
    r"(?P<domain>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+)"
)
BARE_URL = re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>()`]+")
FILE_PATH = re.compile(
    r"(?<![\w./\\:])(?:[A-Za-z]:[\\/][^\s<>()\[\]`\"']+"
    r"|(?:\.\.?/|/)[^\s<>()\[\]`\"']+"
    r"|[A-Za-z_][\w.-]*(?:/[\w.-]+)+)"
)
CODE_FILENAME = re.compile(r"[A-Za-z_][\w.-]*\.[A-Za-z][A-Za-z0-9]{0,9}")
CURRENCY = re.compile(r"(?<!\w)\$(?P<amount>\d(?:[\d,]*\d)?(?:\.\d+)?)")
PERCENTAGE = re.compile(r"(?P<amount>\d(?:[\d,]*\d)?(?:\.\d+)?)\s*%")
SPACED_DOUBLE_HYPHEN = re.compile(r"\s+--\s+")
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
TRAILING_SENTENCE_PUNCTUATION = ".,!?;:"
INITIALISM_EXTENSIONS = frozenset({"md", "py", "ts", "js", "css", "html", "yml", "yaml"})
MAX_INTERMEDIATE_CHARACTERS = 50_000
PROTECTED_TOKEN = re.compile(
    r"(?P<strong>(?<![\w*])(?:\*\*\S[^\n]*?\*\*|__\S[^\n]*?__)(?![\w*]))"
    r"|(?P<emphasis>(?<![\w*])(?:\*(?![\s*])[^*\n]+(?<!\s)\*"
    r"|_(?![\s_])[^_\n]+(?<!\s)_)(?![\w*]))"
    f"|(?P<link>{MARKDOWN_LINK.pattern})|(?P<code>{INLINE_CODE.pattern})"
    f"|(?P<url>{BARE_URL.pattern})|(?P<email>{EMAIL.pattern})"
    f"|(?P<path>{FILE_PATH.pattern})"
)

OPERATORS = (
    (re.compile(r"\s*!=\s*"), " not equals "),
    (re.compile(r"\s*>=\s*"), " greater than or equal to "),
    (re.compile(r"\s*<=\s*"), " less than or equal to "),
    (re.compile(r"\s*==\s*"), " equals "),
)


class TextNormalizationError(ValueError):
    """A supported notation cannot be safely interpreted."""


def bounded(value: str) -> str:
    if len(value) > MAX_INTERMEDIATE_CHARACTERS:
        raise TextNormalizationError(
            f"Text normalization exceeds {MAX_INTERMEDIATE_CHARACTERS} intermediate characters."
        )
    return value


class TextNormalizer:
    """Convert supported notation without depending on an inference engine."""

    def normalize(self, text: str) -> str:
        """Return text with supported notation converted in a fixed rule order."""
        value = bounded(unicodedata.normalize("NFKC", bounded(text)))
        return bounded(normalize_document(value, self._normalize_spans, self._normalize_code))

    def _normalize_spans(self, text: str, depth: int = 0) -> str:
        """Recognize addresses/code before prose rules; never reparse generated spans."""
        if depth > 12:
            raise TextNormalizationError("Text formatting nesting exceeds 12 levels.")
        pieces: list[str] = []
        cursor = 0
        for match in PROTECTED_TOKEN.finditer(text):
            pieces.append(self._normalize_plain(text[cursor : match.start()]))
            raw = match.group(0)
            if match.group("strong") is not None:
                pieces.append(self._normalize_spans(raw[2:-2], depth + 1))
            elif match.group("emphasis") is not None:
                pieces.append(self._normalize_spans(raw[1:-1], depth + 1))
            elif match.group("link") is not None:
                link = MARKDOWN_LINK.fullmatch(raw)
                pieces.append(self._normalize_spans(link.group(1), depth + 1))
            elif match.group("code") is not None:
                pieces.append(self._normalize_code(raw[1:-1]))
            elif match.group("url") is not None:
                pieces.append(self._speak_url(BARE_URL.fullmatch(raw)))
            elif match.group("email") is not None:
                pieces.append(self._speak_email(EMAIL.fullmatch(raw)))
            else:
                path, punctuation = self._split_trailing_punctuation(raw)
                if self._is_path(path):
                    pieces.append(self._speak_file_path(path) + punctuation)
                else:
                    pieces.append(self._normalize_plain(raw))
            cursor = match.end()
        pieces.append(self._normalize_plain(text[cursor:]))
        return bounded("".join(pieces))

    def _normalize_code(self, text: str) -> str:
        # Code context makes a bare filename or two-component path unambiguous.
        if (
            CODE_FILENAME.fullmatch(text)
            or (FILE_PATH.fullmatch(text) and re.search(r"[A-Za-z]", text))
            or re.match(r"^[A-Za-z]:[\\/]", text)
        ):
            return self._speak_file_path(text)
        return self._normalize_spans(text)

    @staticmethod
    def _is_path(value: str) -> bool:
        if not any(character.isalnum() for character in value):
            return False
        if value.startswith(("./", "../", "/")) or re.match(r"^[A-Za-z]:[\\/]", value):
            return True
        return (
            value.count("/") >= 2 or CODE_FILENAME.fullmatch(value.rsplit("/", 1)[-1]) is not None
        )

    @classmethod
    def _speak_file_path(cls, value: str) -> str:
        path = value.replace("\\", "/")
        prefix = ""
        if re.match(r"^[A-Za-z]:/", path):
            prefix, path = f"{path[0].upper()} drive slash ", path[3:]
        elif path.startswith("/"):
            prefix, path = "root slash ", path.lstrip("/")
        while path.startswith("../"):
            prefix += "parent slash "
            path = path[3:]
        path = path.removeprefix("./")
        return prefix + cls._speak_path(path)

    @classmethod
    def _normalize_plain(cls, value: str) -> str:
        value = bounded(CURRENCY.sub(cls._speak_currency, value))
        value = bounded(PERCENTAGE.sub(r"\g<amount> percent", value))
        for pattern, replacement in OPERATORS:
            value = bounded(pattern.sub(replacement, value))
        # Do not turn isolated './ --' noise into a new apparent relative path
        # ('./—...') that a second pass would parse differently.
        before_dash = value
        value = SPACED_DOUBLE_HYPHEN.sub(
            lambda match: (
                match.group(0)
                if before_dash[: match.start()].rstrip().endswith(("/", "\\"))
                else "—"
            ),
            value,
        )
        value = bounded(speak_known_names(value))
        return bounded(IDENTIFIER.sub(cls._speak_identifier, value))

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
        # Spoken addresses retain letters/digits/separators, not typographic case.
        # Canonical casing also prevents a second pass from treating address parts
        # as camelCase identifiers or brand names.
        value = value.lower()
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
        try:
            parsed = urlsplit(parseable_url)
            host = parsed.hostname
            port = parsed.port
            if not host:
                raise ValueError("missing host")
        except ValueError as error:
            raise TextNormalizationError(
                "Invalid URL in synthesis text: malformed host or port."
            ) from error
        if host.lower().startswith("www."):
            host = host[4:]
        spoken = host.replace(".", " dot ")
        if port is not None:
            spoken = f"{spoken} port {port}"

        path = unquote(parsed.path).strip("/")
        if path:
            spoken = f"{spoken} slash {cls._speak_path(path)}"
        if parsed.query:
            spoken = f"{spoken} {cls._speak_query(parsed.query)}"
        if parsed.fragment:
            spoken = f"{spoken} fragment {cls._speak_path(unquote(parsed.fragment))}"

        return spoken + punctuation

    @classmethod
    def _speak_path(cls, path: str) -> str:
        return " slash ".join(
            cls._speak_path_segment(segment) for segment in path.split("/") if segment
        )

    @classmethod
    def _speak_path_segment(cls, segment: str) -> str:
        parts = segment.split(".")
        spoken = cls._normalize_plain(parts[0].replace("-", " "))
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
        if "_" not in value and (not value[0].islower() or not re.search(r"[a-z0-9][A-Z]", value)):
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
