"""Explicit English speech aliases, not model-specific pronunciation guesses."""

import re

LEXICON_VERSION = "1"
PRONUNCIATIONS = {
    "OpenVoice Lab": "Open Voice Lab",
    "OpenVoice": "Open Voice",
    "NaturalSoft": "Natural Soft",
    "SpeechT5": "Speech T five",
    "FastAPI": "Fast A P I",
    "CloudRun": "Cloud Run",
    "C#": "C sharp",
    "C++": "C plus plus",
}
_PATTERN = re.compile(
    r"(?<![\w+#])(?:"
    + "|".join(re.escape(name) for name in sorted(PRONUNCIATIONS, key=len, reverse=True))
    + r")(?![\w+#])"
)


def speak_known_names(text: str) -> str:
    """Longest exact name first; never match a substring inside a larger name."""
    return _PATTERN.sub(lambda match: PRONUNCIATIONS[match.group(0)], text)
