"""A bounded Markdown subset rendered as speech text, never HTML or executable code."""

import re
from collections.abc import Callable

HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
QUOTE = re.compile(r"^ {0,3}(?:>[ \t]?)+(?P<text>.*)$")
LIST_ITEM = re.compile(r"^[ \t]*(?:[-+*]|\d{1,9}[.)])[ \t]+(?P<text>\S.*)$")
INLINE_LIST_MARKER = re.compile(r"(?<=[,:;])[ \t]+\*[ \t]+")
FENCE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})[ \t]*[\w+-]*[ \t]*$")
RULE = re.compile(r"^ {0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*$")
TERMINAL = re.compile(r"[.!?…][\"'”’)]*$")


def finish_item(text: str) -> str:
    text = text.strip()
    if not text or TERMINAL.search(text):
        return text
    return text.rstrip(",;:") + "."


def normalize_document(
    text: str,
    inline: Callable[[str], str],
    code: Callable[[str], str],
) -> str:
    """Retain line/paragraph boundaries; remove only recognized structural markers."""
    output: list[str] = []
    fence: str | None = None
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if fence is not None:
            if re.fullmatch(rf"[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", line):
                fence = None
            else:
                output.append(code(line))
            continue
        opening = FENCE.fullmatch(line)
        if opening:
            fence = opening.group("marker")
            continue
        if RULE.fullmatch(line):
            output.append("")
            continue
        quote = QUOTE.fullmatch(line)
        if quote:
            line = quote.group("text")
        heading = HEADING.fullmatch(line)
        item = LIST_ITEM.fullmatch(line)
        if heading or item:
            content = heading.group(1) if heading else item.group("text")
            output.append(finish_item(inline(content)))
        elif len(INLINE_LIST_MARKER.findall(line)) >= 2:
            lead, *items = INLINE_LIST_MARKER.split(line)
            output.append(inline(lead))
            output.extend(finish_item(inline(item)) for item in items)
        else:
            output.append(inline(line))
    return "\n".join(output)
