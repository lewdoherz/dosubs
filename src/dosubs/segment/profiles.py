"""Per-language segmentation budgets: how long a cue may last and how wide a line is.

The numbers are reading limits, not preferences. A latin cue of two 42-character
lines is about as much as a viewer can take in before the next cue replaces it;
a Chinese glyph carries roughly twice the information of a latin letter, so the
zh-CN profile halves both the line width and the reading speed. Splitting a cue
without those numbers produces the 39 s and 106 s monsters the recogniser hands
back, which is exactly what this package exists to prevent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = [
    "PROFILE_LATIN",
    "PROFILE_ZH_CN",
    "SegmentProfile",
    "char_count",
    "normalize_text",
    "profile_for",
    "required_duration_ms",
    "wrap_text",
]


@dataclass(frozen=True, slots=True)
class SegmentProfile:
    """Timing and width budget for one language family.

    Attributes:
        max_duration_ms: longest a single cue may stay on screen.
        min_duration_ms: shortest a cue should be, so it does not flash by.
        max_chars_per_line: characters a wrapped line may hold, spaces included.
        max_lines: lines a cue is expected to fill at most.
        max_chars_per_cue: readable characters (whitespace excluded) per cue.
        min_gap_ms: silence to keep between two consecutive cues.
        max_reading_speed_cps: readable characters per second of screen time.
        sentence_end_chars: preferred split points, strongest first.
        soft_break_chars: acceptable split points when no sentence ends.
        cjk: True when the script has no spaces between words.
    """

    name: str
    max_duration_ms: int
    min_duration_ms: int
    max_chars_per_line: int
    max_lines: int
    max_chars_per_cue: int
    min_gap_ms: int
    max_reading_speed_cps: float
    sentence_end_chars: str
    soft_break_chars: str
    cjk: bool


#: Latin (and other space-separated) scripts.
PROFILE_LATIN = SegmentProfile(
    "latin", 7000, 900, 42, 2, 84, 60, 21.0, ".!?…", ",;:—–-", False
)

#: Simplified Chinese, where punctuation is full width and glyphs are dense.
PROFILE_ZH_CN = SegmentProfile(
    "zh-CN", 7000, 900, 18, 2, 36, 60, 11.0, "。！？!?…；;", "，、：；—…：", True
)

#: Codepoint blocks that read as CJK: no inter-word spaces, one glyph per character.
_CJK_BLOCKS = (
    (0x2E80, 0x2EFF),  # radicals supplement
    (0x3000, 0x303F),  # CJK symbols and punctuation (。、「」)
    (0x3040, 0x30FF),  # hiragana and katakana
    (0x3400, 0x4DBF),  # unified ideographs extension A
    (0x4E00, 0x9FFF),  # unified ideographs
    (0xF900, 0xFAFF),  # compatibility ideographs
    (0xFF00, 0xFFEF),  # full-width forms (，！？)
)

#: Whisper/VAD happily emit "你 好" for "你好"; those spaces are noise, not pauses.
_SPACE_BETWEEN_CJK_RE = re.compile(
    "(?<=[{0}])\\s+(?=[{0}])".format(
        "".join(f"\\u{low:04x}-\\u{high:04x}" for low, high in _CJK_BLOCKS)
    )
)


def profile_for(language: str | None) -> SegmentProfile:
    """Pick the budget for a BCP-47-ish tag; unknown or missing tags read as latin.

    Only the primary subtag matters: ``zh``, ``zh-CN`` and ``zh-Hans`` are all
    Simplified/Chinese text as far as cue length is concerned.
    """
    primary = (language or "").strip().lower().split("-", 1)[0]
    return PROFILE_ZH_CN if primary == "zh" else PROFILE_LATIN


def normalize_text(text: str, profile: SegmentProfile) -> str:
    """Collapse whitespace runs and strip; drop spaces squeezed between CJK glyphs."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if profile.cjk and cleaned:
        cleaned = _SPACE_BETWEEN_CJK_RE.sub("", cleaned)
    return cleaned


def char_count(text: str) -> int:
    """Characters a viewer actually reads: whitespace is free, every glyph counts once.

    CJK codepoints count as one, not as the two columns they occupy on screen.
    """
    return sum(1 for char in text if not char.isspace())


def required_duration_ms(text: str, profile: SegmentProfile) -> int:
    """Milliseconds a viewer needs to read ``text`` at the profile's reading speed."""
    return math.ceil(char_count(text) * 1000.0 / profile.max_reading_speed_cps)


def wrap_text(text: str, profile: SegmentProfile) -> list[str]:
    """Lay ``text`` out as display lines, each within ``max_chars_per_line``.

    ``max_lines`` is the budget the cue splitter works towards; when the text is
    longer than that budget allows, extra lines are emitted rather than characters
    being dropped, because a lost word is worse than a third line on screen.
    """
    cleaned = normalize_text(text, profile)
    if not cleaned:
        return []
    if profile.cjk:
        return _wrap_cjk(cleaned, profile)
    return _wrap_latin(cleaned, profile)


#: Characters that may not begin a line in Chinese typesetting.
_NO_LINE_START = "，。、；：！？）］｝」』】〕〉》”’…—～%"
#: Characters that may not end a line (opening brackets and quotes).
_NO_LINE_END = "（［｛「『【〔〈《“‘"

#: Runs of latin letters/digits stay together even in a CJK line, so mixed text such
#: as "OK" or "Wi-Fi" is never cut in half.
_LATIN_RUN_RE = re.compile(r"[A-Za-z0-9]+(?:[.'\u2019\-][A-Za-z0-9]+)*")


def _cjk_units(text: str) -> Iterator[str]:
    """Atomic wrap units: one per CJK glyph, one per latin/digit run."""
    index = 0
    while index < len(text):
        match = _LATIN_RUN_RE.match(text, index)
        if match:
            yield match.group(0)
            index = match.end()
        else:
            yield text[index]
            index += 1


def _wrap_cjk(text: str, profile: SegmentProfile) -> list[str]:
    """Chinese wraps between glyphs, with word and punctuation rules applied.

    Applying a fixed-width slice would cut latin words in half and could leave a
    line starting with ``。`` or ending with ``（``, both of which look broken to a
    Chinese reader even when the character count is correct.
    """
    width = max(1, profile.max_chars_per_line)
    lines: list[str] = []
    current = ""

    for unit in _cjk_units(text):
        if len(unit) > width:
            # A single over-long word (a URL, a long name) is hard-broken in place.
            if current:
                lines.append(current)
                current = ""
            while len(unit) > width:
                lines.append(unit[:width])
                unit = unit[width:]
            current = unit
            continue

        if len(current) + len(unit) <= width:
            current += unit
            continue

        # Breaking here: push a trailing opening bracket down, or keep closing
        # punctuation with the line it belongs to instead of starting a line with it.
        if unit in _NO_LINE_START and len(current) > 1:
            carry = current[-1]
            lines.append(current[:-1])
            current = carry + unit
            continue
        if current and current[-1] in _NO_LINE_END and len(current) > 1:
            carry = current[-1]
            current = current[:-1]
            lines.append(current)
            current = carry + unit
            continue

        lines.append(current)
        current = unit

    if current:
        lines.append(current)
    return [line for line in lines if line]


def _wrap_latin(text: str, profile: SegmentProfile) -> list[str]:
    """Greedy word wrap; the separating space counts against the line width."""
    width = max(1, profile.max_chars_per_line)
    lines: list[str] = []
    current = ""
    for token in _tokens(text, width):
        if not current:
            current = token
        elif len(current) + 1 + len(token) <= width:
            current = f"{current} {token}"
        else:
            lines.append(current)
            current = token
    if current:
        lines.append(current)
    return lines


def _tokens(text: str, width: int) -> Iterator[str]:
    """Words of ``text``; a word wider than the line is hard-broken so no line overflows."""
    for word in text.split():
        if len(word) <= width:
            yield word
        else:
            for index in range(0, len(word), width):
                yield word[index : index + width]
