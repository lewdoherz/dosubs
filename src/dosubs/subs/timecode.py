"""Timecode formatting and parsing shared by exporters, validators and the editor."""

from __future__ import annotations

import re

__all__ = [
    "SRT_TIMECODE_RE",
    "format_srt",
    "format_short",
    "parse_timecode",
    "is_valid_srt_timecode",
    "format_duration",
]

#: Strict SRT wire format: comma separator. Used to validate what we write.
SRT_TIMECODE_RE = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d),(\d{3})$")
#: Lenient variant used when *reading*: other tools emit dotted milliseconds.
_LENIENT_TIMECODE_RE = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)[,.](\d{3})$")
#: ``MM:SS.mmm`` shorthand, exactly two colon-separated parts.
_SHORT_RE = re.compile(r"^([0-5]?\d):([0-5]\d)[.,](\d{1,3})$")


def format_srt(ms: int) -> str:
    """``1234`` -> ``00:00:01,234`` (the SRT wire format)."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_short(ms: int) -> str:
    """``1234`` -> ``00:01.234``; hours are only shown when present."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_duration(seconds: float) -> str:
    """``411.03`` -> ``6:51``; used in file lists."""
    total = int(round(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def parse_timecode(text: str) -> int:
    """Parse ``HH:MM:SS,mmm``, ``HH:MM:SS.mmm`` or ``MM:SS.mmm`` into milliseconds.

    Raises:
        ValueError: if the text is not a recognised timecode.
    """
    candidate = text.strip()
    match = _LENIENT_TIMECODE_RE.match(candidate)
    if match:
        hours, minutes, seconds, millis = (int(part) for part in match.groups())
        return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    match = _SHORT_RE.match(candidate)
    if match:
        minutes_text, seconds_text, millis_text = match.groups()
        millis = int(millis_text.ljust(3, "0"))
        return (int(minutes_text) * 60 + int(seconds_text)) * 1000 + millis
    raise ValueError(f"not a timecode: {text!r}")


def is_valid_srt_timecode(text: str) -> bool:
    return bool(SRT_TIMECODE_RE.match(text.strip()))
