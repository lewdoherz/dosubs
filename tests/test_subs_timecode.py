"""Timecode formatting and parsing."""

from __future__ import annotations

import pytest

from dosubs.subs import timecode


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (0, "00:00:00,000"),
        (1, "00:00:00,001"),
        (1234, "00:00:01,234"),
        (59_999, "00:00:59,999"),
        (60_000, "00:01:00,000"),
        (3_600_000, "01:00:00,000"),
        (3_723_456, "01:02:03,456"),
        (-5, "00:00:00,000"),
    ],
)
def test_format_srt(ms: int, expected: str) -> None:
    assert timecode.format_srt(ms) == expected


@pytest.mark.parametrize(
    ("text", "expected_ms"),
    [
        ("00:00:00,000", 0),
        ("00:00:01,234", 1234),
        ("00:00:01.234", 1234),
        ("01:02:03,456", 3_723_456),
        ("02:03.456", 123_456),
        ("1:02:03,456", 3_723_456),
    ],
)
def test_parse_timecode_accepts_srt_and_short_forms(text: str, expected_ms: int) -> None:
    assert timecode.parse_timecode(text) == expected_ms


def test_parse_timecode_round_trips_format_srt() -> None:
    for ms in (0, 999, 1000, 61_000, 3_599_999, 7_200_500):
        assert timecode.parse_timecode(timecode.format_srt(ms)) == ms


@pytest.mark.parametrize("text", ["", "abc", "00:00:00", "00:00:00,12", "1:2:3,456", "00:00:00,1234"])
def test_parse_timecode_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        timecode.parse_timecode(text)


def test_is_valid_srt_timecode() -> None:
    assert timecode.is_valid_srt_timecode("00:00:01,234")
    assert not timecode.is_valid_srt_timecode("00:00:01.234")
    assert not timecode.is_valid_srt_timecode("1:00,000")


def test_format_short_and_duration() -> None:
    assert timecode.format_short(1234) == "00:01.234"
    assert timecode.format_short(3_723_456) == "1:02:03.456"
    assert timecode.format_duration(411.03) == "6:51"
    assert timecode.format_duration(3723.0) == "1:02:03"
    assert timecode.format_duration(-1) == "0:00"
