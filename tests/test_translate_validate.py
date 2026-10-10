"""Window validation and timeline-protection tests (pure, no I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dosubs.models import Cue, MediaInfo, Transcript
from dosubs.translate import (
    ValidationReport,
    assert_valid_window,
    validate_window,
    verify_timeline_unchanged,
)
from dosubs.translate.base import TranslationError, TranslationFormatError

MEDIA = MediaInfo(path=Path("video.mp4"), duration_s=4.0, size_bytes=2048, has_audio=True)


def make_transcript(*rows: tuple[int, int, int, str], text_translated: str = "") -> Transcript:
    """rows are ``(id, start_ms, end_ms, text)``."""
    return Transcript(
        media=MEDIA,
        cues=[
            Cue(
                id=cue_id,
                start_ms=start,
                end_ms=end,
                text_original=text,
                text_translated=text_translated,
            )
            for cue_id, start, end, text in rows
        ],
    )


# --------------------------------------------------------------------------- #
# validate_window
# --------------------------------------------------------------------------- #


def test_happy_path_is_ok():
    requested = [(1, "Hello."), (2, "How are you?")]
    report = validate_window(requested, {1: "你好。", 2: "你好吗？"})
    assert report.ok
    assert isinstance(report, ValidationReport)
    assert report.missing == []
    assert report.empty == []
    assert report.duplicated == {}
    assert report.unexpected == []


def test_missing_id_is_reported():
    report = validate_window([(1, "a"), (2, "b"), (3, "c")], {1: "甲", 3: "丙"})
    assert report.missing == [2]
    assert not report.ok


def test_unexpected_id_is_reported():
    report = validate_window([(1, "a")], {1: "甲", 7: "extra"})
    assert report.unexpected == [7]
    assert not report.ok


def test_empty_translation_is_reported():
    report = validate_window([(1, "a"), (2, "b")], {1: "   ", 2: "乙"})
    assert report.empty == [1]
    assert report.duplicated == {}
    assert not report.ok


def test_duplicated_text_is_reported():
    report = validate_window(
        [(1, "one"), (2, "two"), (3, "three")],
        {1: "same", 2: "same", 3: "other"},
    )
    assert report.duplicated == {"same": [1, 2]}
    assert not report.ok


def test_blank_translations_are_not_duplicates():
    report = validate_window([(1, "a"), (2, "b")], {1: "", 2: ""})
    assert report.empty == [1, 2]
    assert report.duplicated == {}


def test_assert_valid_window_raises_with_detail():
    with pytest.raises(TranslationFormatError) as excinfo:
        assert_valid_window([(1, "a"), (2, "b")], {1: "甲"})
    message = str(excinfo.value)
    assert "missing ids [2]" in message
    assert isinstance(excinfo.value, TranslationError)


def test_assert_valid_window_passes_silently():
    assert_valid_window([(1, "a")], {1: "甲"})


# --------------------------------------------------------------------------- #
# verify_timeline_unchanged
# --------------------------------------------------------------------------- #


def test_timeline_unchanged_on_identical_transcripts():
    before = make_transcript((1, 0, 1000, "a"), (2, 1000, 2000, "b"))
    after = make_transcript((1, 0, 1000, "a"), (2, 1000, 2000, "b"), text_translated="x")
    verify_timeline_unchanged(before, after)


def test_timeline_start_change_raises():
    before = make_transcript((1, 0, 1000, "a"))
    after = make_transcript((1, 250, 1000, "a"))
    with pytest.raises(TranslationError, match="moved"):
        verify_timeline_unchanged(before, after)


def test_timeline_end_change_raises():
    before = make_transcript((1, 0, 1000, "a"))
    after = make_transcript((1, 0, 1200, "a"))
    with pytest.raises(TranslationError):
        verify_timeline_unchanged(before, after)


def test_timeline_count_change_raises():
    before = make_transcript((1, 0, 1000, "a"), (2, 1000, 2000, "b"))
    after = make_transcript((1, 0, 1000, "a"))
    with pytest.raises(TranslationError, match="cue count changed"):
        verify_timeline_unchanged(before, after)


def test_timeline_id_renumbering_raises():
    before = make_transcript((1, 0, 1000, "a"), (2, 1000, 2000, "b"))
    after = make_transcript((1, 0, 1000, "a"), (5, 1000, 2000, "b"))
    with pytest.raises(TranslationError, match="changed id"):
        verify_timeline_unchanged(before, after)
