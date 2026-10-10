"""Subtitle rendering, writing and post-write verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from dosubs.models import Cue, ExportKind, MediaInfo, SubtitleFormat, Transcript
from dosubs.segment.profiles import profile_for
from dosubs.subs import verify
from dosubs.subs.io import (
    read_subtitles,
    render_cue_lines,
    single_line_profile,
    subtitle_text,
    write_subtitle_file,
)


def _transcript(*cues: Cue) -> Transcript:
    media = MediaInfo(
        path=Path("ExampleVideo.mp4"),
        duration_s=10.0,
        size_bytes=1024,
        has_audio=True,
        audio_codec="aac",
        video_codec="h264",
        container="mov,mp4,m4a,3gp,3g2,mj2",
    )
    return Transcript(media=media, cues=list(cues), source_language="en")


def _cue(cue_id: int = 1, *, start: int = 1000, end: int = 4000) -> Cue:
    return Cue(
        id=cue_id,
        start_ms=start,
        end_ms=end,
        text_original="I never thought we would meet again.",
        text_translated="没想到我们还会再见面。",
    )


def test_original_export_uses_only_the_original_text() -> None:
    lines = render_cue_lines(
        _cue(), ExportKind.ORIGINAL, source_profile=profile_for("en"), target_profile=profile_for("zh-CN")
    )
    assert lines == ["I never thought we would meet again."]


def test_translated_export_uses_only_the_translation() -> None:
    lines = render_cue_lines(
        _cue(),
        ExportKind.TRANSLATED,
        source_profile=profile_for("en"),
        target_profile=profile_for("zh-CN"),
    )
    assert lines == ["没想到我们还会再见面。"]


def test_bilingual_export_puts_original_first_and_one_line_per_language() -> None:
    lines = render_cue_lines(
        _cue(),
        ExportKind.BILINGUAL,
        source_profile=profile_for("en"),
        target_profile=profile_for("zh-CN"),
    )
    assert len(lines) == 2
    assert lines[0] == "I never thought we would meet again."
    assert lines[1] == "没想到我们还会再见面。"


def test_bilingual_export_without_translation_keeps_the_original() -> None:
    cue = _cue().copy(text_translated="")
    lines = render_cue_lines(
        cue,
        ExportKind.BILINGUAL,
        source_profile=profile_for("en"),
        target_profile=profile_for("zh-CN"),
    )
    assert lines == ["I never thought we would meet again."]


def test_single_line_profile_shrinks_the_budget() -> None:
    profile = profile_for("en")
    narrowed = single_line_profile(profile)
    assert narrowed.max_lines == 1
    assert narrowed.max_chars_per_cue == profile.max_chars_per_line


def test_srt_text_is_structurally_valid() -> None:
    transcript = _transcript(_cue(1), _cue(2, start=4500, end=7000))
    text = subtitle_text(transcript, ExportKind.BILINGUAL)
    assert verify.check_srt_text(text) == []
    assert "00:00:01,000 --> 00:00:04,000" in text
    assert "I never thought we would meet again." in text
    assert "没想到我们还会再见面。" in text


def test_check_srt_text_flags_broken_files() -> None:
    assert verify.check_srt_text("") == ["file is empty"]
    assert verify.check_srt_text("not a subtitle") != []
    broken = "1\n00:00:05,000 --> 00:00:01,000\nbackwards\n\n"
    assert any("not after start" in problem for problem in verify.check_srt_text(broken))
    overlapping = (
        "1\n00:00:00,000 --> 00:00:05,000\nfirst\n\n"
        "2\n00:00:03,000 --> 00:00:06,000\nsecond\n\n"
    )
    assert any("before previous" in problem for problem in verify.check_srt_text(overlapping))


def test_written_file_round_trips_with_identical_timings(tmp_path: Path) -> None:
    transcript = _transcript(_cue(1), _cue(2, start=4500, end=7000))
    path = write_subtitle_file(
        transcript, tmp_path / "ExampleVideo.original.srt", ExportKind.ORIGINAL
    )
    loaded = read_subtitles(path)
    assert [event.start for event in loaded] == [1000, 4500]
    assert [event.end for event in loaded] == [4000, 7000]
    assert verify.verify_written_file(path, transcript, ExportKind.ORIGINAL) == []


def test_write_is_atomic_and_leaves_no_part_file(tmp_path: Path) -> None:
    transcript = _transcript(_cue())
    write_subtitle_file(transcript, tmp_path / "a.zh-CN.srt", ExportKind.TRANSLATED)
    assert list(tmp_path.glob("*.part")) == []


def test_utf8_without_bom_by_default_and_optional_bom(tmp_path: Path) -> None:
    transcript = _transcript(_cue())
    plain = write_subtitle_file(
        transcript, tmp_path / "a.original.srt", ExportKind.ORIGINAL, bom=False
    )
    raw = plain.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    # The original file carries no translation, but Chinese text must survive
    # encoding intact (the bilingual file below proves it).
    assert "没想到" not in raw.decode("utf-8")

    with_bom = write_subtitle_file(
        transcript, tmp_path / "a.bilingual.srt", ExportKind.BILINGUAL, bom=True
    )
    decoded = with_bom.read_bytes()
    assert decoded.startswith(b"\xef\xbb\xbf")
    assert "没想到我们还会再见面。" in decoded.decode("utf-8-sig")


def test_verification_detects_timing_drift(tmp_path: Path) -> None:
    transcript = _transcript(_cue())
    path = write_subtitle_file(transcript, tmp_path / "a.original.srt", ExportKind.ORIGINAL)
    transcript.cues[0].end_ms = 5000
    problems = verify.verify_written_file(path, transcript, ExportKind.ORIGINAL)
    assert any("timing drifted" in problem for problem in problems)


def test_verification_detects_a_missing_cue(tmp_path: Path) -> None:
    transcript = _transcript(_cue(1), _cue(2, start=4500, end=7000))
    path = write_subtitle_file(transcript, tmp_path / "a.original.srt", ExportKind.ORIGINAL)
    transcript.cues.pop()
    problems = verify.verify_written_file(path, transcript, ExportKind.ORIGINAL)
    assert any("cues written" in problem for problem in problems)


def test_vtt_export_is_written(tmp_path: Path) -> None:
    transcript = _transcript(_cue())
    path = write_subtitle_file(
        transcript,
        tmp_path / "a.zh-CN.vtt",
        ExportKind.TRANSLATED,
        fmt=SubtitleFormat.VTT,
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "没想到我们还会再见面。" in text


def test_ass_export_uses_a_cjk_capable_font(tmp_path: Path) -> None:
    transcript = _transcript(_cue())
    path = write_subtitle_file(
        transcript, tmp_path / "a.bilingual.ass", ExportKind.BILINGUAL, fmt=SubtitleFormat.ASS
    )
    text = path.read_text(encoding="utf-8")
    assert "Microsoft YaHei" in text
    assert "没想到我们还会再见面。" in text


def test_a_cue_within_budget_wraps_to_at_most_two_lines() -> None:
    profile = profile_for("en")
    text = ("word " * 16).strip()  # 79 characters, inside the per-cue budget of 84
    cue = Cue(id=1, start_ms=0, end_ms=4000, text_original=text, text_translated="")
    lines = render_cue_lines(
        cue, ExportKind.ORIGINAL, source_profile=profile, target_profile=profile_for("zh-CN")
    )
    assert len(lines) <= profile.max_lines
    assert all(len(line) <= profile.max_chars_per_line for line in lines)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_over_budget_text_is_never_truncated() -> None:
    """Wrapping favours completeness: the cue builder is what bounds cue length."""
    profile = profile_for("en")
    long_text = ("word " * 30).strip()  # 149 characters, twice the per-cue budget
    cue = Cue(id=1, start_ms=0, end_ms=8000, text_original=long_text, text_translated="")
    lines = render_cue_lines(
        cue, ExportKind.ORIGINAL, source_profile=profile, target_profile=profile_for("zh-CN")
    )
    assert all(len(line) <= profile.max_chars_per_line for line in lines)
    assert "".join(lines).replace(" ", "") == long_text.replace(" ", "")
