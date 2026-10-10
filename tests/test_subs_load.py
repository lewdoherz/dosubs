"""Loading subtitle files back into a transcript (the basis for editing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dosubs.models import Cue, ExportKind, MediaInfo, SubtitleFormat, Transcript
from dosubs.subs.io import write_subtitle_file
from dosubs.subs.load import split_bilingual_text, transcript_from_subtitles


def _media(path: Path | None = None) -> MediaInfo:
    return MediaInfo(
        path=path or Path("ExampleVideo.mp4"),
        duration_s=10.0,
        size_bytes=1,
        has_audio=True,
        audio_codec="aac",
        video_codec="h264",
    )


def _transcript() -> Transcript:
    return Transcript(
        media=_media(),
        source_language="en",
        cues=[
            Cue(
                id=1,
                start_ms=1000,
                end_ms=4000,
                text_original="I never thought we would meet again.",
                text_translated="没想到我们还会再见面。",
            ),
            Cue(
                id=2,
                start_ms=4500,
                end_ms=7000,
                text_original="What are you doing here?",
                text_translated="你怎么会在这里？",
            ),
        ],
    )


def _write_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    transcript = _transcript()
    original = write_subtitle_file(
        transcript, tmp_path / "ExampleVideo.original.srt", ExportKind.ORIGINAL
    )
    translated = write_subtitle_file(
        transcript, tmp_path / "ExampleVideo.zh-CN.srt", ExportKind.TRANSLATED
    )
    bilingual = write_subtitle_file(
        transcript, tmp_path / "ExampleVideo.bilingual.srt", ExportKind.BILINGUAL
    )
    return original, translated, bilingual


def test_original_plus_translated_round_trips(tmp_path: Path) -> None:
    original, translated, _ = _write_pair(tmp_path)
    loaded = transcript_from_subtitles(
        _media(), original=original, translated=translated
    )
    assert [(c.start_ms, c.end_ms) for c in loaded.cues] == [(1000, 4000), (4500, 7000)]
    assert loaded.cues[0].text_original == "I never thought we would meet again."
    assert loaded.cues[0].text_translated == "没想到我们还会再见面。"
    assert loaded.cues[1].text_translated == "你怎么会在这里？"
    assert loaded.warnings == []


def test_bilingual_file_carries_both_languages(tmp_path: Path) -> None:
    _, _, bilingual = _write_pair(tmp_path)
    loaded = transcript_from_subtitles(_media(), bilingual=bilingual)
    assert len(loaded.cues) == 2
    assert loaded.cues[0].text_original == "I never thought we would meet again."
    assert loaded.cues[0].text_translated == "没想到我们还会再见面。"


def test_reader_normalises_pysubs2_line_separators(tmp_path: Path) -> None:
    """pysubs2 reads SRT newlines back as a literal ``\\N``; the reader must undo that."""
    path = tmp_path / "multi.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nfirst line\nsecond line\n\n", encoding="utf-8"
    )
    loaded = transcript_from_subtitles(_media(), original=path)
    assert loaded.cues[0].text_original == "first line second line"
    assert "\\N" not in loaded.cues[0].text_original


def test_bilingual_split_uses_the_script_change_for_wrapped_originals() -> None:
    # This is the shape a real run produced: a long original wrapped over two lines
    # above a one-line Chinese translation.
    text = "Hey, rookie, consider this my welcome\ngift.\n嘿，菜鸟，就当这是我给你的见面礼。"
    original, translated = split_bilingual_text(text)
    assert original == "Hey, rookie, consider this my welcome gift."
    assert translated == "嘿，菜鸟，就当这是我给你的见面礼。"


def test_bilingual_split_of_a_single_line_pair() -> None:
    assert split_bilingual_text("Hello there.\n你好。") == ("Hello there.", "你好。")
    assert split_bilingual_text("Only original.") == ("Only original.", "")
    assert split_bilingual_text("   ") == ("", "")


def test_bilingual_split_falls_back_when_both_scripts_match() -> None:
    original, translated = split_bilingual_text("Hello there.\nHola amigo.")
    assert original == "Hello there."
    assert translated == "Hola amigo."


def test_translated_file_with_an_extra_cue_is_not_trusted(tmp_path: Path) -> None:
    original, translated, _ = _write_pair(tmp_path)
    # A stale translated file that also holds an older, longer timeline.
    stray = Transcript(
        media=_media(),
        cues=[
            Cue(id=1, start_ms=0, end_ms=500, text_original="stale", text_translated="陈旧"),
            Cue(id=2, start_ms=1000, end_ms=4000, text_original="x", text_translated="没错"),
            Cue(id=3, start_ms=4500, end_ms=7000, text_original="y", text_translated="你怎么会在这里？"),
        ],
    )
    write_subtitle_file(stray, translated, ExportKind.TRANSLATED, encoding="utf-8")

    loaded = transcript_from_subtitles(_media(), original=original, translated=translated)
    assert loaded.cues[0].text_translated == "没错"  # matched by timing, not position
    assert loaded.cues[1].text_translated == "你怎么会在这里？"
    assert any("did not line up" in warning for warning in loaded.warnings)


def test_missing_translation_is_reported(tmp_path: Path) -> None:
    original, translated, _ = _write_pair(tmp_path)
    only_first = Transcript(media=_media(), source_language="en", cues=[_transcript().cues[0]])
    write_subtitle_file(only_first, translated, ExportKind.TRANSLATED)

    loaded = transcript_from_subtitles(_media(), original=original, translated=translated)
    assert [cue.is_translated for cue in loaded.cues] == [True, False]
    assert any("no matching translation" in warning for warning in loaded.warnings)


def test_ass_files_load_too(tmp_path: Path) -> None:
    transcript = _transcript()
    path = write_subtitle_file(
        transcript,
        tmp_path / "ExampleVideo.zh-CN.ass",
        ExportKind.TRANSLATED,
        fmt=SubtitleFormat.ASS,
    )
    loaded = transcript_from_subtitles(_media(), original=path)
    assert len(loaded.cues) == 2
    assert loaded.cues[0].text_original == "没想到我们还会再见面。"


def test_unicode_and_spaced_paths_round_trip(tmp_path: Path) -> None:
    directory = tmp_path / "我的 字幕"
    transcript = _transcript()
    original = write_subtitle_file(
        transcript, directory / "我的 视频.original.srt", ExportKind.ORIGINAL
    )
    translated = write_subtitle_file(
        transcript, directory / "我的 视频.zh-CN.srt", ExportKind.TRANSLATED
    )
    loaded = transcript_from_subtitles(
        _media(tmp_path / "我的 视频.mp4"), original=original, translated=translated
    )
    assert loaded.cues[1].text_translated == "你怎么会在这里？"


def test_requires_a_source_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either an original subtitle file or a bilingual one"):
        transcript_from_subtitles(_media(), translated=tmp_path / "whatever.srt")


def test_unreadable_file_raises_a_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.srt"
    broken.write_text("this is not a subtitle file at all", encoding="utf-8")
    with pytest.raises(ValueError):
        transcript_from_subtitles(_media(), original=broken)


def test_edited_transcript_re_exports_identically(tmp_path: Path) -> None:
    """The editor's whole point: edit text, re-export, same timeline."""
    original, translated, _ = _write_pair(tmp_path)
    loaded = transcript_from_subtitles(_media(), original=original, translated=translated)
    loaded.cues[0].text_translated = "我们竟然又见面了。"

    out = tmp_path / "edited"
    rewritten = write_subtitle_file(
        loaded, out / "ExampleVideo.zh-CN.srt", ExportKind.TRANSLATED
    )
    reloaded = transcript_from_subtitles(_media(), original=rewritten)
    assert reloaded.cues[0].text_original == "我们竟然又见面了。"
    assert [(c.start_ms, c.end_ms) for c in reloaded.cues] == [(1000, 4000), (4500, 7000)]
