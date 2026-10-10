"""Subtitle editor behaviour.

The editor is the last place a bad timeline can be introduced, so these tests cover
the guards (rejected timecodes, overlap confirmation) as well as the happy path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from dosubs.models import (  # noqa: E402
    Cue,
    ExportKind,
    ExportSettings,
    MediaInfo,
    OverwritePolicy,
    Transcript,
)
from dosubs.subs.io import read_subtitles, write_subtitle_file  # noqa: E402
from dosubs.ui.editor import SubtitleEditor  # noqa: E402


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


@pytest.fixture()
def editor(qtbot, tmp_path: Path) -> SubtitleEditor:
    widget = SubtitleEditor()
    qtbot.addWidget(widget)
    settings = ExportSettings(output_dir=tmp_path / "out", overwrite=OverwritePolicy.OVERWRITE)
    widget.set_export_settings(settings, "zh-CN")
    widget.set_transcript(_transcript())
    return widget


def test_transcript_is_shown_with_both_languages(editor: SubtitleEditor) -> None:
    assert editor.table.rowCount() == 2
    assert editor.table.item(0, 3).text() == "I never thought we would meet again."
    assert editor.table.item(0, 4).text() == "没想到我们还会再见面。"
    assert editor.table.item(0, 0).text() == "1"
    assert editor.table.item(1, 1).text() == "00:00:04,500"


def test_editing_translation_text_is_captured(editor: SubtitleEditor) -> None:
    editor.table.item(0, 4).setText("我们竟然又见面了。")
    transcript = editor.transcript()
    assert transcript.cues[0].text_translated == "我们竟然又见面了。"
    assert transcript.cues[1].text_translated == "你怎么会在这里？"
    assert editor.is_dirty()


def test_editing_a_timestamp_is_parsed_and_normalised(editor: SubtitleEditor) -> None:
    editor.table.item(1, 2).setText("00:00:09.250")  # dotted form is accepted
    assert editor.table.item(1, 2).text() == "00:00:09,250"
    assert editor.transcript().cues[1].end_ms == 9250


def test_an_invalid_timecode_is_rejected_and_reverted(
    editor: SubtitleEditor, qtbot
) -> None:
    messages: list[str] = []
    editor.status_message.connect(messages.append)

    editor.table.item(0, 1).setText("not a time")

    assert editor.table.item(0, 1).text().startswith("00:00:01")
    assert any("invalid timecode" in message for message in messages)
    assert editor.transcript().cues[0].start_ms == 1000


def test_cue_can_be_deleted_and_rows_are_renumbered(editor: SubtitleEditor) -> None:
    editor.table.setCurrentCell(0, 3)
    editor.delete_selected_cue()
    assert editor.table.rowCount() == 1
    assert editor.table.item(0, 0).text() == "1"
    assert editor.transcript().cues[0].text_original == "What are you doing here?"


def test_a_cue_can_be_added_after_the_last_one(editor: SubtitleEditor) -> None:
    editor.add_cue()
    assert editor.table.rowCount() == 3
    cue = editor.transcript().cues[2]
    assert cue.start_ms >= 7000
    assert cue.end_ms > cue.start_ms


def test_shifting_refuses_to_create_an_overlap(editor: SubtitleEditor) -> None:
    messages: list[str] = []
    editor.status_message.connect(messages.append)

    editor.table.setCurrentCell(1, 3)
    for _ in range(8):
        editor.shift_selected(-100)

    cue = editor.transcript().cues[1]
    # The cue slides to the tightest legal position and stops there: it must keep a
    # positive gap to the previous cue rather than overlapping it.
    assert cue.start_ms == 4100
    assert cue.start_ms > editor.transcript().cues[0].end_ms
    assert any("would overlap the previous cue" in message for message in messages)


def test_shifting_the_last_cue_cannot_run_past_the_next_one(editor: SubtitleEditor) -> None:
    messages: list[str] = []
    editor.status_message.connect(messages.append)

    editor.table.setCurrentCell(0, 3)
    for _ in range(20):
        editor.shift_selected(100)

    cue = editor.transcript().cues[0]
    assert cue.end_ms < editor.transcript().cues[1].start_ms
    assert any("would overlap the next cue" in message for message in messages)


def test_shifting_within_the_gap_moves_both_edges(editor: SubtitleEditor) -> None:
    editor.table.setCurrentCell(0, 3)
    editor.shift_selected(100)
    cue = editor.transcript().cues[0]
    assert (cue.start_ms, cue.end_ms) == (1100, 4100)


def test_preview_shows_the_bilingual_rendering(editor: SubtitleEditor) -> None:
    editor.table.setCurrentCell(0, 3)
    preview = editor.preview_label.text()
    assert "I never thought we would meet again." in preview
    assert "没想到我们还会再见面。" in preview


def test_re_export_writes_all_three_files(editor: SubtitleEditor, tmp_path: Path) -> None:
    editor.table.item(0, 4).setText("我们竟然又见面了。")
    written = editor.save()

    names = sorted(path.name for path in written)
    assert names == [
        "ExampleVideo.bilingual.srt",
        "ExampleVideo.original.srt",
        "ExampleVideo.zh-CN.srt",
    ]
    assert not editor.is_dirty()

    timelines = {
        name: [(event.start, event.end) for event in read_subtitles(tmp_path / "out" / name)]
        for name in names
    }
    assert timelines["ExampleVideo.original.srt"] == timelines["ExampleVideo.zh-CN.srt"]
    assert timelines["ExampleVideo.bilingual.srt"] == timelines["ExampleVideo.original.srt"]
    assert "我们竟然又见面了。" in (tmp_path / "out" / "ExampleVideo.zh-CN.srt").read_text(
        encoding="utf-8"
    )


def test_a_broken_timeline_asks_before_writing(
    editor: SubtitleEditor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.table.item(1, 1).setText("00:00:02,000")  # overlaps the first cue
    asked: list[str] = []

    def refuse(*args, **kwargs):
        asked.append(args[2] if len(args) > 2 else "")
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "warning", refuse)
    assert editor.save() == []
    assert asked, "the user must be warned before writing a broken timeline"
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.srt"))


def test_a_broken_timeline_can_be_forced_through(
    editor: SubtitleEditor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.table.item(1, 1).setText("00:00:02,000")
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Yes)

    written = editor.save()
    assert len(written) == 3
    timeline = [(event.start, event.end) for event in read_subtitles(written[0])]
    assert timeline == [(1000, 4000), (2000, 7000)]  # exactly what the user asked for


def test_editor_round_trips_files_written_by_the_pipeline(tmp_path: Path) -> None:
    transcript = _transcript()
    original = write_subtitle_file(
        transcript, tmp_path / "ExampleVideo.original.srt", ExportKind.ORIGINAL
    )
    translated = write_subtitle_file(
        transcript, tmp_path / "ExampleVideo.zh-CN.srt", ExportKind.TRANSLATED
    )

    widget = SubtitleEditor()
    widget.set_export_settings(ExportSettings(output_dir=tmp_path), "zh-CN")
    assert widget.load_files(_media(), original=original, translated=translated)
    assert widget.table.rowCount() == 2
    assert widget.transcript().cues[1].text_translated == "你怎么会在这里？"

    widget.table.item(1, 4).setText("你在这儿干什么？")
    written = widget.save()
    reloaded = read_subtitles(next(path for path in written if path.name.endswith("zh-CN.srt")))
    assert reloaded[1].text == "你在这儿干什么？"


def test_editor_has_no_transcript_until_something_is_loaded() -> None:
    widget = SubtitleEditor()
    assert widget.table.rowCount() == 0
    assert not widget.save_button.isEnabled()
    with pytest.raises(ValueError):
        widget.transcript()
