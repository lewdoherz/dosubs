"""Subtitle editor panel: review, correct and re-export without re-running the pipeline.

The editor works on a :class:`~dosubs.models.Transcript`, the same canonical object the
pipeline exports from, so editing text or nudging a timestamp and re-exporting cannot
desynchronise the three output files: they are all rendered again from this one
timeline.

Timing problems are surfaced before anything is written. A cue with a bad timecode is
rejected as you type it, and a timeline with overlaps or zero-length cues asks for
confirmation instead of quietly producing broken subtitles.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..logging_setup import get_logger
from ..media.probe import probe_media
from ..models import (
    Cue,
    ExportKind,
    ExportSettings,
    MediaInfo,
    OverwritePolicy,
    SubtitleFormat,
    Transcript,
)
from ..segment.profiles import profile_for, wrap_text
from ..subs import timecode
from ..subs.io import write_subtitle_file
from ..subs.load import transcript_from_subtitles
from ..subs.naming import plan_exports
from ..subs.verify import verify_written_file

log = get_logger(__name__)

__all__ = ["SubtitleEditor"]

_TIME_COLUMNS = (1, 2)
_CJK_FONT = "Microsoft YaHei UI"


class SubtitleEditor(QWidget):
    """Editable view of a transcript with a bilingual preview and re-export."""

    status_message = Signal(str)
    saved = Signal(list)

    COLUMNS = ("#", "Start", "End", "Original", "Translation")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._transcript: Transcript | None = None
        self._settings = ExportSettings()
        self._target_language = "zh-CN"
        self._dirty = False
        self._loading = False
        # Last known-good value per time cell, so rejecting a bad edit can restore it.
        self._valid_times: dict[tuple[int, int], int] = {}

        self.info_label = QLabel("No subtitles loaded. Open a subtitle file to review or edit it.")
        self.info_label.setWordWrap(True)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setFont(QFont(_CJK_FONT, 10))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._update_preview)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setFont(QFont(_CJK_FONT, 11))

        self.open_button = QPushButton("Open subtitles…")
        self.open_button.clicked.connect(self.open_files_dialog)
        self.add_button = QPushButton("Add cue")
        self.add_button.clicked.connect(self.add_cue)
        self.delete_button = QPushButton("Delete cue")
        self.delete_button.clicked.connect(self.delete_selected_cue)
        self.earlier_button = QPushButton("−100 ms")
        self.earlier_button.clicked.connect(lambda: self.shift_selected(-100))
        self.later_button = QPushButton("+100 ms")
        self.later_button.clicked.connect(lambda: self.shift_selected(100))
        self.save_button = QPushButton("Re-export subtitles")
        self.save_button.clicked.connect(self.save)
        self.save_as_button = QPushButton("Re-export to…")
        self.save_as_button.clicked.connect(self.save_as)

        buttons = QHBoxLayout()
        for widget in (
            self.open_button,
            self.add_button,
            self.delete_button,
            self.earlier_button,
            self.later_button,
        ):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        buttons.addWidget(self.save_as_button)
        buttons.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.preview_label)
        layout.addLayout(buttons)
        self._set_editing_enabled(False)

    # -- public API --------------------------------------------------------- #

    def set_export_settings(self, settings: ExportSettings, target_language: str = "zh-CN") -> None:
        """Remember where and how to re-export (handed over by the main window)."""
        self._settings = settings
        self._target_language = target_language

    def set_transcript(self, transcript: Transcript) -> None:
        """Show a transcript, e.g. the result of a run that just finished."""
        self._transcript = transcript
        self._dirty = False
        self._fill_table()
        self._set_editing_enabled(True)
        self.info_label.setText(
            f"{transcript.media.path.name}: {len(transcript.cues)} cue(s), "
            f"source language {transcript.source_language or 'unknown'}, "
            f"target {self._target_language}"
        )
        if transcript.warnings:
            self.status_message.emit("; ".join(transcript.warnings))

    def load_files(
        self,
        media: MediaInfo,
        *,
        original: Path | None = None,
        translated: Path | None = None,
        bilingual: Path | None = None,
    ) -> bool:
        """Load subtitles from disk into the editor."""
        try:
            transcript = transcript_from_subtitles(
                media, original=original, translated=translated, bilingual=bilingual
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Could not open subtitles", str(exc))
            return False
        self.set_transcript(transcript)
        return True

    def open_files_dialog(self) -> bool:
        """Ask for an original (or bilingual) subtitle file and load it."""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open subtitle file",
            str(self._settings.output_dir or ""),
            "Subtitles (*.srt *.ass *.vtt)",
        )
        if not chosen:
            return False
        path = Path(chosen)
        media = self._media_for(path)
        return self.load_files(media, original=path)

    def transcript(self) -> Transcript:
        """The current edited transcript.

        Raises:
            ValueError: if a timecode cell holds something unparseable or a cue ends
                before it starts — the table normally prevents both, so this is a
                last line of defence before writing files.
        """
        if self._transcript is None:
            raise ValueError("no transcript is loaded")
        cues: list[Cue] = []
        for row in range(self.table.rowCount()):
            start = self._read_time(row, 1)
            end = self._read_time(row, 2)
            if end <= start:
                raise ValueError(
                    f"cue {row + 1}: end ({timecode.format_short(end)}) is not after start "
                    f"({timecode.format_short(start)})"
                )
            cues.append(
                Cue(
                    id=row + 1,
                    start_ms=start,
                    end_ms=end,
                    text_original=self._cell_text(row, 3),
                    text_translated=self._cell_text(row, 4),
                )
            )
        return Transcript(
            media=self._transcript.media,
            cues=cues,
            source_language=self._transcript.source_language,
            language_probability=self._transcript.language_probability,
            language_forced=self._transcript.language_forced,
            asr_params=dict(self._transcript.asr_params),
        )

    def is_dirty(self) -> bool:
        return self._dirty

    def save(self) -> list[Path]:
        """Re-export into the configured output directory, overwriting the files."""
        return self._export(self._settings.output_dir, OverwritePolicy.OVERWRITE)

    def save_as(self) -> list[Path]:
        """Re-export into a directory the user picks."""
        directory = QFileDialog.getExistingDirectory(
            self, "Choose an output folder", str(self._settings.output_dir or "")
        )
        if not directory:
            return []
        return self._export(Path(directory), OverwritePolicy.OVERWRITE)

    # -- editing operations ------------------------------------------------- #

    def add_cue(self) -> None:
        row = self.table.rowCount()
        previous_end = self._read_time(row - 1, 2) if row else 0
        start = previous_end + 100 if row else 0
        end = start + 1500
        self._append_row(row, start, end, "", "")
        self.table.setCurrentCell(row, 3)
        self._mark_dirty()

    def delete_selected_cue(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            self.status_message.emit("select a cue to delete")
            return
        self._loading = True
        self.table.removeRow(row)
        self._loading = False
        self._renumber()
        self._mark_dirty()

    def shift_selected(self, delta_ms: int) -> None:
        """Nudge the selected cue's timing (keeps the order, refuses to cross neighbours)."""
        row = self.table.currentRow()
        if row < 0:
            self.status_message.emit("select a cue to shift")
            return
        start = self._read_time(row, 1)
        end = self._read_time(row, 2)
        new_start = max(0, start + delta_ms)
        new_end = end + delta_ms
        if row > 0 and new_start <= self._read_time_safe(row - 1, 2):
            self.status_message.emit("shift refused: it would overlap the previous cue")
            return
        if row + 1 < self.table.rowCount() and new_end >= self._read_time_safe(row + 1, 1):
            self.status_message.emit("shift refused: it would overlap the next cue")
            return
        self._loading = True
        self._write_time(row, 1, new_start)
        self._write_time(row, 2, new_end)
        self._loading = False
        self._mark_dirty()

    # -- internals ---------------------------------------------------------- #

    def _media_for(self, subtitle_path: Path) -> MediaInfo:
        """Find the media file a subtitle belongs to, falling back to a stub."""
        stem = subtitle_path.stem
        for suffix in ("original", self._target_language, "bilingual"):
            if stem.endswith(f".{suffix}"):
                stem = stem[: -(len(suffix) + 1)]
                break
        for extension in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac"):
            candidate = subtitle_path.with_name(stem + extension)
            if candidate.exists():
                try:
                    return probe_media(candidate)
                except Exception as exc:  # probing is a nicety here, not a requirement
                    log.warning("could not probe %s: %s", candidate, exc)
                    break
        return MediaInfo(
            path=subtitle_path.with_name(stem),
            duration_s=0.0,
            size_bytes=0,
            has_audio=True,
        )

    def _set_editing_enabled(self, enabled: bool) -> None:
        for widget in (
            self.add_button,
            self.delete_button,
            self.earlier_button,
            self.later_button,
            self.save_button,
            self.save_as_button,
            self.table,
        ):
            widget.setEnabled(enabled)

    def _fill_table(self) -> None:
        self._loading = True
        try:
            self.table.setRowCount(0)
            assert self._transcript is not None
            for cue in self._transcript.cues:
                self._append_row(
                    self.table.rowCount(),
                    cue.start_ms,
                    cue.end_ms,
                    cue.text_original,
                    cue.text_translated,
                )
        finally:
            self._loading = False
        self._reindex_valid_times()
        self._update_preview()

    def _append_row(self, row: int, start_ms: int, end_ms: int, original: str, translated: str) -> None:
        self.table.insertRow(row)
        index_item = QTableWidgetItem(str(row + 1))
        index_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(row, 0, index_item)
        for column, value in ((1, start_ms), (2, end_ms)):
            item = QTableWidgetItem(timecode.format_srt(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, column, item)
        for column, text in ((3, original), (4, translated)):
            item = QTableWidgetItem(text)
            item.setFont(QFont(_CJK_FONT, 10))
            self.table.setItem(row, column, item)

    def _renumber(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None:
                item.setText(str(row + 1))
        self._reindex_valid_times()

    def _cell_text(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _read_time(self, row: int, column: int) -> int:
        if row < 0 or row >= self.table.rowCount():
            return 0
        try:
            return timecode.parse_timecode(self._cell_text(row, column))
        except ValueError as exc:
            raise ValueError(f"cue {row + 1}: {exc}") from exc

    def _write_time(self, row: int, column: int, ms: int) -> None:
        item = self.table.item(row, column)
        if item is not None:
            item.setText(timecode.format_srt(ms))
        self._valid_times[(row, column)] = ms

    def _reindex_valid_times(self) -> None:
        """Rebuild the last-known-good map after rows are added, removed or renumbered."""
        self._valid_times = {}
        for row in range(self.table.rowCount()):
            for column in _TIME_COLUMNS:
                try:
                    self._valid_times[(row, column)] = timecode.parse_timecode(
                        self._cell_text(row, column)
                    )
                except ValueError:
                    continue

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        row, column = item.row(), item.column()
        if column in _TIME_COLUMNS:
            self._validate_time_cell(row, column)
        self._mark_dirty()
        self._update_preview()

    def _validate_time_cell(self, row: int, column: int) -> None:
        """Reject an unparseable timecode as it is typed and restore the previous one."""
        try:
            value = timecode.parse_timecode(self._cell_text(row, column))
        except ValueError as exc:
            self.status_message.emit(f"invalid timecode, change rejected: {exc}")
            self._loading = True
            self._write_time(row, column, self._valid_times.get((row, column), 0))
            self._loading = False
            return
        if column == 1 and row > 0 and value < self._valid_times.get((row - 1, 2), 0):
            self.status_message.emit("start is before the previous cue ends; shift the cue instead")
        self._write_time(row, column, value)

    def _read_time_safe(self, row: int, column: int) -> int:
        try:
            return self._read_time(row, column)
        except ValueError:
            return self._valid_times.get((row, column), 0)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.info_label.setText(
            f"{self.table.rowCount()} cue(s) — edited, not yet re-exported"
            if self._transcript
            else ""
        )

    def _update_preview(self) -> None:
        """Show the selected cue exactly as it will be written (bilingual rendering)."""
        row = self.table.currentRow()
        if row < 0 or self._transcript is None:
            self.preview_label.clear()
            return
        original = self._cell_text(row, 3)
        translated = self._cell_text(row, 4)
        source_profile = profile_for(self._transcript.source_language)
        target_profile = profile_for(self._target_language)
        lines: list[str] = []
        if original:
            lines.extend(wrap_text(original, source_profile))
        if translated:
            lines.extend(wrap_text(translated, target_profile))
        self.preview_label.setText("Preview (bilingual): " + " / ".join(lines))

    def _export(self, output_dir: Path | None, policy: OverwritePolicy) -> list[Path]:
        if self._transcript is None:
            self.status_message.emit("nothing to export")
            return []
        try:
            transcript = self.transcript()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot export", str(exc))
            return []

        problems = transcript.validate_timeline()
        if problems:
            preview = "\n".join(problems[:8])
            more = "" if len(problems) <= 8 else f"\n… and {len(problems) - 8} more"
            answer = QMessageBox.warning(
                self,
                "Timeline problems",
                f"The edited timeline has {len(problems)} problem(s):\n\n{preview}{more}\n\n"
                "Export anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer is not QMessageBox.StandardButton.Yes:
                return []

        plan = plan_exports(
            transcript.media.path,
            self._settings.wanted_kinds(),
            target_language=self._target_language,
            fmt=self._settings.format,
            output_dir=output_dir,
            policy=policy,
        )
        written: list[Path] = []
        verification: list[str] = []
        for target in plan.targets:
            path = write_subtitle_file(
                transcript,
                target.path,
                target.kind,
                target_language=self._target_language,
                fmt=self._settings.format,
                encoding=self._settings.encoding,
                bom=self._settings.bom,
            )
            written.append(path)
            verification.extend(
                verify_written_file(
                    path,
                    transcript,
                    target.kind,
                    expect_translation=target.kind is not ExportKind.ORIGINAL,
                )
            )
        self._transcript = transcript
        self._dirty = False
        message = f"re-exported {len(written)} file(s)"
        if verification:
            message += f" with {len(verification)} verification problem(s): {verification[0]}"
            log.error("editor export verification: %s", verification)
        self.info_label.setText(f"{len(transcript.cues)} cue(s) — {message}")
        self.status_message.emit(message)
        self.saved.emit(written)
        return written
