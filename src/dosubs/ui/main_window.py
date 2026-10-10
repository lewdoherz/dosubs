"""Main window: media list, settings, progress and the log pane.

Everything the window does is either Qt bookkeeping or handing work to
:class:`~dosubs.ui.worker.PipelineWorker`. The GUI thread never runs recognition,
translation or a long probe: file metadata is filled in through a
``QTimer.singleShot`` chain, and every pipeline step arrives as a signal.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, get_api_key, mask_secret, save_config, set_api_key
from ..gpu import probe as probe_gpu
from ..media.probe import SUPPORTED_EXT, MediaError, expand_inputs, probe_media
from ..models import (
    PROVIDER_PRESETS,
    JobResult,
    JobSpec,
    JobStatus,
    OverwritePolicy,
    ProgressEvent,
    SubtitleFormat,
    TranslationSettings,
)
from ..subs.timecode import format_duration
from ..subs import naming
from .editor import SubtitleEditor
from .worker import ConflictRequest, PipelineWorker

log = logging.getLogger(__name__)

__all__ = ["MainWindow", "cjk_font", "gpu_summary_text"]

#: Model families faster-whisper ships; ``large-v3`` is the quality default.
MODEL_CHOICES = ("large-v3", "large-v3-turbo", "medium", "small", "base", "tiny")
DEVICE_CHOICES = ("auto", "cuda", "cpu")
COMPUTE_TYPE_CHOICES = ("auto", "float16", "int8_float16", "int8", "float32")
SOURCE_LANGUAGE_CHOICES: tuple[tuple[str, str | None], ...] = (
    ("Auto detect", None),
    ("en (English)", "en"),
    ("zh (Chinese)", "zh"),
    ("ja (Japanese)", "ja"),
    ("ko (Korean)", "ko"),
    ("es (Spanish)", "es"),
    ("fr (French)", "fr"),
    ("de (German)", "de"),
    ("ru (Russian)", "ru"),
    ("it (Italian)", "it"),
    ("pt (Portuguese)", "pt"),
    ("vi (Vietnamese)", "vi"),
    ("th (Thai)", "th"),
    ("ar (Arabic)", "ar"),
    ("hi (Hindi)", "hi"),
)
TARGET_LANGUAGE_CHOICES = ("zh-CN", "zh-TW", "ja", "ko", "es", "fr", "de", "ru", "en")
FORMAT_CHOICES: tuple[tuple[str, SubtitleFormat], ...] = (
    ("srt", SubtitleFormat.SRT),
    ("ass", SubtitleFormat.ASS),
    ("vtt", SubtitleFormat.VTT),
)
OVERWRITE_CHOICES: tuple[tuple[str, OverwritePolicy], ...] = (
    ("ask", OverwritePolicy.ASK),
    ("overwrite", OverwritePolicy.OVERWRITE),
    ("skip", OverwritePolicy.SKIP),
    ("rename", OverwritePolicy.RENAME),
)

#: Shown in the duration column until ffprobe answers.
_PENDING_DURATION = "..."
_LOG_PANE_MAX_BLOCKS = 5000

_gpu_summary_cache: list[str] = []


def gpu_summary_text() -> str:
    """One line describing usable acceleration, computed once per process.

    ``gpu.probe()`` never raises, but a broken install should still read as a
    sentence in the window rather than a traceback, so the whole thing is guarded.
    """
    if not _gpu_summary_cache:
        try:
            result = probe_gpu()
            text = result.summary()
            if not result.cuda_available:
                text += (
                    f" - recommended: {result.recommended_device()} / "
                    f"{result.recommended_compute_type()}"
                )
        except Exception as exc:  # defensive: the UI must always have something to show
            text = f"Acceleration could not be checked ({exc}); transcription will use the CPU"
        _gpu_summary_cache.append(text)
    return _gpu_summary_cache[0]


def cjk_font(point_size: int | None = None) -> QFont:
    """Prefer a font that has CJK glyphs, so Chinese subtitles are never tofu.

    The families are a preference list: Qt substitutes the first one the system
    actually has, which also works when the platform plugin reports no font list.
    """
    font = QFont()
    font.setFamilies(
        ["Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "SimSun"]
    )
    if point_size is not None:
        font.setPointSize(point_size)
    return font


def _monospace_font() -> QFont:
    """Fixed-width for the log pane, with a CJK fallback for translated lines."""
    font = QFont()
    font.setFamilies(["Consolas", "Cascadia Mono", "Microsoft YaHei UI", "Courier New"])
    font.setPointSize(9)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def _path_key(path: Path) -> str:
    """Case- and separator-insensitive identity for de-duplicating the file list."""
    try:
        resolved = path.resolve()
    except OSError:  # a path we cannot resolve still needs a key
        resolved = path
    return str(resolved).replace("\\", "/").casefold()


class _ProviderTester(QThread):
    """One provider round-trip on a short-lived thread so the window keeps painting."""

    checked = Signal(bool, str)

    def __init__(self, settings: TranslationSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings

    def run(self) -> None:
        from ..translate.openai_compat import build_provider

        try:
            provider = build_provider(self._settings)
        except Exception as exc:
            self.checked.emit(False, f"{type(exc).__name__}: {exc}")
            return
        try:
            self.checked.emit(True, provider.check_connection())
        except Exception as exc:
            self.checked.emit(False, f"{type(exc).__name__}: {exc}")
        finally:
            try:
                provider.close()
            except Exception:  # closing is best effort
                pass


class MainWindow(QMainWindow):
    """The DoSubs desktop window."""

    def __init__(
        self,
        config: AppConfig,
        *,
        pipeline_factory: Callable[..., object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self._pipeline_factory = pipeline_factory

        self.worker: PipelineWorker | None = None
        self._tester: _ProviderTester | None = None
        self._running = False
        self._loading = True
        self._files: list[Path] = []
        self._row_of: dict[str, int] = {}
        self._probe_queue: list[int] = []
        self._probing = False

        self.setAcceptDrops(True)
        self.setWindowTitle("DoSubs")
        self._build_ui()
        self._apply_config_to_widgets()
        self._connect_settings_signals()
        self._loading = False
        self._update_buttons()

    # ------------------------------------------------------------------ UI -- #

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical, self)

        files_group = self._build_files_group()
        self._files_group = files_group
        splitter.addWidget(files_group)

        settings_row = QWidget(self)
        row_layout = QHBoxLayout(settings_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._recognition_group = self._build_recognition_group()
        self._translation_group = self._build_translation_group()
        self._export_group = self._build_export_group()
        row_layout.addWidget(self._recognition_group, 1)
        row_layout.addWidget(self._translation_group, 1)
        row_layout.addWidget(self._export_group, 1)
        splitter.addWidget(settings_row)

        processing_group = self._build_processing_group()
        self._processing_group = processing_group
        splitter.addWidget(processing_group)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 3)

        self.editor = SubtitleEditor(self)
        self.editor.status_message.connect(self._append_log)
        self.editor.saved.connect(self._on_editor_saved)

        queue_page = QWidget(self)
        queue_layout = QVBoxLayout(queue_page)
        queue_layout.addWidget(splitter)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(queue_page, "Queue")
        self.tabs.addTab(self.editor, "Subtitles")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_files_group(self) -> QGroupBox:
        group = QGroupBox("Files", self)

        self.table = QTableWidget(0, 4, group)
        self.table.setHorizontalHeaderLabels(["File", "Duration", "Status", "Output"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAcceptDrops(False)  # let drops bubble up to the window
        self.table.viewport().setAcceptDrops(False)
        self.table.setFont(cjk_font())
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        self.add_files_button = QPushButton("Add files", group)
        self.add_folder_button = QPushButton("Add folder", group)
        self.remove_button = QPushButton("Remove selected", group)
        self.clear_button = QPushButton("Clear", group)
        self.add_files_button.clicked.connect(self.add_files)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_files)

        buttons = QHBoxLayout()
        for button in (
            self.add_files_button,
            self.add_folder_button,
            self.remove_button,
            self.clear_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)

        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.table)
        return group

    def _build_recognition_group(self) -> QGroupBox:
        group = QGroupBox("Recognition", self)
        form = QFormLayout(group)

        self.model_combo = QComboBox(group)
        self.model_combo.addItems(MODEL_CHOICES)
        self.device_combo = QComboBox(group)
        self.device_combo.addItems(DEVICE_CHOICES)
        self.compute_combo = QComboBox(group)
        self.compute_combo.addItems(COMPUTE_TYPE_CHOICES)
        self.source_language_combo = QComboBox(group)
        for label, code in SOURCE_LANGUAGE_CHOICES:
            self.source_language_combo.addItem(label, code)

        self.vad_check = QCheckBox("Voice activity detection", group)
        self.words_check = QCheckBox("Word-level timestamps", group)
        self.batch_spin = QSpinBox(group)
        self.batch_spin.setRange(0, 32)
        self.batch_spin.setSpecialValueText("0 (sequential)")
        self.batch_spin.setToolTip(
            "Above 0 enables batched inference: faster, with coarser raw segments."
        )

        self.gpu_label = QLabel(gpu_summary_text(), group)
        self.gpu_label.setWordWrap(True)
        self.gpu_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form.addRow("Model", self.model_combo)
        form.addRow("Device", self.device_combo)
        form.addRow("Compute type", self.compute_combo)
        form.addRow("Source language", self.source_language_combo)
        form.addRow("", self.vad_check)
        form.addRow("", self.words_check)
        form.addRow("Batch size", self.batch_spin)
        form.addRow(QLabel("Acceleration", group))
        form.addRow(self.gpu_label)
        return group

    def _build_translation_group(self) -> QGroupBox:
        group = QGroupBox("Translation", self)
        form = QFormLayout(group)

        self.translate_check = QCheckBox("Enable translation", group)
        self.target_language_combo = QComboBox(group)
        self.target_language_combo.addItems(TARGET_LANGUAGE_CHOICES)
        self.provider_combo = QComboBox(group)
        self.provider_combo.addItems(sorted(PROVIDER_PRESETS))
        self.model_edit = QLineEdit(group)
        self.base_url_edit = QLineEdit(group)
        self.provider_note_label = QLabel("", group)
        self.provider_note_label.setWordWrap(True)
        self.key_status_label = QLabel("", group)
        self.connection_label = QLabel("", group)
        self.connection_label.setWordWrap(True)

        self.set_key_button = QPushButton("Set API key...", group)
        self.test_button = QPushButton("Test connection", group)
        self.set_key_button.clicked.connect(self.set_api_key_interactive)
        self.test_button.clicked.connect(self.test_connection)

        key_row = QHBoxLayout()
        key_row.addWidget(self.key_status_label, 1)
        key_row.addWidget(self.set_key_button)

        actions = QHBoxLayout()
        actions.addWidget(self.test_button)
        actions.addStretch(1)

        form.addRow("", self.translate_check)
        form.addRow("Target language", self.target_language_combo)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Model", self.model_edit)
        form.addRow("Base URL", self.base_url_edit)
        form.addRow(self.provider_note_label)
        form.addRow(key_row)
        form.addRow(actions)
        form.addRow(self.connection_label)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export", self)
        form = QFormLayout(group)

        self.output_edit = QLineEdit(group)
        self.output_edit.setPlaceholderText("Empty: write next to each video")
        self.browse_button = QPushButton("Browse...", group)
        self.browse_button.clicked.connect(self.browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.browse_button)

        self.original_check = QCheckBox("Original subtitles", group)
        self.translated_check = QCheckBox("Translated subtitles", group)
        self.bilingual_check = QCheckBox("Bilingual subtitles", group)
        self.format_combo = QComboBox(group)
        for label, value in FORMAT_CHOICES:
            self.format_combo.addItem(label, value)
        self.overwrite_combo = QComboBox(group)
        for label, value in OVERWRITE_CHOICES:
            self.overwrite_combo.addItem(label, value)

        form.addRow("Output directory", output_row)
        form.addRow("", self.original_check)
        form.addRow("", self.translated_check)
        form.addRow("", self.bilingual_check)
        form.addRow("Format", self.format_combo)
        form.addRow("On conflict", self.overwrite_combo)
        return group

    def _build_processing_group(self) -> QGroupBox:
        group = QGroupBox("Processing", self)

        self.start_button = QPushButton("Start", group)
        self.cancel_button = QPushButton("Cancel", group)
        self.start_button.clicked.connect(self.start_job)
        self.cancel_button.clicked.connect(self.cancel_job)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)

        self.progress_bar = QProgressBar(group)
        self.progress_bar.setRange(0, 100)
        self.stage_label = QLabel("Idle", group)
        self.message_label = QLabel("", group)
        self.message_label.setWordWrap(True)

        self.log_pane = QPlainTextEdit(group)
        self.log_pane.setReadOnly(True)
        self.log_pane.setMaximumBlockCount(_LOG_PANE_MAX_BLOCKS)
        self.log_pane.setFont(_monospace_font())
        self.log_pane.setPlaceholderText("Pipeline log")

        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.message_label)
        layout.addWidget(self.log_pane)
        return group

    # ------------------------------------------------------- config wiring -- #

    def _connect_settings_signals(self) -> None:
        for combo in (
            self.model_combo,
            self.device_combo,
            self.compute_combo,
            self.source_language_combo,
            self.target_language_combo,
            self.format_combo,
            self.overwrite_combo,
        ):
            combo.currentIndexChanged.connect(self._on_setting_changed)
        for check in (
            self.translate_check,
            self.vad_check,
            self.words_check,
            self.original_check,
            self.translated_check,
            self.bilingual_check,
        ):
            check.toggled.connect(self._on_setting_changed)
        for edit in (self.model_edit, self.base_url_edit, self.output_edit):
            edit.textChanged.connect(self._on_setting_changed)
        self.batch_spin.valueChanged.connect(self._on_setting_changed)
        # Switching provider must carry the preset endpoint and model across: the
        # dataclass defaults describe DeepSeek, so this is the classic foot-gun.
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)

    def _on_setting_changed(self, *_args: object) -> None:
        if self._loading:
            return
        self._collect_settings()

    def _on_provider_changed(self, provider: str) -> None:
        if self._loading:
            return
        preset = PROVIDER_PRESETS.get(provider)
        if preset is not None:
            self.model_edit.setText(preset.model)
            self.base_url_edit.setText(preset.base_url)
            self.provider_note_label.setText(preset.note)
        self.refresh_key_status()
        self._collect_settings()

    def _apply_config_to_widgets(self) -> None:
        asr, translation, export = self.config.asr, self.config.translation, self.config.export

        self.model_combo.setCurrentText(asr.model)
        self.device_combo.setCurrentText(asr.device)
        self.compute_combo.setCurrentText(asr.compute_type)
        index = self.source_language_combo.findData(asr.language)
        self.source_language_combo.setCurrentIndex(max(0, index))
        self.vad_check.setChecked(asr.vad_filter)
        self.words_check.setChecked(asr.word_timestamps)
        self.batch_spin.setValue(int(asr.batch_size))

        self.translate_check.setChecked(translation.enabled)
        self.target_language_combo.setCurrentText(translation.target_language)
        provider = translation.provider if translation.provider in PROVIDER_PRESETS else "deepseek"
        self.provider_combo.setCurrentText(provider)
        # Use the persisted endpoint/model rather than the preset: a user may have
        # pointed a preset at a proxy, and that choice has to survive a restart.
        self.model_edit.setText(translation.model)
        self.base_url_edit.setText(translation.base_url)
        self.provider_note_label.setText(PROVIDER_PRESETS[provider].note)

        self.output_edit.setText(str(export.output_dir) if export.output_dir else "")
        self.original_check.setChecked(export.write_original)
        self.translated_check.setChecked(export.write_translated)
        self.bilingual_check.setChecked(export.write_bilingual)
        self.format_combo.setCurrentIndex(
            max(0, self.format_combo.findData(export.format.value))
        )
        self.overwrite_combo.setCurrentIndex(
            max(0, self.overwrite_combo.findData(export.overwrite.value))
        )
        self.refresh_key_status()

    def _collect_settings(self) -> None:
        """Copy the widgets back into ``self.config`` (called on every change)."""
        asr, translation, export = self.config.asr, self.config.translation, self.config.export

        asr.model = self.model_combo.currentText()
        asr.device = self.device_combo.currentText()
        asr.compute_type = self.compute_combo.currentText()
        asr.language = self.source_language_combo.currentData()
        asr.vad_filter = self.vad_check.isChecked()
        asr.word_timestamps = self.words_check.isChecked()
        asr.batch_size = int(self.batch_spin.value())

        translation.enabled = self.translate_check.isChecked()
        translation.target_language = self.target_language_combo.currentText()
        translation.provider = self.provider_combo.currentText()
        translation.model = self.model_edit.text().strip()
        translation.base_url = self.base_url_edit.text().strip()

        output = self.output_edit.text().strip()
        export.output_dir = Path(output) if output else None
        export.write_original = self.original_check.isChecked()
        export.write_translated = self.translated_check.isChecked()
        export.write_bilingual = self.bilingual_check.isChecked()
        export.format = SubtitleFormat(self.format_combo.currentData() or "srt")
        export.overwrite = OverwritePolicy(self.overwrite_combo.currentData() or "ask")

    def current_specs(self) -> list[JobSpec]:
        """The jobs the current widgets describe (one per listed file)."""
        self._collect_settings()
        return [
            JobSpec(
                media=path,
                asr=self.config.asr,
                translation=self.config.translation,
                export=self.config.export,
            )
            for path in self._files
        ]

    # ------------------------------------------------------------ files ---- #

    def add_paths(self, paths: Iterable[Path]) -> list[Path]:
        """Add files and one level of folder contents; return the paths actually added.

        Unsupported files, duplicates and missing paths are ignored so a drag of a
        whole folder never aborts on a stray text file.
        """
        candidates = [Path(path) for path in paths if str(path).strip()]
        if not candidates:
            return []
        try:
            expanded = expand_inputs(candidates)
        except MediaError as exc:
            self._append_log(f"Input problem: {exc}")
            return []

        added: list[Path] = []
        for path in expanded:
            key = _path_key(path)
            if key in self._row_of:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._files.append(path)
            self._row_of[key] = row
            self._set_cell(row, 0, path.name, tooltip=str(path))
            self._set_cell(row, 1, _PENDING_DURATION)
            self._set_cell(row, 2, JobStatus.PENDING.value.capitalize())
            self._set_cell(row, 3, "")
            self._probe_queue.append(row)
            added.append(path)

        if added:
            self._append_log(f"Added {len(added)} file(s).")
        self._schedule_probe()
        self._update_buttons()
        return added

    def add_files(self) -> list[Path]:
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXT))
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Add media files",
            self.config.last_input_dir or "",
            f"Media files ({patterns});;All files (*)",
        )
        if not selected:
            return []
        self.config.last_input_dir = str(Path(selected[0]).parent)
        return self.add_paths(Path(path) for path in selected)

    def add_folder(self) -> list[Path]:
        directory = QFileDialog.getExistingDirectory(
            self, "Add a folder of media", self.config.last_input_dir or ""
        )
        if not directory:
            return []
        self.config.last_input_dir = directory
        return self.add_paths([Path(directory)])

    def remove_selected(self) -> None:
        if self._running:
            return
        selection = self.table.selectionModel()
        rows = {index.row() for index in selection.selectedRows()}
        if not rows:
            rows = {index.row() for index in selection.selectedIndexes()}
        self._remove_rows(rows)

    def clear_files(self) -> None:
        if self._running:
            return
        self._probe_queue.clear()
        self.table.setRowCount(0)
        self._files.clear()
        self._row_of.clear()
        self._update_buttons()

    def _remove_rows(self, rows: Iterable[int]) -> None:
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self._files):
                self.table.removeRow(row)
                del self._files[row]
        self._probe_queue.clear()
        self._row_of = {_path_key(path): index for index, path in enumerate(self._files)}
        for row in range(len(self._files)):
            if self.table.item(row, 1) is not None and self.table.item(row, 1).text() == _PENDING_DURATION:
                self._probe_queue.append(row)
        if self._probe_queue:
            self._schedule_probe()
        self._update_buttons()

    def _set_cell(self, row: int, column: int, text: str, tooltip: str | None = None) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, column, item)
        item.setText(text)
        if tooltip is not None:
            item.setToolTip(tooltip)

    # -- drag and drop ------------------------------------------------------ #

    def _drop_targets(self, event: QDragEnterEvent | QDragMoveEvent | QDropEvent) -> list[Path]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        found: list[Path] = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir() or (path.is_file() and path.suffix.lower() in SUPPORTED_EXT):
                found.append(path)
        return found

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt naming
        if not self._running and self._drop_targets(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802 - Qt naming
        if not self._running and self._drop_targets(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt naming
        targets = self._drop_targets(event) if not self._running else []
        if targets:
            event.acceptProposedAction()
            self.add_paths(targets)
        else:
            event.ignore()

    # -- asynchronous probing ----------------------------------------------- #

    def _schedule_probe(self) -> None:
        if self._probe_queue and not self._probing:
            self._probing = True
            QTimer.singleShot(0, self._probe_next)

    def _probe_next(self) -> None:
        """Fill in one duration per timer tick so the window keeps repainting."""
        if self._probe_queue:
            row = self._probe_queue.pop(0)
            if 0 <= row < len(self._files):
                path = self._files[row]
                try:
                    info = probe_media(path)
                except MediaError as exc:
                    self._set_cell(row, 1, "unavailable", tooltip=str(exc))
                    self._append_log(f"{path.name}: {exc}")
                except Exception as exc:  # defensive: never let a probe kill the window
                    self._set_cell(row, 1, "unavailable", tooltip=str(exc))
                    self._append_log(f"{path.name}: unexpected probe failure: {exc}")
                else:
                    self._set_cell(row, 1, format_duration(info.duration_s))
        if self._probe_queue:
            QTimer.singleShot(0, self._probe_next)
        else:
            self._probing = False

    # --------------------------------------------------------- credentials -- #

    def refresh_key_status(self) -> None:
        provider = self.provider_combo.currentText()
        try:
            key = get_api_key(provider)
        except Exception as exc:  # a locked credential store must not break the UI
            self.key_status_label.setText(f"API key: unavailable ({exc})")
            return
        self.key_status_label.setText(f"API key: {mask_secret(key) if key else 'not set'}")

    def set_api_key_interactive(self) -> None:
        provider = self.provider_combo.currentText()
        key, accepted = QInputDialog.getText(
            self,
            f"{provider} API key",
            f"Paste the API key for {provider}:",
            QLineEdit.EchoMode.Password,
        )
        if not accepted or not key.strip():
            self._append_log("No API key entered; nothing stored.")
            return
        try:
            set_api_key(provider, key.strip())
        except Exception as exc:
            self._append_log(f"Could not store the {provider} key: {exc}")
            return
        self._append_log(f"Stored {provider} key {mask_secret(key.strip())}")
        self.refresh_key_status()

    def test_connection(self) -> None:
        self._collect_settings()
        if self._tester is not None and self._tester.isRunning():
            self._append_log("A provider check is already running.")
            return
        self.connection_label.setText("Testing connection...")
        self._append_log(f"Testing {self.config.translation.provider} ...")
        tester = _ProviderTester(replace(self.config.translation), self)
        self._tester = tester  # keep a reference: Qt does not own Python threads
        tester.checked.connect(self._on_connection_checked)
        tester.start()

    def _on_connection_checked(self, ok: bool, message: str) -> None:
        self.connection_label.setText("Connection OK" if ok else "Connection failed")
        self._append_log(f"Provider check {'OK' if ok else 'FAILED'}: {message}")

    # ----------------------------------------------------------- processing -- #

    def start_job(self) -> None:
        """Collect the settings and start a run (also the test entry point)."""
        if self._running:
            return
        specs = self.current_specs()
        if not specs:
            self._append_log("No media files to process.")
            return
        try:
            save_config(self.config)  # never stores an API key: keys live in keyring
        except Exception as exc:
            self._append_log(f"Could not save the settings: {exc}")

        self._reset_progress()
        for row in range(len(self._files)):
            self._set_cell(row, 2, JobStatus.PENDING.value.capitalize())
            self._set_cell(row, 3, "")
        self._set_running(True)
        self._append_log(f"Starting run: {len(specs)} file(s).")

        worker = PipelineWorker(
            self.config, specs, pipeline_factory=self._pipeline_factory
        )
        worker.progress.connect(self._on_progress)
        worker.conflict.connect(self._on_conflict)
        worker.log_line.connect(self._append_log)
        worker.finished_run.connect(self._on_finished_run)
        self.worker = worker
        worker.start()

    def cancel_job(self) -> None:
        worker = self.worker
        if worker is None or not self._running:
            return
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self._append_log("Cancellation requested; finishing the current step.")
        worker.cancel()

    def _reset_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting...")
        self.message_label.setText("")

    def _set_running(self, running: bool) -> None:
        self._running = running
        for group in (
            self._files_group,
            self._recognition_group,
            self._translation_group,
            self._export_group,
        ):
            group.setEnabled(not running)
        self.start_button.setEnabled(not running and bool(self._files))
        self.cancel_button.setEnabled(running)
        if running:
            self.cancel_button.setText("Cancel")

    def _update_buttons(self) -> None:
        has_files = bool(self._files)
        self.start_button.setEnabled(has_files and not self._running)
        self.remove_button.setEnabled(has_files and not self._running)
        self.clear_button.setEnabled(has_files and not self._running)

    # -- worker signals ----------------------------------------------------- #

    def _on_progress(self, event: ProgressEvent) -> None:
        self.stage_label.setText(f"{event.stage.label} ({event.file_index}/{event.file_total})")
        self.message_label.setText(event.message)
        if event.fraction is not None:
            total = max(1, event.file_total)
            overall = (max(1, event.file_index) - 1 + event.fraction) / total
            self.progress_bar.setValue(int(round(min(1.0, max(0.0, overall)) * 100)))
        row = event.file_index - 1
        if 0 <= row < len(self._files):
            self._set_cell(row, 2, JobStatus.RUNNING.value.capitalize())

    def _on_conflict(self, request: ConflictRequest) -> None:
        """Ask the user in the GUI thread; the worker is blocked until we answer."""
        names = ", ".join(path.name for path in request.conflicts[:5])
        self._append_log(
            f"{request.media.name}: {len(request.conflicts)} subtitle file(s) already exist ({names})"
        )
        try:
            request.policy = self.ask_overwrite_policy(request)
        finally:
            request.event.set()

    def ask_overwrite_policy(self, request: ConflictRequest) -> OverwritePolicy:
        """Show the overwrite dialog. Separate so a test can substitute an answer."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Subtitles already exist")
        names = "\n".join(str(path) for path in request.conflicts[:5])
        box.setText(
            f"{len(request.conflicts)} subtitle file(s) for {request.media.name} already exist:"
        )
        box.setDetailedText(names)
        overwrite = box.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        rename = box.addButton("Rename", QMessageBox.ButtonRole.ActionRole)
        cancel = box.addButton("Cancel run", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(skip)
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite:
            return OverwritePolicy.OVERWRITE
        if clicked is rename:
            return OverwritePolicy.RENAME
        if clicked is cancel:
            self.cancel_job()
            return OverwritePolicy.SKIP
        return OverwritePolicy.SKIP

    def _on_finished_run(self, results: Sequence[JobResult]) -> None:
        output_dir = self.config.export.output_dir
        counts = {status: 0 for status in JobStatus}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
            row = self._row_of.get(_path_key(Path(result.media)))
            if row is None:
                continue
            self._set_cell(row, 2, result.status.value.capitalize())
            self._set_cell(
                row,
                3,
                _describe_output(result, output_dir),
                tooltip=result.error or None,
            )

        summary = (
            f"Run finished: {counts.get(JobStatus.DONE, 0)} done, "
            f"{counts.get(JobStatus.FAILED, 0)} failed, "
            f"{counts.get(JobStatus.CANCELLED, 0)} cancelled, "
            f"{counts.get(JobStatus.SKIPPED, 0)} skipped."
        )
        self._append_log(summary)
        self.statusBar().showMessage(summary)
        if not counts.get(JobStatus.CANCELLED, 0):
            self.progress_bar.setValue(100)
        self.stage_label.setText("Idle")
        self._set_running(False)
        self._load_editor_from(results)

    def _load_editor_from(self, results: Sequence[JobResult]) -> None:
        """Open the first completed job's subtitles in the editor for review."""
        job = next(
            (result for result in results if result.status is JobStatus.DONE and result.written),
            None,
        )
        if job is None:
            return
        written = list(job.written)

        tag = naming.normalize_language_tag(self.config.translation.target_language)

        def pick(suffix: str) -> Path | None:
            for path in written:
                if path.stem.endswith(suffix):
                    return path
            return None

        try:
            media = probe_media(Path(job.media))
        except MediaError as exc:
            self._append_log(f"could not reopen {Path(job.media).name} for editing: {exc}")
            return

        self.editor.set_export_settings(self.config.export, tag)
        if self.editor.load_files(
            media,
            original=pick(".original"),
            translated=pick(f".{tag}"),
            bilingual=pick(".bilingual"),
        ):
            self.tabs.setCurrentWidget(self.editor)
            self.statusBar().showMessage(
                f"Loaded {Path(job.media).name} into the subtitle editor"
            )

    def _on_editor_saved(self, written: Sequence[Path]) -> None:
        for path in written:
            self._append_log(f"editor wrote {path}")

    # ------------------------------------------------------------------ log -- #

    def _append_log(self, text: str) -> None:
        """Append one status line; never logs, so the handler cannot recurse."""
        if text:
            self.log_pane.appendPlainText(str(text))

    # ---------------------------------------------------------------- misc -- #

    def browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose an output folder", self.output_edit.text().strip() or ""
        )
        if directory:
            self.output_edit.setText(directory)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(5000)
        super().closeEvent(event)


def _describe_output(result: JobResult, output_dir: Path | None) -> str:
    """Human-readable result cell: written files, or why nothing was written."""
    if result.written:
        names: list[str] = []
        for path in result.written:
            if output_dir is not None:
                try:
                    names.append(str(path.relative_to(output_dir)))
                    continue
                except ValueError:
                    pass
            names.append(path.name)
        return ", ".join(names)
    if result.skipped:
        return "left existing: " + ", ".join(path.name for path in result.skipped)
    if result.error:
        return result.error
    return ""
