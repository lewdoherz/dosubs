"""Tests for the desktop UI: worker thread, file list, settings and run wiring.

The Qt platform is forced offscreen *before* Qt is imported/instantiated, so these
tests run on a build machine with no display. A real pipeline is never constructed:
each test injects a fake ``pipeline_factory`` and asserts observable behaviour -
table cells, buttons, log text - rather than Qt wiring.
"""

from __future__ import annotations

import json
import os

# Must happen before QApplication exists; monkeypatching later is too late for a
# session-scoped Qt app, so the module sets it up front and the fixture re-asserts it.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import re
import threading
import time
from pathlib import Path

import pytest

from dosubs.config import AppConfig
from dosubs.models import (
    PROVIDER_PRESETS,
    JobResult,
    JobStatus,
    OverwritePolicy,
    ProgressEvent,
    Stage,
    SubtitleFormat,
)
from dosubs.pipeline.orchestrator import JobCancelled
from dosubs.ui.main_window import MainWindow
from dosubs.ui.worker import ConflictRequest, PipelineWorker

# --------------------------------------------------------------------------- #
# Fixtures and fakes
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep runs off the real profile: config and keys stay inside tmp_path."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    # Never touch the Windows credential store from a test.
    monkeypatch.setattr("dosubs.ui.main_window.get_api_key", lambda provider: None)


def _make_window(qtbot, factory=None) -> MainWindow:
    window = MainWindow(AppConfig(), pipeline_factory=factory)
    qtbot.addWidget(window)
    return window


def _add_media(window: MainWindow, tmp_path: Path, name: str = "clip.mp4") -> Path:
    """Add one (empty) file that the window accepts as media."""
    media = tmp_path / name
    media.write_bytes(b"")
    assert window.add_paths([media]) == [media]
    return media


class _RecordingPipeline:
    """Stand-in for ``Pipeline``: reports progress twice and returns one DONE job."""

    def __init__(self, config, *, on_progress=None, should_cancel=None, resolve_conflicts=None, **_):
        self.config = config
        self.on_progress = on_progress
        self.should_cancel = should_cancel
        self.resolve_conflicts = resolve_conflicts
        self.specs: list = []

    def run(self, specs):
        self.specs = list(specs)
        total = len(specs)
        self.on_progress(ProgressEvent(Stage.TRANSCRIBE, "recognising speech", 0.4, 1, total))
        self.on_progress(ProgressEvent(Stage.EXPORT, "writing subtitles", 1.0, 1, total))
        media = specs[0].media
        written = media.with_suffix(".original.srt")
        written.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        return [JobResult(media=media, status=JobStatus.DONE, written=[written])]


class _BlockingPipeline:
    """Blocks inside ``run`` until the cancel callback says stop."""

    def __init__(self, config, *, on_progress=None, should_cancel=None, **_):
        self.on_progress = on_progress
        self.should_cancel = should_cancel
        self.cancelled = False

    def run(self, specs):
        self.on_progress(ProgressEvent(Stage.TRANSCRIBE, "recognising speech", 0.2, 1, len(specs)))
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.should_cancel():
                self.cancelled = True
                raise JobCancelled("cancelled by user")
            time.sleep(0.01)
        raise AssertionError("the run was never cancelled")


class _ConflictPipeline:
    """Asks the overwrite question in the middle of a run and records the answer."""

    def __init__(self, config, *, resolve_conflicts=None, **_):
        self.resolve_conflicts = resolve_conflicts
        self.policy: OverwritePolicy | None = None

    def run(self, specs):
        media = specs[0].media
        existing = media.with_suffix(".original.srt")
        existing.write_text("old\n", encoding="utf-8")
        self.policy = self.resolve_conflicts(media, [existing])
        return [JobResult(media=media, status=JobStatus.DONE, written=[existing])]


# --------------------------------------------------------------------------- #
# Construction and presets
# --------------------------------------------------------------------------- #


def test_window_populates_preset_and_gpu_status(qtbot):
    window = _make_window(qtbot)

    preset = PROVIDER_PRESETS["deepseek"]
    assert window.provider_combo.currentText() == "deepseek"
    assert window.model_edit.text() == preset.model
    assert window.base_url_edit.text() == preset.base_url
    assert window.target_language_combo.currentText() == "zh-CN"
    assert window.model_combo.currentText() == "large-v3"

    gpu_text = window.gpu_label.text().strip()
    assert gpu_text, "the acceleration label must never be blank"
    assert "CUDA" in gpu_text or "CPU" in gpu_text
    assert window.key_status_label.text().startswith("API key:")

    assert not window.start_button.isEnabled()  # no media yet


def test_switching_provider_carries_preset_model_and_base_url(qtbot):
    """The foot-gun: dataclass defaults describe DeepSeek, so the preset must win."""
    window = _make_window(qtbot)

    for name in ("openai", "ollama", "deepseek"):
        window.provider_combo.setCurrentText(name)
        preset = PROVIDER_PRESETS[name]
        assert window.model_edit.text() == preset.model
        assert window.base_url_edit.text() == preset.base_url
        assert window.config.translation.provider == name
        assert window.config.translation.model == preset.model
        assert window.config.translation.base_url == preset.base_url


# --------------------------------------------------------------------------- #
# File list
# --------------------------------------------------------------------------- #


def test_add_file_probes_duration_and_dedupes(qtbot, tmp_path, make_video_file):
    window = _make_window(qtbot)
    video = make_video_file("clip.mp4", seconds=2)

    assert window.add_paths([video]) == [video]
    assert window.table.rowCount() == 1
    assert window.table.item(0, 0).text() == "clip.mp4"
    assert window.table.item(0, 2).text() == "Pending"
    assert window.start_button.isEnabled()

    # ffprobe runs off the GUI thread, so wait for the duration to land.
    qtbot.waitUntil(
        lambda: bool(re.fullmatch(r"\d+:\d\d", window.table.item(0, 1).text())),
        timeout=20000,
    )

    assert window.add_paths([video]) == []
    assert window.table.rowCount() == 1


def test_current_specs_reflect_widgets(qtbot, tmp_path):
    window = _make_window(qtbot)
    media = _add_media(window, tmp_path)

    window.translate_check.setChecked(True)
    window.target_language_combo.setCurrentText("ja")
    window.original_check.setChecked(False)
    window.translated_check.setChecked(True)
    window.bilingual_check.setChecked(False)
    window.format_combo.setCurrentText("ass")
    window.overwrite_combo.setCurrentText("rename")

    specs = window.current_specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.media == media
    assert spec.translation.enabled is True
    assert spec.translation.target_language == "ja"
    assert spec.export.write_original is False
    assert spec.export.write_translated is True
    assert spec.export.write_bilingual is False
    assert spec.export.format is SubtitleFormat.ASS
    assert spec.export.overwrite is OverwritePolicy.RENAME
    # The widgets are the source of truth: self.config is updated in place.
    assert window.config.export.write_bilingual is False
    assert window.config.translation.target_language == "ja"


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def test_start_run_updates_row_buttons_and_log(qtbot, tmp_path):
    created: list[_RecordingPipeline] = []

    def factory(config, **kwargs):
        pipeline = _RecordingPipeline(config, **kwargs)
        created.append(pipeline)
        return pipeline

    window = _make_window(qtbot, factory)
    media = _add_media(window, tmp_path)

    window.start_job()
    qtbot.waitUntil(lambda: window.table.item(0, 2).text() == "Done", timeout=20000)

    assert created[0].specs[0].media == media
    assert window.table.item(0, 3).text() == "clip.original.srt"
    assert window.progress_bar.value() == 100
    assert window.message_label.text() == "writing subtitles"

    log_text = window.log_pane.toPlainText()
    assert "Run finished: 1 done" in log_text
    assert "1 done" in window.statusBar().currentMessage()

    assert window.start_button.isEnabled()
    assert not window.cancel_button.isEnabled()

    # The run persisted the settings, and never an API key.
    text = (tmp_path / "appdata" / "DoSubs" / "config.json").read_text("utf-8")
    saved = json.loads(text)
    assert saved["translation"]["model"] == window.model_edit.text()
    assert "api_key" not in text


def test_cancel_marks_row_cancelled_and_reenables_start(qtbot, tmp_path):
    created: list[_BlockingPipeline] = []

    def factory(config, **kwargs):
        pipeline = _BlockingPipeline(config, **kwargs)
        created.append(pipeline)
        return pipeline

    window = _make_window(qtbot, factory)
    _add_media(window, tmp_path)

    window.start_job()
    qtbot.waitUntil(lambda: bool(created), timeout=20000)
    assert not window.start_button.isEnabled()

    window.cancel_job()
    qtbot.waitUntil(lambda: window.table.item(0, 2).text() == "Cancelled", timeout=20000)

    assert created[0].cancelled is True
    assert window.start_button.isEnabled()
    assert not window.cancel_button.isEnabled()


def test_conflict_request_is_answered_without_deadlock(qtbot, tmp_path, monkeypatch):
    created: list[_ConflictPipeline] = []

    def factory(config, **kwargs):
        pipeline = _ConflictPipeline(config, **kwargs)
        created.append(pipeline)
        return pipeline

    window = _make_window(qtbot, factory)
    media = _add_media(window, tmp_path)

    seen: list[ConflictRequest] = []

    def answer(request: ConflictRequest) -> OverwritePolicy:
        seen.append(request)
        return OverwritePolicy.OVERWRITE

    monkeypatch.setattr(window, "ask_overwrite_policy", answer)

    started = time.monotonic()
    window.start_job()
    qtbot.waitUntil(lambda: window.table.item(0, 2).text() == "Done", timeout=20000)
    elapsed = time.monotonic() - started

    # A deadlock would hit the worker's five-minute cap, so finishing fast proves
    # the GUI thread answered the worker's blocked request.
    assert elapsed < 15.0
    assert len(seen) == 1
    assert seen[0].media == media
    assert seen[0].conflicts == [media.with_suffix(".original.srt")]
    assert seen[0].event.is_set()
    assert created[0].policy is OverwritePolicy.OVERWRITE


def test_cancel_releases_a_pending_conflict(qtbot, tmp_path):
    """Cancel while an overwrite prompt is unanswered must release the worker with SKIP."""
    worker = PipelineWorker(AppConfig(), [])
    seen: list[ConflictRequest] = []
    worker.conflict.connect(seen.append)

    outcome: list[OverwritePolicy] = []
    resolver = threading.Thread(
        target=lambda: outcome.append(
            worker._resolve_conflict(tmp_path / "a.mp4", [tmp_path / "a.srt"])
        ),
        daemon=True,
    )
    started = time.monotonic()
    resolver.start()
    qtbot.waitUntil(lambda: bool(seen), timeout=5000)

    worker.cancel()
    resolver.join(5)
    assert not resolver.is_alive()
    assert outcome == [OverwritePolicy.SKIP]
    assert seen[0].event.is_set()
    assert seen[0].policy is OverwritePolicy.SKIP
    assert time.monotonic() - started < 5.0
