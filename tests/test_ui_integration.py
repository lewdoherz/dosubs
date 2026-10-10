"""Drive the real desktop window through a real job.

The pipeline tests prove the core; this proves the surface a user actually touches:
files added to the table, Start pressed, a worker thread running the real recogniser
and a real HTTP endpoint, the row and log pane updating, and the finished subtitles
landing in the editor for review.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from dosubs.config import AppConfig  # noqa: E402
from dosubs.models import OverwritePolicy  # noqa: E402
from dosubs.subs.io import read_subtitles  # noqa: E402
from dosubs.ui.main_window import MainWindow  # noqa: E402

pytestmark = pytest.mark.integration

MODEL_ENV = "DOSUBS_TEST_MODEL"


@pytest.fixture()
def isolated_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the developer's real config, cache or logs."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return tmp_path


def test_window_runs_a_job_and_opens_the_editor(
    qtbot,
    real_media: Path,
    stub_server: str,
    isolated_profile: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A conflict dialog would block the run; this job writes into an empty folder.
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Yes
    )

    config = AppConfig()
    config.asr.model = os.environ.get(MODEL_ENV, "tiny")
    config.asr.device = "auto"
    config.translation.provider = "custom"
    config.translation.base_url = stub_server
    config.translation.model = "stub"
    config.translation.target_language = "zh-CN"
    config.export.output_dir = tmp_path / "out"
    config.export.overwrite = OverwritePolicy.OVERWRITE

    window = MainWindow(config)
    qtbot.addWidget(window)

    added = window.add_paths([real_media])
    assert added == [real_media]
    assert window.table.rowCount() == 1

    window.start_job()
    qtbot.waitUntil(
        lambda: window.table.item(0, 2).text() in {"Done", "Failed", "Skipped", "Cancelled"},
        timeout=600_000,
    )

    status = window.table.item(0, 2).text()
    log_text = window.log_pane.toPlainText()
    assert status == "Done", log_text[-2000:]
    assert "Run finished" in log_text

    written = sorted((tmp_path / "out").glob("*.srt"))
    assert [path.name for path in written] == [
        f"{real_media.stem}.bilingual.srt",
        f"{real_media.stem}.original.srt",
        f"{real_media.stem}.zh-CN.srt",
    ]

    timelines = {
        path.name: tuple((event.start, event.end) for event in read_subtitles(path))
        for path in written
    }
    assert len(set(timelines.values())) == 1, timelines
    assert timelines[written[0].name]

    # The finished job is handed to the editor, ready to review and re-export.
    assert window.editor.table.rowCount() == len(timelines[written[0].name])
    assert window.tabs.currentWidget() is window.editor

    # Re-export from the editor writes the same timeline without re-recognising.
    window.editor.table.item(0, 4).setText("编辑器改过的译文。")
    reexported = window.editor.save()
    assert reexported
    translated = next(path for path in reexported if path.name.endswith("zh-CN.srt"))
    assert "编辑器改过的译文。" in translated.read_text(encoding="utf-8")
