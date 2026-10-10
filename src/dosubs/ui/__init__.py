"""Desktop (PySide6) front end for DoSubs.

The GUI is a thin shell around the same pipeline the CLI drives: the window only
collects settings, and a :class:`~dosubs.ui.worker.PipelineWorker` runs the
pipeline on its own thread so recognition never freezes the window.
"""

from __future__ import annotations

from .main_window import MainWindow
from .worker import ConflictRequest, LogEmitter, PipelineWorker

__all__ = ["ConflictRequest", "LogEmitter", "MainWindow", "PipelineWorker"]
