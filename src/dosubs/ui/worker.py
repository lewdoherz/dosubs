"""Background worker that runs the subtitle pipeline off the GUI thread.

Nothing here touches the database, the network or the filesystem: the worker only
owns the thread, the cancel event, the log bridge and the overwrite question, so
the window can stay responsive while speech recognition and translation run.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QObject, QThread, Signal

from ..config import AppConfig
from ..logging_setup import get_logger
from ..models import JobResult, JobSpec, JobStatus, OverwritePolicy

log = get_logger(__name__)

__all__ = [
    "CONFLICT_POLL_INTERVAL_S",
    "CONFLICT_WAIT_TIMEOUT_S",
    "ConflictRequest",
    "LogEmitter",
    "PipelineWorker",
]

#: How long an unanswered overwrite prompt blocks the run before we skip.
CONFLICT_WAIT_TIMEOUT_S = 300.0
#: Cancel is honoured while a conflict prompt is open at this granularity.
CONFLICT_POLL_INTERVAL_S = 0.1


class LogEmitter(logging.Handler):
    """Handler that forwards formatted records to a Qt signal (thread-safe by Qt queuing).

    The root logger is used, so every module's records (the pipeline included) end
    up in the window's log pane without the pipeline knowing a GUI exists.
    """

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__()
        self._sink = sink
        # One GUI line per record: the level and the logger name are what matter,
        # the timestamp would just repeat what the pane already interleaves.
        self.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(self.format(record))
        except Exception:  # a broken sink must never take the run down
            self.handleError(record)


@dataclass
class ConflictRequest:
    """A question the pipeline asks when subtitle files already exist.

    The worker emits this object and waits on :attr:`event`; the GUI thread writes
    :attr:`policy` and then sets the event, which keeps the decision in the GUI
    thread without a Qt object ever crossing threads.
    """

    media: Path
    conflicts: list[Path]
    event: threading.Event = field(default_factory=threading.Event)
    #: Set by the GUI before the event is signalled; SKIP is the safe default.
    policy: OverwritePolicy = OverwritePolicy.SKIP


class PipelineWorker(QThread):
    """Runs ``Pipeline.run`` for a fixed list of specs on its own thread."""

    progress = Signal(object)  # ProgressEvent
    conflict = Signal(object)  # ConflictRequest; the worker then blocks on request.event
    log_line = Signal(str)
    finished_run = Signal(object)  # list[JobResult]

    def __init__(
        self,
        config: AppConfig,
        specs: Sequence[JobSpec],
        *,
        pipeline_factory: Callable[..., object] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._specs = list(specs)
        self._pipeline_factory = pipeline_factory
        self._stop = threading.Event()
        self._pending: ConflictRequest | None = None
        self._emitter = LogEmitter(self.log_line.emit)

    # -- public API --------------------------------------------------------- #

    def cancel(self) -> None:
        """Ask the run to stop, releasing any conflict prompt with ``SKIP``."""
        self._stop.set()
        pending = self._pending
        if pending is not None and not pending.event.is_set():
            pending.policy = OverwritePolicy.SKIP
            pending.event.set()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    # -- pipeline callbacks ------------------------------------------------- #

    def _resolve_conflict(self, media: Path, conflicts: list[Path]) -> OverwritePolicy:
        request = ConflictRequest(
            media=Path(media),
            conflicts=[Path(path) for path in conflicts],
        )
        self._pending = request
        self.conflict.emit(request)
        deadline = time.monotonic() + CONFLICT_WAIT_TIMEOUT_S
        while not request.event.wait(CONFLICT_POLL_INTERVAL_S):
            if self._stop.is_set():
                break
            if time.monotonic() >= deadline:
                self.log_line.emit(
                    "No answer to the overwrite prompt; leaving the existing files alone."
                )
                break
        self._pending = None
        if self._stop.is_set():
            return OverwritePolicy.SKIP
        return request.policy

    # -- thread body -------------------------------------------------------- #

    def run(self) -> None:  # noqa: D401 - QThread entry point
        # Imported here so opening the window never pays for faster-whisper/ctranslate2.
        from ..pipeline.orchestrator import JobCancelled, Pipeline

        root = logging.getLogger()
        root.addHandler(self._emitter)
        self._pending = None
        try:
            factory = self._pipeline_factory or Pipeline
            try:
                pipeline = factory(
                    self._config,
                    on_progress=self.progress.emit,
                    should_cancel=self._stop.is_set,
                    resolve_conflicts=self._resolve_conflict,
                )
            except Exception as exc:
                log.exception("could not build the pipeline")
                self.log_line.emit(f"Could not start the run: {type(exc).__name__}: {exc}")
                self.finished_run.emit(self._failed(f"{type(exc).__name__}: {exc}"))
                return

            self.log_line.emit(f"Processing {len(self._specs)} file(s)")
            try:
                results = list(pipeline.run(self._specs))
            except JobCancelled as exc:
                log.info("run cancelled: %s", exc)
                self.log_line.emit(f"Run cancelled: {exc}")
                results = self._cancelled(str(exc) or "cancelled by user")
            except Exception as exc:  # never let a crash escape into Qt's thread
                log.exception("the pipeline raised while running")
                self.log_line.emit(f"Run failed: {type(exc).__name__}: {exc}")
                results = self._failed(f"{type(exc).__name__}: {exc}")
            self.finished_run.emit(results)
        finally:
            root.removeHandler(self._emitter)
            self._pending = None

    # -- helpers ------------------------------------------------------------ #

    def _failed(self, error: str) -> list[JobResult]:
        return [
            JobResult(media=spec.media, status=JobStatus.FAILED, error=error)
            for spec in self._specs
        ]

    def _cancelled(self, error: str) -> list[JobResult]:
        return [
            JobResult(media=spec.media, status=JobStatus.CANCELLED, error=error)
            for spec in self._specs
        ]
