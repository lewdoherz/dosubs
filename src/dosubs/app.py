"""Desktop entry point: ``dosubs-gui``.

Kept separate from the CLI so a headless install can import :mod:`dosubs` (and the
pipeline) without pulling in PySide6, and so the window can be constructed with a
test-supplied pipeline factory.
"""

from __future__ import annotations

import sys

from . import APP_NAME, __version__
from .config import ConfigError, load_config
from .logging_setup import setup_logging

__all__ = ["WINDOW_SIZE", "WINDOW_TITLE", "main"]

WINDOW_TITLE = f"{APP_NAME} {__version__} - video subtitle transcriber and translator"
WINDOW_SIZE = (1100, 760)


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop app: logging, QApplication, MainWindow, config load/save.

    ``--selfcheck`` writes an environment report to the log folder instead of opening
    the window. A frozen GUI build has no console, so this is how a packaged install
    can be diagnosed: run ``DoSubs.exe --selfcheck``.
    """
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if "--selfcheck" in arguments:
        return _write_selfcheck(deep="--deep" in arguments)

    from PySide6.QtWidgets import QApplication, QMessageBox

    from .ui.main_window import MainWindow, cjk_font

    setup_logging()

    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setFont(cjk_font())  # Chinese/Japanese subtitles must not render as tofu

    try:
        config = load_config()
    except ConfigError as exc:
        QMessageBox.critical(
            None,
            f"{APP_NAME} - configuration problem",
            f"The settings file could not be read:\n\n{exc}\n\n"
            "Fix or remove it, then start DoSubs again.",
        )
        return 1

    window = MainWindow(config)
    window.setWindowTitle(WINDOW_TITLE)
    window.resize(*WINDOW_SIZE)
    window.show()
    return int(app.exec())


def _write_selfcheck(deep: bool = False) -> int:
    """Run the CLI self-check and save its report, since a GUI build has no console.

    Any failure is written into the report as well: PyInstaller's windowed bootloader
    shows unhandled exceptions in a modal dialog, and a diagnostic that hangs waiting
    for a click is worse than useless.
    """
    import contextlib
    import io
    import traceback

    from .cli import _selfcheck
    from .config import logs_dir

    target = logs_dir() / "selfcheck.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exit_code = _selfcheck(deep=deep)
    except Exception:
        buffer.write("\nself-check raised:\n")
        buffer.write(traceback.format_exc())
        exit_code = 1
    target.write_text(buffer.getvalue(), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())