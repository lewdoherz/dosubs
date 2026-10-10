"""Logging setup with secret redaction.

Every log record passes through a filter that scrubs API keys, bearer tokens and
anything explicitly registered with :func:`register_secret`, so credentials can
never leak into the console, the log file or the GUI log pane.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path

from .config import logs_dir

__all__ = ["setup_logging", "get_logger", "register_secret", "RedactingFilter", "LOG_FILENAME"]

LOG_FILENAME = "dosubs.log"

#: Patterns that look like credentials regardless of registration.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b(?:Bearer|Api-Key|api_key|api-key)\s*[:=]?\s*[A-Za-z0-9_\-\.]{8,}", re.I),
    re.compile(r"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|token|secret|password)[A-Za-z0-9_]*)(\s*[=:]\s*)([^\s,;\"']{4,})"),
)

_redacted_values: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember a secret so it is replaced wherever it appears in log output."""
    if value and len(value) >= 6:
        _redacted_values.add(value)


class RedactingFilter(logging.Filter):
    """Replace credentials in the rendered message and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact(value) if isinstance(value, str) else value for value in record.args
                )
        return True


def redact(text: str) -> str:
    """Scrub known and pattern-matching secrets from ``text``."""
    result = text
    for value in _redacted_values:
        if value in result:
            result = result.replace(value, "***redacted***")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 3:
            result = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}***redacted***", result)
        else:
            result = pattern.sub("***redacted***", result)
    return result


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: Path | None = None,
    to_stderr: bool = True,
) -> Path | None:
    """Configure the root logger once; returns the log file path, if any."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if isinstance(level, int) else _level_from_name(level))

    # Third-party libraries that log far too much at INFO for a console UI.
    for noisy in ("httpx", "httpcore", "keyring", "keyring.backend", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for handler in list(root.handlers):
        if getattr(handler, "_dosubs", False):
            root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    if to_stderr:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.addFilter(RedactingFilter())
        stream._dosubs = True  # type: ignore[attr-defined]
        root.addHandler(stream)

    target: Path | None
    if log_file is False:
        target = None
    elif log_file is None:
        target = logs_dir() / LOG_FILENAME
    else:
        target = Path(log_file)

    if target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                target, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            rotating.setFormatter(formatter)
            rotating.addFilter(RedactingFilter())
            rotating._dosubs = True  # type: ignore[attr-defined]
            root.addHandler(rotating)
        except OSError:
            target = None
    return target


def _level_from_name(level: str) -> int:
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
