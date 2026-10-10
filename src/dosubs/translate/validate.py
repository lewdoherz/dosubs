"""Checks that a provider reply really matches the window we sent.

Both checks are cheap and run on every window: a wrong mapping is the most
common failure mode of chat-completions translation (the model drops a cue,
merges two cues or echoes the untranslated text), and a silently renumbered or
re-timed cue would corrupt every exported subtitle file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Transcript
from .base import TranslationError, TranslationFormatError

__all__ = [
    "ValidationReport",
    "validate_window",
    "assert_valid_window",
    "verify_timeline_unchanged",
]


@dataclass(slots=True)
class ValidationReport:
    """Everything wrong with one window reply; empty lists mean the window is fine."""

    missing: list[int] = field(default_factory=list)
    empty: list[int] = field(default_factory=list)
    duplicated: dict[str, list[int]] = field(default_factory=dict)
    unexpected: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.empty or self.duplicated or self.unexpected)

    def describe(self) -> str:
        """One-line summary used as the exception text and by the caller's log."""
        problems: list[str] = []
        if self.missing:
            problems.append(f"missing ids {self.missing}")
        if self.empty:
            problems.append(f"empty translations for ids {self.empty}")
        if self.duplicated:
            detail = ", ".join(
                f"{ids} share {text!r}" for text, ids in sorted(self.duplicated.items())
            )
            problems.append(f"duplicated text ({detail})")
        if self.unexpected:
            problems.append(f"unexpected ids {self.unexpected}")
        return "invalid translation window: " + "; ".join(problems)


def validate_window(
    requested: list[tuple[int, str]], result: dict[int, str]
) -> ValidationReport:
    """Compare a reply against the cues we asked for, without raising."""
    requested_ids = [cue_id for cue_id, _ in requested]
    wanted = set(requested_ids)
    report = ValidationReport()

    report.missing = [cue_id for cue_id in requested_ids if cue_id not in result]
    report.unexpected = sorted(cue_id for cue_id in result if cue_id not in wanted)

    groups: dict[str, list[int]] = {}
    for cue_id in requested_ids:
        if cue_id not in result:
            continue
        text = result[cue_id]
        stripped = text.strip() if isinstance(text, str) else ""
        if not stripped:
            report.empty.append(cue_id)
        else:
            groups.setdefault(stripped, []).append(cue_id)
    report.duplicated = {text: ids for text, ids in groups.items() if len(ids) > 1}
    return report


def assert_valid_window(requested: list[tuple[int, str]], result: dict[int, str]) -> None:
    """Raise :class:`TranslationFormatError` unless the window maps back 1:1."""
    report = validate_window(requested, result)
    if not report.ok:
        raise TranslationFormatError(report.describe())


def verify_timeline_unchanged(before: Transcript, after: Transcript) -> None:
    """Last line of defence: translation must never move or renumber a cue."""
    if len(before.cues) != len(after.cues):
        raise TranslationError(
            f"cue count changed during translation: {len(before.cues)} -> {len(after.cues)}"
        )
    for index, (old, new) in enumerate(zip(before.cues, after.cues)):
        if old.id != new.id:
            raise TranslationError(
                f"cue at position {index} changed id {old.id} -> {new.id}"
            )
        if old.start_ms != new.start_ms or old.end_ms != new.end_ms:
            raise TranslationError(
                f"cue {old.id} moved from {old.start_ms}-{old.end_ms} ms "
                f"to {new.start_ms}-{new.end_ms} ms"
            )
