"""Translation provider contracts: errors, counters and the provider protocol.

A provider only knows how to translate one *window* of cues. Batching, retrying
and splitting live in :mod:`dosubs.translate.batch`, so every endpoint (hosted
API or local Ollama) gets the same resilience without re-implementing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "TranslationError",
    "TranslationAuthError",
    "TranslationFormatError",
    "TranslationCancelled",
    "TranslationStats",
    "TranslationProvider",
]


class TranslationError(RuntimeError):
    """Translation failed in a way that is worth retrying."""


class TranslationAuthError(TranslationError):
    """The endpoint rejected our credentials; retrying cannot help."""


class TranslationFormatError(TranslationError):
    """The endpoint replied, but the reply could not be mapped back to cues."""


class TranslationCancelled(TranslationError):
    """The caller asked us to stop mid-run (user pressed Cancel)."""


@dataclass(slots=True)
class TranslationStats:
    """Counters for one transcript translation, also shown in the job summary."""

    windows: int = 0
    cues: int = 0
    retries: int = 0
    input_chars: int = 0
    output_chars: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    def merge_usage(self, usage: Mapping[str, int] | None) -> None:
        """Add one endpoint's token counters (different providers report different keys)."""
        if not isinstance(usage, Mapping) or not usage:
            return
        for key, value in usage.items():
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


@runtime_checkable
class TranslationProvider(Protocol):
    """What the batching layer needs from a translation backend.

    ``translate_window`` must return exactly one entry per input id; the caller
    validates that and retries/splits when it does not hold. Implementations
    SHOULD expose ``last_usage`` (a ``dict[str, int]``) after each call so
    retries and window counts can report token usage.
    """

    name: str
    model: str

    def translate_window(
        self,
        items: list[tuple[int, str]],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]: ...

    def check_connection(self) -> str:
        """Return a human-readable success message; raise TranslationError otherwise."""
        ...

    def close(self) -> None: ...
