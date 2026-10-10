"""Machine translation: provider presets, window batching and reply validation."""

from .base import (
    TranslationAuthError,
    TranslationCancelled,
    TranslationError,
    TranslationFormatError,
    TranslationProvider,
    TranslationStats,
)
from .batch import iter_windows, translate_transcript
from .openai_compat import OpenAICompatibleProvider, build_provider
from .validate import (
    ValidationReport,
    assert_valid_window,
    validate_window,
    verify_timeline_unchanged,
)

__all__ = [
    "TranslationError",
    "TranslationAuthError",
    "TranslationFormatError",
    "TranslationCancelled",
    "TranslationStats",
    "TranslationProvider",
    "ValidationReport",
    "validate_window",
    "assert_valid_window",
    "verify_timeline_unchanged",
    "OpenAICompatibleProvider",
    "build_provider",
    "iter_windows",
    "translate_transcript",
]
