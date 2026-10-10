"""Window batching, retries and splitting for transcript translation.

A window is a run of consecutive cues small enough for one request to stay
reliable. When a window keeps failing we split it instead of hammering the same
prompt: a wrong mapping is usually the model losing track of ids as the window
grows, and halves (then single cues) recover the window instead of losing the
whole job. Every cue therefore ends up translated exactly once - never dropped,
never duplicated.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence

from ..models import Cue, Transcript, TranslationSettings
from .base import (
    TranslationAuthError,
    TranslationCancelled,
    TranslationError,
    TranslationProvider,
    TranslationStats,
)
from .validate import assert_valid_window, verify_timeline_unchanged

__all__ = ["iter_windows", "translate_transcript"]

log = logging.getLogger(__name__)

#: First backoff after a failed window; doubles per attempt, then saturates.
_BASE_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 8.0
#: Cancel requests are honoured during a backoff at this granularity.
_SLEEP_SLICE_S = 0.25


def iter_windows(cues: Sequence[Cue], window_chars: int) -> Iterator[list[Cue]]:
    """Group consecutive cues into windows of at most ``window_chars`` characters.

    Cue text is never split, so ids stay 1:1 with the transcript; a cue longer
    than ``window_chars`` becomes a window of its own.
    """
    window: list[Cue] = []
    size = 0
    for cue in cues:
        length = len(cue.text_original)
        if window and size + length > window_chars:
            yield window
            window = []
            size = 0
        window.append(cue)
        size += length
    if window:
        yield window


def translate_transcript(
    transcript: Transcript,
    provider: TranslationProvider,
    *,
    settings: TranslationSettings,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_window_error: Callable[[int, Exception], None] | None = None,
) -> TranslationStats:
    """Translate every cue of ``transcript`` in place and report what it cost.

    Args:
        settings: :class:`~dosubs.models.TranslationSettings`; uses
            ``window_chars``, ``max_retries`` and ``target_language``.
        progress: called as ``progress(done_cues, total_cues)`` after each window.
        should_cancel: polled between windows and between retries; ``True`` stops
            the run with :class:`TranslationCancelled`.
        on_window_error: called as ``on_window_error(window_index, error)`` for
            every failed attempt, for logging by the caller.

    Raises:
        TranslationError: after ``max_retries`` attempts on a single cue.
    """
    before = Transcript(media=transcript.media, cues=[cue.copy() for cue in transcript.cues])
    stats = TranslationStats()
    total = len(transcript.cues)
    max_attempts = max(1, settings.max_retries)
    split_after = max(1, max_attempts // 2)
    done = 0

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise TranslationCancelled("translation cancelled by the caller")

    def sleep_backoff(attempt: int) -> None:
        delay = min(_BASE_BACKOFF_S * 2 ** (attempt - 1), _MAX_BACKOFF_S)
        delay *= random.uniform(0.5, 1.0)  # jitter, never longer than the raw delay
        deadline = time.monotonic() + delay
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, _SLEEP_SLICE_S))
            check_cancel()

    def translate_block(block: list[Cue], window_index: int) -> None:
        nonlocal done
        last_error: TranslationError | None = None
        for attempt in range(1, max_attempts + 1):
            check_cancel()
            items = [(cue.id, cue.text_original) for cue in block]
            try:
                result = provider.translate_window(
                    items,
                    source_language=transcript.source_language,
                    target_language=settings.target_language,
                )
                assert_valid_window(items, result)
            except (TranslationCancelled, TranslationAuthError):
                raise
            except TranslationError as exc:
                last_error = exc
                stats.retries += 1
                if on_window_error is not None:
                    on_window_error(window_index, exc)
                log.warning(
                    "%s: window %d (%d cues) attempt %d/%d failed: %s",
                    provider.name,
                    window_index,
                    len(block),
                    attempt,
                    max_attempts,
                    exc,
                )
                if len(block) > 1 and attempt >= split_after:
                    middle = len(block) // 2
                    log.info(
                        "%s: splitting window %d into %d + %d cues",
                        provider.name,
                        window_index,
                        middle,
                        len(block) - middle,
                    )
                    translate_block(block[:middle], window_index)
                    translate_block(block[middle:], window_index)
                    return
                if attempt == max_attempts:
                    break
                sleep_backoff(attempt)
                continue

            stats.windows += 1
            stats.cues += len(block)
            stats.input_chars += sum(len(text) for _, text in items)
            stats.output_chars += sum(len(result[cue.id]) for cue in block)
            stats.merge_usage(getattr(provider, "last_usage", None))
            for cue in block:
                cue.text_translated = result[cue.id]
            done += len(block)
            if progress is not None:
                progress(done, total)
            return

        ids = [cue.id for cue in block]
        detail = f"cue {ids[0]}" if len(ids) == 1 else f"cues {ids}"
        raise TranslationError(
            f"translation failed for {detail} after {max_attempts} attempts: {last_error}"
        )

    for window_index, window in enumerate(iter_windows(transcript.cues, settings.window_chars)):
        check_cancel()
        translate_block(window, window_index)

    verify_timeline_unchanged(before, transcript)
    log.info(
        "%s: translated %d/%d cues in %d windows (%d retries, %d -> %d chars)%s",
        provider.name,
        stats.cues,
        total,
        stats.windows,
        stats.retries,
        stats.input_chars,
        stats.output_chars,
        f", usage={stats.usage}" if stats.usage else "",
    )
    return stats
