"""Turn raw recogniser segments into subtitle cues a viewer can actually read.

faster-whisper with VAD returns whatever it heard in one breath: real recordings
measured 39.8 s, 68.7 s and 106.3 s segments, and the batched pipeline merged two
cues into a single 87 s cue. None of that survives as a subtitle, so this module
splits oversized segments, gives dense ones room to be read and re-unites
fragments that belong to the same sentence.

The steps run in a fixed order, each seeing only timings an earlier step already
made valid: normalise -> drop -> split -> extend for reading -> merge -> clamp
overlaps -> number. Words drive splitting whenever they exist, because word
timings are the only boundaries the engine actually observed.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import replace

from ..models import AsrSegment, Cue, Word
from .profiles import SegmentProfile, char_count, normalize_text, required_duration_ms

__all__ = ["build_cues"]

logger = logging.getLogger(__name__)


def build_cues(
    segments: Iterable[AsrSegment],
    *,
    profile: SegmentProfile,
    drop_short_ms: int = 250,
    merge_gap_ms: int = 120,
) -> list[Cue]:
    """Build a readable, monotonic, non-overlapping cue list from raw ASR segments.

    Args:
        segments: raw engine output; it is consumed once and never modified.
        profile: the language's timing and width budget.
        drop_short_ms: segments shorter than this hold no usable dialogue.
        merge_gap_ms: fragments closer together than this are one thought.

    Returns:
        Cues numbered ``1..N`` in timeline order, ``text_translated`` left empty
        for the translator to fill.
    """
    cues: list[Cue] = []
    for segment in _usable_segments(segments, profile, drop_short_ms):
        cues.extend(_split_segment(segment, profile))
    _extend_into_gaps(cues, profile)
    cues = _merge_adjacent(cues, profile, merge_gap_ms)
    _clamp_overlaps(cues, profile)
    for number, cue in enumerate(cues, start=1):
        cue.id = number
    return cues


# --------------------------------------------------------------------------- #
# Step 1 and 2: normalise, drop
# --------------------------------------------------------------------------- #


def _usable_segments(
    segments: Iterable[AsrSegment], profile: SegmentProfile, drop_short_ms: int
) -> list[AsrSegment]:
    """Normalise text, drop empty and very short segments, order the rest by time."""
    usable: list[AsrSegment] = []
    dropped = 0
    for segment in segments:
        text = normalize_text(segment.text, profile)
        start_ms = max(0, segment.start_ms)
        if not text or segment.end_ms - start_ms < drop_short_ms:
            dropped += 1
            continue
        usable.append(replace(segment, start_ms=start_ms, text=text))
    if dropped:
        logger.debug("dropped %d segment(s) without usable dialogue", dropped)
    usable.sort(key=lambda item: (item.start_ms, item.end_ms))
    return usable


# --------------------------------------------------------------------------- #
# Step 3: split
# --------------------------------------------------------------------------- #


def _split_segment(segment: AsrSegment, profile: SegmentProfile) -> list[Cue]:
    """One segment -> one cue, or the shortest split that respects the hard caps."""
    if (
        segment.end_ms - segment.start_ms <= profile.max_duration_ms
        and char_count(segment.text) <= profile.max_chars_per_cue
    ):
        # Cutting cannot create reading time: a segment that is merely *dense* is
        # fixed by extending it into the following gap (step 4), not by chopping.
        return [_cue(segment.start_ms, segment.end_ms, segment.text, segment.words)]
    return _word_split(segment, profile) or _text_split(segment, profile)


def _cue(start_ms: int, end_ms: int, text: str, words: tuple[Word, ...]) -> Cue:
    """A cue fragment; ids are assigned once the timeline is final."""
    start_ms = max(0, start_ms)
    return Cue(
        id=0,
        start_ms=start_ms,
        end_ms=max(end_ms, start_ms + 1),
        text_original=text,
        words=words,
    )


def _word_split(segment: AsrSegment, profile: SegmentProfile) -> list[Cue]:
    """Split on word timings, the only boundaries that were really observed."""
    words = tuple(
        word
        for word in sorted(segment.words, key=lambda item: (item.start_ms, item.end_ms))
        if word.text.strip()
    )
    if not words or _squeeze(_join_words(words, profile)) != _squeeze(segment.text):
        # Engines disagree about what a "word" holds; when the words do not add up
        # to the segment text, trusting them would silently drop dialogue.
        return []
    cues: list[Cue] = []
    groups = _word_groups(words, profile)
    if any(
        len(group) == 1
        and group[0].end_ms - group[0].start_ms > profile.max_duration_ms
        for group in groups
    ):
        # A lone word cannot be split, and when it outlasts the budget the engine's
        # timings are broken anyway: spread the text over the segment instead.
        return []
    previous_end = segment.start_ms
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        start_ms = (
            segment.start_ms
            if index == 0
            else max(previous_end, group[0].start_ms)
        )
        end_ms = max(group[-1].end_ms, start_ms + 1)
        if last and end_ms < segment.end_ms <= start_ms + profile.max_duration_ms:
            # Only the final piece owns the trailing silence inside the segment.
            end_ms = segment.end_ms
        previous_end = end_ms
        cues.append(_cue(start_ms, end_ms, _join_words(group, profile), group))
    return cues


def _word_groups(
    words: tuple[Word, ...], profile: SegmentProfile
) -> list[tuple[Word, ...]]:
    """Pack words until the duration or the character budget would overflow."""
    groups: list[tuple[Word, ...]] = []
    current: list[Word] = []
    chars = 0
    for word in words:
        word_chars = char_count(word.text)
        if current and (
            word.end_ms - current[0].start_ms > profile.max_duration_ms
            or chars + word_chars > profile.max_chars_per_cue
        ):
            groups.append(tuple(current))
            current, chars = [], 0
        current.append(word)
        chars += word_chars
    if current:
        groups.append(tuple(current))
    return groups


def _text_split(segment: AsrSegment, profile: SegmentProfile) -> list[Cue]:
    """Cut the text itself: at punctuation where possible, at the budget where not."""
    text = segment.text
    duration = segment.end_ms - segment.start_ms
    total = char_count(text) or 1
    pieces = max(
        1,
        math.ceil(char_count(text) / profile.max_chars_per_cue),
        math.ceil(duration / profile.max_duration_ms),
    )
    budget = max(1, math.ceil(char_count(text) / pieces))
    cues: list[Cue] = []
    cursor = segment.start_ms
    for begin, stop in _cut_spans(text, profile, budget):
        piece = text[begin:stop].strip()
        if not piece:
            continue
        share = round(duration * char_count(piece) / total)
        end_ms = cursor + min(max(share, 1), profile.max_duration_ms)
        cues.append(_cue(cursor, end_ms, piece, ()))
        cursor = end_ms
    return cues


def _cut_spans(text: str, profile: SegmentProfile, budget: int) -> list[tuple[int, int]]:
    """Character slices of ``text``, each holding at most ``budget`` read characters."""
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        limit = _budget_index(text, start, budget)
        if limit >= len(text):
            spans.append((start, len(text)))
            break
        stop = _best_cut(text, start, limit, profile)
        spans.append((start, stop))
        start = stop
    return spans


def _budget_index(text: str, start: int, budget: int) -> int:
    """Index just past the ``budget``-th read character counting from ``start``."""
    spent = 0
    for index in range(start, len(text)):
        if not text[index].isspace():
            spent += 1
            if spent >= budget:
                return index + 1
    return len(text)


def _best_cut(text: str, start: int, limit: int, profile: SegmentProfile) -> int:
    """Cut after the last sentence end inside the budget, then a soft break, then hard."""
    for marks in (profile.sentence_end_chars, profile.soft_break_chars):
        for index in range(limit - 1, start, -1):
            if text[index] in marks:
                return index + 1
    return limit


# --------------------------------------------------------------------------- #
# Step 4: extend short cues into the following gap
# --------------------------------------------------------------------------- #


def _extend_into_gaps(cues: list[Cue], profile: SegmentProfile) -> None:
    """Give a cue the reading time it lacks, using the silence that follows it."""
    for index, cue in enumerate(cues):
        deficit = required_duration_ms(cue.text_original, profile) - cue.duration_ms
        if deficit <= 0:
            continue
        limit = cue.start_ms + profile.max_duration_ms
        if index + 1 < len(cues):
            limit = min(limit, cues[index + 1].start_ms - profile.min_gap_ms)
        if limit > cue.end_ms:
            cue.end_ms = min(limit, cue.end_ms + deficit)


# --------------------------------------------------------------------------- #
# Step 5: merge fragments
# --------------------------------------------------------------------------- #


def _merge_adjacent(
    cues: list[Cue], profile: SegmentProfile, merge_gap_ms: int
) -> list[Cue]:
    """Re-join cues that a too-short pause or a split boundary tore apart."""
    merged: list[Cue] = []
    for cue in cues:
        if merged and _can_merge(merged[-1], cue, profile, merge_gap_ms):
            previous = merged[-1]
            previous.text_original = _combine(
                previous.text_original, cue.text_original, profile
            )
            previous.end_ms = max(previous.end_ms, cue.end_ms)
            previous.words = previous.words + cue.words
            continue
        merged.append(cue)
    return merged


def _can_merge(
    previous: Cue, cue: Cue, profile: SegmentProfile, merge_gap_ms: int
) -> bool:
    gap = cue.start_ms - previous.end_ms
    if gap < 0 or gap >= merge_gap_ms:
        return False
    text = _combine(previous.text_original, cue.text_original, profile)
    return (
        char_count(text) <= profile.max_chars_per_cue
        and cue.end_ms - previous.start_ms <= profile.max_duration_ms
    )


# --------------------------------------------------------------------------- #
# Step 6: clamp overlaps
# --------------------------------------------------------------------------- #


def _clamp_overlaps(cues: list[Cue], profile: SegmentProfile) -> None:
    """Force monotonic, non-overlapping timings, keeping ``min_gap_ms`` between cues."""
    for index in range(len(cues) - 1):
        current, following = cues[index], cues[index + 1]
        if current.end_ms <= following.start_ms:
            continue
        candidate = following.start_ms - profile.min_gap_ms
        if candidate - current.start_ms >= profile.min_duration_ms:
            current.end_ms = candidate
        else:
            # Not enough room to shorten the first cue: the second one moves instead.
            following.start_ms = current.end_ms
            following.end_ms = max(following.end_ms, following.start_ms + 1)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def _join_words(words: Iterable[Word], profile: SegmentProfile) -> str:
    """Word texts in reading order; CJK needs no separators, latin does."""
    parts = [word.text.strip() for word in words if word.text.strip()]
    joined = "".join(parts) if profile.cjk else " ".join(parts)
    return joined.strip()


def _combine(first: str, second: str, profile: SegmentProfile) -> str:
    """Glue two cue texts the way the script is written."""
    separator = "" if profile.cjk else " "
    return f"{first}{separator}{second}".strip()


def _squeeze(text: str) -> str:
    """Whitespace-free view of a text, used to check that nothing was lost."""
    return "".join(text.split())
