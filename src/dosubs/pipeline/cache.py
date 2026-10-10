"""On-disk cache of raw recognition results.

Re-translating a video with a different target language, model or prompt must not
pay for speech recognition again. The cache stores the *raw* recogniser segments
(including word timings) rather than finished cues, so a cached run still goes
through the current cue builder and inherits any segmentation or profile change.

Entries are keyed by media identity (resolved path, size, modification time) plus
every recognition parameter that changes the transcript content. Anything that does
not match exactly is a miss: a stale entry is never silently reused.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import cache_dir
from ..logging_setup import get_logger
from ..models import AsrSegment, AsrSettings, Word

log = get_logger(__name__)

__all__ = [
    "CACHE_VERSION",
    "CachedTranscript",
    "cache_key",
    "cache_path",
    "load_cached_transcript",
    "store_transcript",
    "clear_cache",
]

CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CachedTranscript:
    """A validated cache hit."""

    segments: tuple[AsrSegment, ...]
    language: str | None
    language_probability: float | None
    language_forced: bool
    duration_s: float
    path: Path


def _fingerprint(media: Path) -> dict[str, object]:
    resolved = Path(media).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        # Windows file times are 100 ns ticks; int() keeps the JSON portable.
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_key(media: Path, asr: AsrSettings) -> str:
    """Stable key for (media identity, recognition parameters)."""
    payload = json.dumps(
        {"version": CACHE_VERSION, "media": _fingerprint(media), "asr": asr.cache_key()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def cache_path(media: Path, asr: AsrSettings, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else cache_dir() / "transcripts"
    return base / f"{cache_key(media, asr)}.json.gz"


def _serialise_segment(segment: AsrSegment) -> list[object]:
    return [
        segment.start_ms,
        segment.end_ms,
        segment.text,
        [[w.start_ms, w.end_ms, w.text, round(w.probability, 4)] for w in segment.words],
    ]


def _deserialise_segment(raw: object) -> AsrSegment:
    start, end, text, words = raw  # type: ignore[misc]
    return AsrSegment(
        start_ms=int(start),
        end_ms=int(end),
        text=str(text),
        words=tuple(
            Word(start_ms=int(w[0]), end_ms=int(w[1]), text=str(w[2]), probability=float(w[3]))
            for w in words
        ),
    )


def store_transcript(
    media: Path,
    asr: AsrSettings,
    *,
    segments: list[AsrSegment] | tuple[AsrSegment, ...],
    language: str | None,
    language_probability: float | None,
    language_forced: bool,
    duration_s: float,
    root: Path | None = None,
) -> Path | None:
    """Persist a recognition result. Failures are logged, never raised."""
    target = cache_path(media, asr, root)
    payload = {
        "version": CACHE_VERSION,
        "media": _fingerprint(media),
        "asr": asr.cache_key(),
        "language": language,
        "language_probability": language_probability,
        "language_forced": language_forced,
        "duration_s": duration_s,
        "has_words": any(segment.words for segment in segments),
        "segments": [_serialise_segment(segment) for segment in segments],
    }
    temporary = target.with_name(target.name + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary, target)
    except OSError as exc:  # a cache we cannot write is not a job failure
        log.warning("could not write the transcript cache entry %s: %s", target, exc)
        return None
    log.debug("cached %d segments for %s at %s", len(segments), media.name, target.name)
    return target


def load_cached_transcript(
    media: Path, asr: AsrSettings, *, root: Path | None = None
) -> CachedTranscript | None:
    """Return a validated cache hit, or ``None``.

    Every rejection path (missing file, unreadable gzip, bad JSON, wrong version,
    changed media, different recognition parameters) is a plain miss.
    """
    target = cache_path(media, asr, root)
    if not target.exists():
        return None
    try:
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable transcript cache entry %s: %s", target, exc)
        return None

    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        log.debug("ignoring transcript cache entry with a different version: %s", target.name)
        return None
    if payload.get("media") != _fingerprint(media):
        log.debug("ignoring transcript cache entry for a changed file: %s", target.name)
        return None
    if payload.get("asr") != asr.cache_key():
        log.debug("ignoring transcript cache entry with different settings: %s", target.name)
        return None
    # ``word_timestamps`` is deliberately not part of the key: a cached entry that
    # already carries word timings satisfies a request that does not need them, but
    # a wordless entry cannot satisfy a request that does (cue splitting would
    # silently lose its word boundaries).
    if asr.word_timestamps and not payload.get("has_words"):
        log.debug("ignoring wordless transcript cache entry for a word-timing run: %s", target.name)
        return None

    try:
        segments = tuple(_deserialise_segment(raw) for raw in payload["segments"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        log.warning("ignoring malformed transcript cache entry %s: %s", target, exc)
        return None

    return CachedTranscript(
        segments=segments,
        language=payload.get("language"),
        language_probability=payload.get("language_probability"),
        language_forced=bool(payload.get("language_forced", False)),
        duration_s=float(payload.get("duration_s") or 0.0),
        path=target,
    )


def clear_cache(root: Path | None = None) -> tuple[int, int]:
    """Delete cached transcripts; returns (files removed, bytes reclaimed)."""
    base = Path(root) if root is not None else cache_dir() / "transcripts"
    removed = 0
    freed = 0
    if not base.exists():
        return (0, 0)
    for entry in sorted(base.glob("*.json.gz*")):
        try:
            size = entry.stat().st_size
            entry.unlink()
        except OSError as exc:
            log.warning("could not remove %s: %s", entry, exc)
            continue
        removed += 1
        freed += size
    return (removed, freed)
