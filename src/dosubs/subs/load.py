"""Load subtitle files back into a :class:`~dosubs.models.Transcript`.

This is what makes editing possible without re-running recognition: the editor opens
the files that were just written (or an older pair), you fix text or timings, and the
exporters render the edited transcript again. It also lets a user reuse subtitles that
were produced elsewhere.

Pairing rules matter here. The translated file is matched to the original by cue
timing, not by position, so a translated file with a different cue count cannot
silently shift every line by one; unmatched cues are reported as warnings and left
untranslated.
"""

from __future__ import annotations

from pathlib import Path

import pysubs2

from ..logging_setup import get_logger
from ..models import Cue, MediaInfo, Transcript
from .io import event_lines, read_subtitles

log = get_logger(__name__)

__all__ = ["transcript_from_subtitles", "split_bilingual_text"]

#: Codepoint ranges treated as CJK when guessing where a bilingual cue splits.
_CJK_RANGES = (
    (0x2E80, 0x2EFF),
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)


def _script_bucket(text: str) -> str:
    """Coarse script class of a line: ``cjk``, ``latin`` or ``other``."""
    cjk = latin = 0
    for char in text:
        if char.isspace() or not char.isalnum():
            continue
        if _is_cjk(char):
            cjk += 1
        elif char.isascii():
            latin += 1
    if cjk and cjk >= latin:
        return "cjk"
    if latin:
        return "latin"
    return "other"


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def split_bilingual_text(text: str) -> tuple[str, str]:
    """Split a bilingual cue into (original, translation).

    The original occupies the first line(s) and the translation the rest, so the split
    is placed at the first line whose script differs from the first line's — which is
    what happens when a long original wraps to two lines above a Chinese translation.
    When both languages share a script the boundary cannot be inferred and the split
    falls back to the first line; loading the original/translated pair instead is
    always exact.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ("", "")
    if len(lines) == 1:
        return (lines[0], "")

    boundary = 1
    first_script = _script_bucket(lines[0])
    for index in range(1, len(lines)):
        if _script_bucket(lines[index]) != first_script:
            boundary = index
            break

    return (" ".join(lines[:boundary]), " ".join(lines[boundary:]))


def _events_to_cues(events: list[pysubs2.SSAEvent]) -> list[Cue]:
    return [
        Cue(
            id=index,
            start_ms=int(event.start),
            end_ms=int(event.end),
            text_original=" ".join(event_lines(event)),
        )
        for index, event in enumerate(events, start=1)
    ]


def transcript_from_subtitles(
    media: MediaInfo,
    *,
    original: Path | None = None,
    translated: Path | None = None,
    bilingual: Path | None = None,
    encoding: str = "utf-8",
) -> Transcript:
    """Build a transcript from one or more subtitle files.

    Exactly one of ``original``/``bilingual`` must be given: the bilingual file alone
    carries both languages, while the original plus translated pair is matched by
    timing.

    Raises:
        ValueError: if no source file was given or a file cannot be parsed.
    """
    if original is None and bilingual is None:
        raise ValueError("give either an original subtitle file or a bilingual one")

    warnings: list[str] = []
    source_path = original if original is not None else bilingual
    assert source_path is not None  # for type checkers
    try:
        subs = read_subtitles(source_path, encoding=encoding)
    except Exception as exc:
        raise ValueError(f"could not read {Path(source_path).name}: {exc}") from exc

    if bilingual is not None and original is None:
        cues: list[Cue] = []
        for index, event in enumerate(subs, start=1):
            original_text, translated_text = split_bilingual_text(
                "\n".join(event_lines(event))
            )
            cues.append(
                Cue(
                    id=index,
                    start_ms=int(event.start),
                    end_ms=int(event.end),
                    text_original=original_text,
                    text_translated=translated_text,
                )
            )
        return Transcript(media=media, cues=cues)

    cues = _events_to_cues(list(subs))

    if translated is not None:
        try:
            translated_subs = read_subtitles(translated, encoding=encoding)
        except Exception as exc:
            raise ValueError(f"could not read {Path(translated).name}: {exc}") from exc

        by_timing = {
            (int(event.start), int(event.end)): " ".join(event_lines(event))
            for event in translated_subs
        }
        for cue in cues:
            match = by_timing.pop((cue.start_ms, cue.end_ms), None)
            if match is None:
                continue
            cue.text_translated = match
        if by_timing:
            warnings.append(
                f"{len(by_timing)} translated cue(s) did not line up with the original timeline "
                "and were ignored"
            )
        unmatched = [cue.id for cue in cues if not cue.is_translated]
        if unmatched:
            warnings.append(f"{len(unmatched)} cue(s) have no matching translation")

    if warnings:
        for warning in warnings:
            log.warning("%s: %s", media.path.name, warning)

    return Transcript(media=media, cues=cues, warnings=warnings)
