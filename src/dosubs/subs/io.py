"""Subtitle rendering and file writing.

Every export renders from the same :class:`~dosubs.models.Transcript`, so the
original, translated and bilingual files share one timeline by construction. Line
wrapping is a presentation concern applied here, never in the cue builder, which is
why the same cue can be two lines in the original file and one line in the
bilingual file while its timestamps stay identical.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pysubs2

from ..logging_setup import get_logger
from ..models import Cue, ExportKind, SubtitleFormat, Transcript
from ..segment.profiles import SegmentProfile, profile_for, wrap_text

log = get_logger(__name__)

__all__ = [
    "render_cue_lines",
    "transcript_to_subs",
    "subtitle_text",
    "write_subtitle_file",
    "read_subtitles",
    "single_line_profile",
    "normalize_event_text",
    "event_lines",
]


def normalize_event_text(text: str) -> str:
    """Turn pysubs2's line separators back into real newlines.

    pysubs2 1.9.0 writes real newlines for SRT but reads them back as the literal
    two-character sequence ``\\N``, so every consumer of a loaded event must undo
    that or multi-line cues come out glued together with a stray backslash.
    """
    return text.replace("\\N", "\n").replace("\\n", "\n")


def event_lines(event: pysubs2.SSAEvent) -> list[str]:
    """The non-empty display lines of a loaded event."""
    return [line.strip() for line in normalize_event_text(event.text).splitlines() if line.strip()]

#: Font used for ASS exports so Simplified Chinese renders correctly out of the box.
ASS_FONT = "Microsoft YaHei"


def single_line_profile(profile: SegmentProfile) -> SegmentProfile:
    """A variant that keeps a language on exactly one line for bilingual files."""
    return replace(profile, max_lines=1, max_chars_per_cue=profile.max_chars_per_line)


def render_cue_lines(
    cue: Cue,
    kind: ExportKind,
    *,
    source_profile: SegmentProfile,
    target_profile: SegmentProfile,
) -> list[str]:
    """Wrap one cue for one output file.

    The bilingual file puts the original on the first line(s) and the translation
    directly underneath, both from the same cue, so the two languages can never
    drift apart.
    """
    original = cue.text_original.strip()
    translated = cue.text_translated.strip()

    if kind is ExportKind.ORIGINAL:
        return wrap_text(original, source_profile) if original else []
    if kind is ExportKind.TRANSLATED:
        return wrap_text(translated, target_profile) if translated else []
    if kind is ExportKind.BILINGUAL:
        lines: list[str] = []
        if original:
            lines.extend(wrap_text(original, single_line_profile(source_profile)))
        if translated:
            lines.extend(wrap_text(translated, single_line_profile(target_profile)))
        return lines
    raise ValueError(f"unknown export kind: {kind!r}")


def transcript_to_subs(
    transcript: Transcript,
    kind: ExportKind,
    *,
    target_language: str = "zh-CN",
    fmt: SubtitleFormat = SubtitleFormat.SRT,
) -> pysubs2.SSAFile:
    """Build an in-memory subtitle file for one export kind."""
    source_profile = profile_for(transcript.source_language)
    target_profile = profile_for(target_language)
    subs = pysubs2.SSAFile()

    if fmt is SubtitleFormat.ASS:
        subs.styles["Default"] = pysubs2.SSAStyle(
            fontname=ASS_FONT, fontsize=48, primarycolor=pysubs2.Color(255, 255, 255, 0)
        )
        subs.info["Title"] = transcript.media.path.stem

    dropped = 0
    for cue in transcript.cues:
        lines = render_cue_lines(
            cue,
            kind,
            source_profile=source_profile,
            target_profile=target_profile,
        )
        if not lines:
            dropped += 1
            continue
        event = pysubs2.SSAEvent(
            start=cue.start_ms, end=cue.end_ms, text="\n".join(lines), type="Dialogue"
        )
        subs.append(event)

    if dropped:
        log.warning(
            "%s export dropped %d cue(s) with no %s text",
            kind.value,
            dropped,
            "original" if kind is ExportKind.ORIGINAL else "translated",
        )
    return subs


def subtitle_text(
    transcript: Transcript,
    kind: ExportKind,
    *,
    target_language: str = "zh-CN",
    fmt: SubtitleFormat = SubtitleFormat.SRT,
    bom: bool = False,
) -> str:
    """Serialise one export to text without touching the filesystem."""
    subs = transcript_to_subs(
        transcript, kind, target_language=target_language, fmt=fmt
    )
    text = subs.to_string(fmt.value)
    return ("\ufeff" + text) if bom else text


def write_subtitle_file(
    transcript: Transcript,
    path: Path,
    kind: ExportKind,
    *,
    target_language: str = "zh-CN",
    fmt: SubtitleFormat = SubtitleFormat.SRT,
    encoding: str = "utf-8",
    bom: bool = False,
) -> Path:
    """Write one export atomically and return the path written.

    Writing to a sibling temporary file and replacing it means an interrupted run
    can never leave a half-written subtitle file behind.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = subtitle_text(
        transcript, kind, target_language=target_language, fmt=fmt, bom=bom
    )
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(text.encode(encoding))
    os.replace(temporary, target)
    log.info("wrote %s (%d cues)", target, len(transcript.cues))
    return target


def read_subtitles(path: Path, *, encoding: str = "utf-8") -> pysubs2.SSAFile:
    """Load a subtitle file (used by verification and by the editor)."""
    return pysubs2.load(str(path), encoding=encoding)
