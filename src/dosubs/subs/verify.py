"""Post-write verification of exported subtitle files.

The export stage re-reads what it wrote and compares it against the canonical
transcript. A mismatch is reported instead of being written off as success, which is
what keeps the "all three files share one timeline" guarantee honest.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ExportKind, Transcript
from . import timecode
from .io import event_lines, read_subtitles

__all__ = ["check_srt_text", "verify_written_file"]


def check_srt_text(text: str) -> list[str]:
    """Structural checks on SRT text (indices, timecodes, monotonic timing)."""
    problems: list[str] = []
    if text.startswith("\ufeff"):
        text = text[1:]
    if not text.strip():
        return ["file is empty"]

    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    previous_end = -1
    for position, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 2:
            problems.append(f"block {position}: expected an index line and a timecode line")
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            problems.append(f"block {position}: {lines[0].strip()!r} is not a cue index")
            continue
        if index != position:
            problems.append(f"cue {position}: index {index} is out of sequence")

        marker = lines[1]
        if "-->" not in marker:
            problems.append(f"cue {index}: missing '-->' in {marker!r}")
            continue
        start_text, _, end_text = marker.partition("-->")
        try:
            start = timecode.parse_timecode(start_text.strip())
            end = timecode.parse_timecode(end_text.strip())
        except ValueError as exc:
            problems.append(f"cue {index}: {exc}")
            continue
        if end <= start:
            problems.append(f"cue {index}: end {end_text.strip()} is not after start")
        if start < previous_end:
            problems.append(
                f"cue {index}: starts at {start_text.strip()} before previous cue ends at "
                f"{timecode.format_srt(previous_end)}"
            )
        if not "\n".join(lines[2:]).strip():
            problems.append(f"cue {index}: empty text")
        previous_end = max(previous_end, end)

    return problems


def verify_written_file(
    path: Path,
    transcript: Transcript,
    kind: ExportKind,
    *,
    expect_translation: bool = True,
) -> list[str]:
    """Compare a written file against the transcript it came from."""
    problems: list[str] = []
    try:
        subs = read_subtitles(path)
    except Exception as exc:
        return [f"could not parse {path.name}: {exc}"]

    expected = [
        cue
        for cue in transcript.cues
        if (cue.text_original.strip() if kind is ExportKind.ORIGINAL else True)
    ]
    if kind is ExportKind.TRANSLATED and expect_translation:
        expected = [cue for cue in expected if cue.text_translated.strip()]
    if kind is ExportKind.BILINGUAL:
        expected = [cue for cue in expected if cue.text_original.strip() or cue.text_translated.strip()]

    if len(subs) != len(expected):
        problems.append(
            f"{path.name}: {len(subs)} cues written but {len(expected)} cues expected"
        )

    for written, cue in zip(subs, expected):
        if written.start != cue.start_ms or written.end != cue.end_ms:
            problems.append(
                f"{path.name}: cue {cue.id} timing drifted "
                f"({written.start}->{written.end} ms instead of {cue.start_ms}->{cue.end_ms} ms)"
            )
        if kind is not ExportKind.TRANSLATED and cue.text_original.strip():
            lines = event_lines(written)
            first = lines[0] if lines else ""
            if first and not cue.text_original.strip().startswith(first[:4]):
                problems.append(f"{path.name}: cue {cue.id} original text does not match")
        if kind is ExportKind.TRANSLATED and expect_translation:
            if not "".join(event_lines(written)):
                problems.append(f"{path.name}: cue {cue.id} is empty")
    return problems
