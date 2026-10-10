"""Subtitle output naming and overwrite planning.

Naming follows the specification exactly, e.g. for ``ExampleVideo.mp4`` with
target language ``zh-CN``::

    ExampleVideo.original.srt
    ExampleVideo.zh-CN.srt
    ExampleVideo.bilingual.srt
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import (
    ExportKind,
    ExportPlan,
    ExportTarget,
    OverwritePolicy,
    SubtitleFormat,
)

__all__ = [
    "normalize_language_tag",
    "subtitle_path",
    "plan_exports",
    "free_path",
]

#: BCP-47-ish: primary subtag, optional script and region subtags.
_LANG_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def normalize_language_tag(tag: str) -> str:
    """Validate and canonicalise a language tag such as ``zh-cn`` -> ``zh-CN``.

    Raises:
        ValueError: if the tag could not be used safely in a filename.
    """
    candidate = (tag or "").strip()
    if not candidate or not _LANG_TAG_RE.match(candidate):
        raise ValueError(f"invalid language tag: {tag!r}")
    parts = candidate.split("-")
    normalized = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 2:  # region, e.g. CN
            normalized.append(part.upper())
        elif len(part) == 4:  # script, e.g. Hans
            normalized.append(part.capitalize())
        else:
            normalized.append(part)
    return "-".join(normalized)


def subtitle_path(
    media: Path,
    kind: ExportKind,
    *,
    target_language: str = "zh-CN",
    fmt: SubtitleFormat = SubtitleFormat.SRT,
    output_dir: Path | None = None,
) -> Path:
    """Where one export of ``media`` should be written."""
    if kind is ExportKind.TRANSLATED:
        suffix = normalize_language_tag(target_language)
    else:
        suffix = kind.value
    directory = Path(output_dir) if output_dir is not None else Path(media).parent
    return directory / f"{Path(media).stem}.{suffix}.{fmt.value}"


def free_path(path: Path) -> Path:
    """``a.srt`` -> ``a (2).srt`` when ``a.srt`` already exists."""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free filename near {path}")


def plan_exports(
    media: Path,
    kinds: list[ExportKind] | tuple[ExportKind, ...],
    *,
    target_language: str = "zh-CN",
    fmt: SubtitleFormat = SubtitleFormat.SRT,
    output_dir: Path | None = None,
    policy: OverwritePolicy = OverwritePolicy.ASK,
) -> ExportPlan:
    """Resolve the output files for one job without writing anything.

    ``OverwritePolicy.ASK`` never resolves on its own: existing files are returned
    in :attr:`ExportPlan.conflicts` so the caller (CLI prompt or GUI dialog) can
    decide. ``SKIP`` drops the existing file, ``RENAME`` picks ``name (2).ext`` and
    ``OVERWRITE`` keeps the original name.
    """
    targets: list[ExportTarget] = []
    conflicts: list[Path] = []
    skipped: list[ExportTarget] = []

    seen: set[str] = set()
    for kind in kinds:
        path = subtitle_path(
            media, kind, target_language=target_language, fmt=fmt, output_dir=output_dir
        )
        key = str(path).lower()
        if key in seen:
            raise ValueError(f"two exports would write the same file: {path}")
        seen.add(key)

        target = ExportTarget(kind=kind, path=path)
        if not path.exists():
            targets.append(target)
            continue
        if policy is OverwritePolicy.OVERWRITE:
            targets.append(target)
        elif policy is OverwritePolicy.SKIP:
            skipped.append(target)
        elif policy is OverwritePolicy.RENAME:
            targets.append(ExportTarget(kind=kind, path=free_path(path)))
        else:  # ASK
            conflicts.append(path)

    return ExportPlan(
        targets=tuple(targets), conflicts=tuple(conflicts), skipped=tuple(skipped)
    )
