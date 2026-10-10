"""Locate FFmpeg/ffprobe and read file metadata without decoding.

Every long-running pipeline stage starts from one question: is this file
something the recogniser can eat, and does it actually carry audio? ffprobe
answers both for a few milliseconds of JSON, so the rest of the app never has to
open a media file itself.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ..config import data_dir
from ..models import MediaInfo

__all__ = [
    "SUPPORTED_AUDIO_EXT",
    "SUPPORTED_EXT",
    "SUPPORTED_VIDEO_EXT",
    "MediaError",
    "expand_inputs",
    "ffmpeg_path",
    "ffprobe_path",
    "have_ffmpeg",
    "is_supported",
    "media_kind",
    "probe_media",
]

log = logging.getLogger(__name__)

#: Containers we hand to faster-whisper. Extensions are lowercase and include
#: the leading dot so a plain ``path.suffix.lower()`` comparison is enough.
SUPPORTED_VIDEO_EXT: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}
)
SUPPORTED_AUDIO_EXT: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".m4b"}
)
SUPPORTED_EXT: frozenset[str] = SUPPORTED_VIDEO_EXT | SUPPORTED_AUDIO_EXT

#: A GUI app must never flash a console window when shelling out.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: ffprobe stderr is echoed into our errors; keep it short enough for a dialog.
_STDERR_TAIL = 400

#: name -> absolute path, or None when the binary could not be found anywhere.
_resolved: dict[str, str | None] = {}


class MediaError(RuntimeError):
    """Raised when a file cannot be probed or FFmpeg is missing."""


# --------------------------------------------------------------------------- #
# Binary discovery
# --------------------------------------------------------------------------- #


def _project_root() -> Path:
    """Best guess at the checkout root, used for a bundled ``tools/`` copy."""
    return Path(__file__).resolve().parents[3]


def _local_candidates(name: str) -> list[Path]:
    """Places we ship or drop a private FFmpeg build, in priority order."""
    candidates: list[Path] = []
    bin_dir = data_dir() / "bin"
    for directory in (bin_dir, _project_root() / "tools" / "ffmpeg" / "bin"):
        candidates.append(directory / f"{name}.exe")
        candidates.append(directory / name)
    return candidates


def _resolve_binary(name: str) -> str | None:
    """Find ``ffmpeg``/``ffprobe`` once and remember the answer, miss included."""
    if name in _resolved:
        return _resolved[name]

    found = shutil.which(name)
    if not found:
        found = shutil.which(f"{name}.exe")
    if found:
        _resolved[name] = found
        return found

    for candidate in _local_candidates(name):
        if candidate.is_file():
            _resolved[name] = str(candidate)
            return str(candidate)

    _resolved[name] = None
    return None


def ffmpeg_path() -> str:
    """Absolute path to the ffmpeg executable.

    Raises:
        MediaError: when no copy is on ``PATH``, in the app data dir or bundled.
    """
    found = _resolve_binary("ffmpeg")
    if not found:
        raise MediaError("ffmpeg executable not found (looked on PATH, %LOCALAPPDATA%\\DoSubs\\bin and tools/ffmpeg/bin)")
    return found


def ffprobe_path() -> str:
    """Absolute path to the ffprobe executable.

    Raises:
        MediaError: when no copy is on ``PATH``, in the app data dir or bundled.
    """
    found = _resolve_binary("ffprobe")
    if not found:
        raise MediaError("ffprobe executable not found (looked on PATH, %LOCALAPPDATA%\\DoSubs\\bin and tools/ffmpeg/bin)")
    return found


def have_ffmpeg() -> bool:
    """Whether the full FFmpeg toolchain (ffmpeg *and* ffprobe) is available.

    Never raises: callers use it to gate UI options, so a missing binary is a
    ``False``, not an exception.
    """
    try:
        ffmpeg_path()
        ffprobe_path()
    except MediaError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def is_supported(path: os.PathLike[str] | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT


def media_kind(path: os.PathLike[str] | str) -> str:
    """``"audio"`` only for audio-only extensions, otherwise ``"video"``."""
    return "audio" if Path(path).suffix.lower() in SUPPORTED_AUDIO_EXT else "video"


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #


def _stderr_tail(completed: subprocess.CompletedProcess[str]) -> str:
    text = (completed.stderr or completed.stdout or "").strip()
    text = " ".join(text.split())
    return text[-_STDERR_TAIL:] or "no output"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration(container: dict[str, Any], streams: list[dict[str, Any]]) -> float:
    """Prefer the container duration, fall back to the longest stream."""
    from_format = _as_float(container.get("duration"))
    if from_format:
        return from_format
    stream_durations = [d for d in (_as_float(s.get("duration")) for s in streams) if d]
    return max(stream_durations, default=0.0)


def _first_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    return next((s for s in streams if s.get("codec_type") == codec_type), None)


def probe_media(path: Path) -> MediaInfo:
    """Read duration, streams and container for one media file.

    Raises:
        MediaError: for a missing file, unsupported container, unreadable file or
            any ffprobe failure. A file that simply has no audio stream is *not*
            an error; the caller decides whether silence is fatal.
    """
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"media file not found: {path}")
    if not is_supported(path):
        raise MediaError(f"unsupported media container: {path}")

    try:
        completed = subprocess.run(
            [
                ffprobe_path(),
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaError(f"ffprobe could not read {path}: {exc}") from exc

    if completed.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {_stderr_tail(completed)}")

    try:
        payload = json.loads(completed.stdout or "")
        container = payload["format"]
        streams = payload["streams"]
        if not isinstance(container, dict) or not isinstance(streams, list):
            raise ValueError("unexpected ffprobe JSON shape")
    except (ValueError, KeyError, TypeError) as exc:
        raise MediaError(f"could not parse ffprobe output for {path}: {exc}") from exc

    audio = _first_stream(streams, "audio")
    video = _first_stream(streams, "video")

    duration = _duration(container, streams)
    if duration <= 0:
        raise MediaError(f"ffprobe reported no usable duration for {path}")

    tags = container.get("tags") or {}
    title = tags.get("title") if isinstance(tags, dict) else None

    size_bytes = _as_int(container.get("size"))
    if size_bytes is None:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0

    info = MediaInfo(
        path=path,
        duration_s=duration,
        size_bytes=size_bytes,
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_channels=_as_int(audio.get("channels")) if audio else None,
        audio_sample_rate=_as_int(audio.get("sample_rate")) if audio else None,
        video_codec=video.get("codec_name") if video else None,
        container=container.get("format_name"),
        title=str(title) if title else None,
    )
    log.debug("probed %s: %s", path, info)
    return info


# --------------------------------------------------------------------------- #
# Input expansion
# --------------------------------------------------------------------------- #


def expand_inputs(paths: Iterable[Path]) -> list[Path]:
    """Turn a mixed list of files and folders into a sorted, unique media list.

    Directories are scanned one level deep only: dragging a season folder onto
    the window should not silently recurse into every subfolder.

    Raises:
        MediaError: if any input path does not exist.
    """
    seen: set[str] = set()
    kept: list[Path] = []

    def add(candidate: Path) -> None:
        key = os.path.normcase(str(candidate.resolve())).casefold()
        if key in seen:
            return
        seen.add(key)
        kept.append(candidate)

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise MediaError(f"input path does not exist: {path}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and is_supported(child):
                    add(child)
        elif is_supported(path):
            add(path)

    kept.sort(key=lambda p: os.path.normcase(str(p)))
    return kept
