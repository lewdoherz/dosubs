"""Media discovery and probing: FFmpeg locations plus read-only ffprobe metadata.

Nothing in this package decodes or writes media; it only answers "what is this
file and can we hand it to the recogniser?".
"""

from __future__ import annotations

from .probe import (
    SUPPORTED_AUDIO_EXT,
    SUPPORTED_EXT,
    SUPPORTED_VIDEO_EXT,
    MediaError,
    expand_inputs,
    ffmpeg_path,
    ffprobe_path,
    have_ffmpeg,
    is_supported,
    media_kind,
    probe_media,
)

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
