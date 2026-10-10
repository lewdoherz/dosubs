"""Segmentation: raw recogniser segments in, readable subtitle cues out."""

from .cue_builder import build_cues
from .profiles import (
    PROFILE_LATIN,
    PROFILE_ZH_CN,
    SegmentProfile,
    profile_for,
    required_duration_ms,
    wrap_text,
)

__all__ = [
    "PROFILE_LATIN",
    "PROFILE_ZH_CN",
    "SegmentProfile",
    "build_cues",
    "profile_for",
    "required_duration_ms",
    "wrap_text",
]
