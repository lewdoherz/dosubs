"""Unit tests for the per-language segmentation profiles (no media, no network)."""

from __future__ import annotations

import math

from dosubs.segment import (
    PROFILE_LATIN,
    PROFILE_ZH_CN,
    profile_for,
    required_duration_ms,
    wrap_text,
)

#: Exactly 200 characters, the length the profile numbers are calibrated against.
_LATIN_200 = (
    "Subtitle segmentation is mostly about giving the reader enough time and "
    "enough room on the screen to take in what the speaker has just said before "
    "the next line of spoken dialogue quietly replaces it."
)

#: Exactly 100 CJK characters with no spaces at all.
_CJK_100 = "这是一段很长的中文台词" * 9 + "。"


def test_profile_for_maps_every_zh_tag_to_the_chinese_profile():
    assert profile_for("zh") is PROFILE_ZH_CN
    assert profile_for("zh-CN") is PROFILE_ZH_CN
    assert profile_for("zh-Hans") is PROFILE_ZH_CN
    assert profile_for("ZH-hant") is PROFILE_ZH_CN


def test_profile_for_falls_back_to_latin():
    assert profile_for("en") is PROFILE_LATIN
    assert profile_for("de-DE") is PROFILE_LATIN
    assert profile_for(None) is PROFILE_LATIN
    assert profile_for("") is PROFILE_LATIN


def test_required_duration_grows_with_the_text():
    texts = [
        "",
        "Hi.",
        "Hello there.",
        "Hello there, this is a considerably longer line of spoken dialogue.",
    ]
    durations = [required_duration_ms(text, PROFILE_LATIN) for text in texts]
    assert durations == sorted(durations)
    assert durations[0] == 0
    assert durations[-1] > durations[1]


def test_required_duration_uses_the_profile_reading_speed():
    text = "Hello there."
    readable = sum(1 for char in text if not char.isspace())
    assert readable == 11
    assert required_duration_ms(text, PROFILE_LATIN) == math.ceil(
        readable * 1000 / PROFILE_LATIN.max_reading_speed_cps
    )


def test_required_duration_counts_cjk_glyphs_as_single_characters():
    text = "你好世界，我们开始吧。"
    assert len(text) == 11
    assert required_duration_ms(text, PROFILE_ZH_CN) == math.ceil(
        11 * 1000 / PROFILE_ZH_CN.max_reading_speed_cps
    ) == 1000
    # The same glyphs take longer to read than they would in a latin sentence,
    # which is why the Chinese profile is slower.
    assert required_duration_ms(text, PROFILE_ZH_CN) > required_duration_ms(
        text, PROFILE_LATIN
    )


def test_wrap_text_latin_splits_on_spaces_without_losing_words():
    assert len(_LATIN_200) == 200
    lines = wrap_text(_LATIN_200, PROFILE_LATIN)

    assert all(len(line) <= PROFILE_LATIN.max_chars_per_line for line in lines)
    assert " ".join(lines).split() == _LATIN_200.split()
    # 200 characters cannot fit two 42-character lines, so the wrap emits more
    # lines instead of dropping dialogue; the cue splitter keeps cues within budget.
    assert len(lines) <= math.ceil(len(_LATIN_200) / PROFILE_LATIN.max_chars_per_line)
    assert len(lines) > 2


def test_wrap_text_latin_keeps_text_within_the_cue_budget():
    text = "One short line of dialogue."
    lines = wrap_text(text, PROFILE_LATIN)
    assert len(lines) <= PROFILE_LATIN.max_lines
    assert lines == [text]


def test_wrap_text_cjk_breaks_between_characters():
    assert len(_CJK_100) == 100
    lines = wrap_text(_CJK_100, PROFILE_ZH_CN)

    assert all(len(line) <= PROFILE_ZH_CN.max_chars_per_line for line in lines)
    assert all(line for line in lines)
    assert "".join(lines) == _CJK_100


def test_wrap_text_drops_spaces_between_cjk_characters():
    assert wrap_text("你好 世界", PROFILE_ZH_CN) == ["你好世界"]
    assert wrap_text("Hello  world", PROFILE_LATIN) == ["Hello world"]


def test_wrap_text_of_blank_text_is_empty():
    assert wrap_text("   ", PROFILE_LATIN) == []
    assert wrap_text("", PROFILE_ZH_CN) == []
