"""Cue segmentation tests built from synthetic ASR segments (no media, no network)."""

from __future__ import annotations

from dosubs.models import AsrSegment, Word
from dosubs.segment import (
    PROFILE_LATIN,
    PROFILE_ZH_CN,
    build_cues,
    required_duration_ms,
    wrap_text,
)


def _segment(
    start_ms: int, end_ms: int, text: str, words: tuple[Word, ...] = ()
) -> AsrSegment:
    return AsrSegment(start_ms=start_ms, end_ms=end_ms, text=text, words=words)


def _word_stream(
    text: str, *, start_ms: int = 0, step_ms: int = 250
) -> tuple[tuple[Word, ...], int]:
    """One back-to-back word per whitespace token; returns the words and their end.

    The concatenated word texts reproduce ``text``, which is what the real engines
    do as well; the helper keeps that property so the builder can trust the words.
    """
    words: list[Word] = []
    cursor = start_ms
    for token in text.split():
        words.append(Word(start_ms=cursor, end_ms=cursor + step_ms, text=token))
        cursor += step_ms
    return tuple(words), cursor


def _joined(cues) -> str:
    """Every cue's text without whitespace, for the no-dialogue-lost checks."""
    return "".join("".join(cue.text_original.split()) for cue in cues)


def _assert_invariants(cues, profile) -> None:
    """Every guarantee the rest of the pipeline relies on.

    Only call this for inputs whose geometry leaves room for the reading speed:
    a cue cannot be given reading time the timeline does not contain.
    """
    assert [cue.id for cue in cues] == list(range(1, len(cues) + 1))
    assert all(0 <= cue.start_ms < cue.end_ms for cue in cues)
    assert all(cue.text_original.strip() for cue in cues)
    assert all(cue.text_translated == "" for cue in cues)
    for earlier, later in zip(cues, cues[1:]):
        assert earlier.start_ms <= later.start_ms  # sorted
        assert earlier.end_ms <= later.start_ms  # non-overlapping
    for cue in cues:
        assert cue.duration_ms <= profile.max_duration_ms
        assert required_duration_ms(cue.text_original, profile) <= cue.duration_ms


# --------------------------------------------------------------------------- #
# Splitting oversized segments
# --------------------------------------------------------------------------- #


def test_106_second_segment_with_words_is_split_into_contiguous_short_cues():
    text = " ".join(["word"] * 424)
    words, end_ms = _word_stream(text)
    assert end_ms == 106_000

    cues = build_cues([_segment(0, end_ms, text, words)], profile=PROFILE_LATIN)

    _assert_invariants(cues, PROFILE_LATIN)
    assert len(cues) > 10
    assert [cue.id for cue in cues] == list(range(1, len(cues) + 1))
    assert all(0 <= cue.start_ms < cue.end_ms for cue in cues)
    assert all(cue.duration_ms <= PROFILE_LATIN.max_duration_ms for cue in cues)
    assert all(cue.words for cue in cues)
    # Back-to-back words make the pieces abut and cover the whole segment.
    assert all(cues[index].end_ms == cues[index + 1].start_ms for index in range(len(cues) - 1))
    assert cues[0].start_ms == 0
    assert cues[-1].end_ms == 106_000
    assert _joined(cues) == "".join(text.split())


def test_39_second_wordless_segment_is_split_at_punctuation():
    text = " ".join(f"This is sentence number {index}." for index in range(1, 13))
    cues = build_cues([_segment(0, 39_000, text)], profile=PROFILE_LATIN)

    _assert_invariants(cues, PROFILE_LATIN)
    assert len(cues) >= 2
    assert all(cue.duration_ms <= PROFILE_LATIN.max_duration_ms for cue in cues)
    assert all(cue.start_ms < cue.end_ms for cue in cues)
    assert all(cue.text_original.endswith(".") for cue in cues)
    assert _joined(cues) == "".join(text.split())


def test_regression_measured_68s_and_106s_segments_become_many_short_cues():
    first_text = " ".join(["word"] * 272)
    first_words, first_end = _word_stream(first_text)
    second_text = " ".join(["word"] * 424)
    second_words, second_end = _word_stream(second_text, start_ms=70_000)
    assert (first_end, second_end) == (68_000, 176_000)

    cues = build_cues(
        [
            _segment(0, first_end, first_text, first_words),
            _segment(70_000, second_end, second_text, second_words),
        ],
        profile=PROFILE_LATIN,
    )

    assert len(cues) > 20
    _assert_invariants(cues, PROFILE_LATIN)
    assert all(cue.duration_ms <= PROFILE_LATIN.max_duration_ms for cue in cues)
    assert [cue.id for cue in cues] == list(range(1, len(cues) + 1))
    assert all(cues[index].end_ms <= cues[index + 1].start_ms for index in range(len(cues) - 1))
    assert _joined(cues) == "".join((first_text + second_text).split())


# --------------------------------------------------------------------------- #
# Reading speed
# --------------------------------------------------------------------------- #


def test_dense_three_second_segment_is_extended_until_it_can_be_read():
    text = (
        "This line of dialogue is far too dense for three seconds of screen time "
        "and it keeps going"
    )
    needed = required_duration_ms(text, PROFILE_LATIN)
    assert needed > 3_000

    cues = build_cues([_segment(0, 3_000, text)], profile=PROFILE_LATIN)

    assert len(cues) == 1
    only = cues[0]
    assert only.start_ms == 0
    assert only.duration_ms >= needed
    assert only.end_ms == needed
    assert only.duration_ms <= PROFILE_LATIN.max_duration_ms


def test_short_cue_extends_into_the_gap_but_stops_before_the_next_cue():
    cues = build_cues(
        [
            _segment(0, 1_000, "One two three four five six seven eight nine ten"),
            _segment(5_000, 7_000, "Later line."),
        ],
        profile=PROFILE_LATIN,
    )

    assert len(cues) == 2
    needed = required_duration_ms(cues[0].text_original, PROFILE_LATIN)
    assert needed == 1_858
    assert cues[0].end_ms == needed
    assert cues[0].end_ms <= cues[1].start_ms - PROFILE_LATIN.min_gap_ms


# --------------------------------------------------------------------------- #
# Merging, clamping, dropping
# --------------------------------------------------------------------------- #


def test_two_short_fragments_fifty_milliseconds_apart_merge_into_one_cue():
    cues = build_cues(
        [_segment(0, 200, "Hi"), _segment(250, 450, "there")],
        profile=PROFILE_LATIN,
        drop_short_ms=100,  # 200 ms fragments are below the 250 ms default
    )

    assert len(cues) == 1
    assert cues[0].id == 1
    assert cues[0].text_original == "Hi there"
    assert cues[0].start_ms == 0
    assert cues[0].end_ms >= 450
    assert cues[0].words == ()


def test_overlapping_input_segments_produce_non_overlapping_cues():
    cues = build_cues(
        [
            _segment(0, 3_000, "First line of speech."),
            _segment(2_000, 5_000, "Second line of speech."),
        ],
        profile=PROFILE_LATIN,
    )

    assert len(cues) == 2
    _assert_invariants(cues, PROFILE_LATIN)
    assert all(cue.start_ms < cue.end_ms for cue in cues)
    assert cues[0].end_ms <= cues[1].start_ms
    assert cues[0].end_ms == 2_000 - PROFILE_LATIN.min_gap_ms


def test_empty_and_very_short_segments_are_dropped():
    cues = build_cues(
        [
            _segment(0, 2_000, "   "),
            _segment(2_000, 2_100, "tiny"),
            _segment(3_000, 5_000, "A real line of dialogue."),
        ],
        profile=PROFILE_LATIN,
    )

    assert [cue.text_original for cue in cues] == ["A real line of dialogue."]
    assert [cue.id for cue in cues] == [1]
    assert all(cue.text_translated == "" for cue in cues)


def test_cjk_cues_never_exceed_eighteen_characters_per_line():
    text = "这是一段很长的中文台词" * 9 + "。"

    cues = build_cues([_segment(0, 20_000, text)], profile=PROFILE_ZH_CN)

    _assert_invariants(cues, PROFILE_ZH_CN)
    assert len(cues) == 3
    for cue in cues:
        lines = wrap_text(cue.text_original, PROFILE_ZH_CN)
        assert lines
        assert all(len(line) <= PROFILE_ZH_CN.max_chars_per_line for line in lines)
        assert "".join(lines) == cue.text_original
        assert cue.duration_ms <= PROFILE_ZH_CN.max_duration_ms
    assert "".join(cue.text_original for cue in cues) == text


# --------------------------------------------------------------------------- #
# Contract guarantees
# --------------------------------------------------------------------------- #


def test_translated_text_is_left_for_the_translator():
    cues = build_cues([_segment(0, 3_000, "Hello there.")], profile=PROFILE_LATIN)
    assert cues[0].text_translated == ""
    assert cues[0].is_translated is False


def test_reading_speed_is_respected_when_the_timeline_leaves_room():
    cues = build_cues(
        [
            _segment(
                0,
                3_000,
                "This line is far too dense for three seconds of screen time and it "
                "keeps going",
            ),
            _segment(9_000, 12_000, "A calmer line that fits comfortably."),
        ],
        profile=PROFILE_LATIN,
    )

    assert len(cues) == 2
    _assert_invariants(cues, PROFILE_LATIN)


def test_many_input_shapes_uphold_the_caller_contract():
    text = " ".join(["word"] * 100)
    words, end_ms = _word_stream(text)
    corpora = [
        ([], PROFILE_LATIN),
        ([_segment(0, 1_000, "A single short line.")], PROFILE_LATIN),
        ([_segment(500, 40_000, text, words)], PROFILE_LATIN),
        ([_segment(0, 40_000, text)], PROFILE_LATIN),  # the same text without word timings
        ([_segment(0, 30_000, "这是一句台词，" * 8)], PROFILE_ZH_CN),
        (
            [
                _segment(0, 2_000, "Opening line of the scene."),
                _segment(2_100, 5_000, "A second, overlapping line of speech."),
                _segment(20_000, 27_000, "This is sentence one. This is sentence two."),
            ],
            PROFILE_LATIN,
        ),
    ]

    for segments, profile in corpora:
        cues = build_cues(list(segments), profile=profile)
        _assert_invariants(cues, profile)
        ordered = sorted(segments, key=lambda segment: (segment.start_ms, segment.end_ms))
        expected = "".join("".join(segment.text.split()) for segment in ordered)
        assert _joined(cues) == expected


def test_build_cues_does_not_modify_the_input_iterable():
    segments = [
        _segment(0, 3_000, "  Hello   there  "),
        _segment(3_000, 8_000, "General  Kenobi."),
    ]
    before = list(segments)

    cues = build_cues(segments, profile=PROFILE_LATIN)

    assert segments == before
    assert [cue.text_original for cue in cues] == ["Hello there", "General Kenobi."]


def test_idless_output_is_not_possible_with_an_empty_input():
    assert build_cues([], profile=PROFILE_ZH_CN) == []
