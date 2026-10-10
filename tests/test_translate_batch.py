"""Batching, retry, split and cancellation tests against an in-memory provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from dosubs.models import Cue, MediaInfo, Transcript, TranslationSettings
from dosubs.translate import iter_windows, translate_transcript
from dosubs.translate.base import (
    TranslationAuthError,
    TranslationCancelled,
    TranslationError,
)

MEDIA = MediaInfo(path=Path("video.mp4"), duration_s=30.0, size_bytes=4096, has_audio=True)


def make_transcript(count: int = 3, *, text_len: int = 10) -> Transcript:
    """Cues whose ids, timings and texts are easy to assert on."""
    return Transcript(
        media=MEDIA,
        source_language="en",
        cues=[
            Cue(
                id=index,
                start_ms=(index - 1) * 1000,
                end_ms=index * 1000,
                text_original=f"c{index}".ljust(text_len, "."),
            )
            for index in range(1, count + 1)
        ],
    )


def settings(**overrides) -> TranslationSettings:
    values = {"provider": "fake", "target_language": "zh-CN", "window_chars": 1600, "max_retries": 4}
    values.update(overrides)
    return TranslationSettings(**values)


class FakeProvider:
    """In-memory TranslationProvider with programmable failures.

    ``fail_first`` fails the first N calls of each distinct window (the key is
    the tuple of cue ids); ``fail_multi`` / ``permanent`` always fail windows
    that are bigger than one cue / that contain one of the listed ids;
    ``drop_ids_once`` omits ids from the first reply only, which the validator
    must catch.
    """

    name = "fake"
    model = "fake-1"

    def __init__(
        self,
        *,
        fail_first: int = 0,
        fail_multi: bool = False,
        permanent: tuple[int, ...] = (),
        drop_ids_once: tuple[int, ...] = (),
        fail_with: type[TranslationError] = TranslationError,
    ) -> None:
        self.calls: list[list[int]] = []
        self.seen_ids: list[tuple[int, ...]] = []
        self.failures: dict[tuple[int, ...], int] = {}
        self.last_usage: dict[str, int] = {}
        self.fail_first = fail_first
        self.fail_multi = fail_multi
        self.permanent = set(permanent)
        self.drop_ids_once = set(drop_ids_once)
        self.fail_with = fail_with

    def translate_window(
        self,
        items: list[tuple[int, str]],
        *,
        source_language: str | None,
        target_language: str,
    ) -> dict[int, str]:
        ids = tuple(cue_id for cue_id, _ in items)
        self.calls.append([cue_id for cue_id, _ in items])
        self.seen_ids.append(ids)
        self.last_usage = {"prompt_tokens": len(items), "completion_tokens": 2 * len(items)}
        if any(cue_id in self.permanent for cue_id in ids):
            raise self.fail_with(f"permanent failure for ids {list(ids)}")
        if self.fail_multi and len(items) > 1:
            raise self.fail_with(f"this window is too big: {list(ids)}")
        seen = self.failures.get(ids, 0)
        if seen < self.fail_first:
            self.failures[ids] = seen + 1
            raise self.fail_with(f"transient failure for ids {list(ids)}")
        reply = {cue_id: f"[{target_language}]{text}" for cue_id, text in items}
        for cue_id in tuple(self.drop_ids_once):
            if cue_id in reply:
                del reply[cue_id]
                self.drop_ids_once = self.drop_ids_once - {cue_id}
                break
        return reply

    def check_connection(self) -> str:
        return "fake/fake-1 reachable"

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# iter_windows
# --------------------------------------------------------------------------- #


def test_iter_windows_respects_window_chars_without_splitting_cues():
    cues = make_transcript(5, text_len=10).cues
    windows = list(iter_windows(cues, 25))

    assert [[cue.id for cue in window] for window in windows] == [[1, 2], [3, 4], [5]]
    assert all(sum(len(cue.text_original) for cue in window) <= 25 for window in windows)
    assert [cue.id for window in windows for cue in window] == [1, 2, 3, 4, 5]


def test_iter_windows_gives_an_oversized_cue_its_own_window():
    cues = make_transcript(3, text_len=4).cues
    cues[1].text_original = "x" * 100

    windows = list(iter_windows(cues, 20))

    assert [[cue.id for cue in window] for window in windows] == [[1], [2], [3]]


def test_iter_windows_handles_empty_input():
    assert list(iter_windows([], 100)) == []


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


def test_translation_is_one_to_one_and_reports_stats():
    transcript = make_transcript(4)
    provider = FakeProvider()
    reported: list[tuple[int, int]] = []

    stats = translate_transcript(
        transcript,
        provider,
        settings=settings(window_chars=25),
        progress=lambda done, total: reported.append((done, total)),
    )

    assert [cue.text_translated for cue in transcript.cues] == [
        "[zh-CN]c1........",
        "[zh-CN]c2........",
        "[zh-CN]c3........",
        "[zh-CN]c4........",
    ]
    assert [cue.id for cue in transcript.cues] == [1, 2, 3, 4]
    assert stats.windows == 2
    assert stats.cues == 4
    assert stats.retries == 0
    assert stats.input_chars == sum(len(cue.text_original) for cue in transcript.cues)
    assert stats.output_chars == sum(len(cue.text_translated) for cue in transcript.cues)
    assert stats.usage == {"prompt_tokens": 4, "completion_tokens": 8}
    assert reported == [(2, 4), (4, 4)]
    assert provider.seen_ids == [(1, 2), (3, 4)]


def test_empty_transcript_needs_no_provider_calls():
    provider = FakeProvider()
    stats = translate_transcript(Transcript(media=MEDIA), provider, settings=settings())
    assert stats.windows == 0
    assert stats.cues == 0
    assert provider.calls == []


def test_timings_are_untouched():
    transcript = make_transcript(5)
    before = [(cue.id, cue.start_ms, cue.end_ms) for cue in transcript.cues]

    translate_transcript(transcript, FakeProvider(), settings=settings(window_chars=25))

    assert [(cue.id, cue.start_ms, cue.end_ms) for cue in transcript.cues] == before


# --------------------------------------------------------------------------- #
# retries and splitting
# --------------------------------------------------------------------------- #


def test_transient_failure_is_retried():
    transcript = make_transcript(2)
    provider = FakeProvider(fail_first=1)

    stats = translate_transcript(transcript, provider, settings=settings(max_retries=4))

    assert stats.retries == 1
    assert stats.windows == 1
    assert provider.calls == [[1, 2], [1, 2]]
    assert all(cue.is_translated for cue in transcript.cues)


def test_incomplete_reply_is_caught_and_retried():
    transcript = make_transcript(2)
    provider = FakeProvider(drop_ids_once=(2,))

    stats = translate_transcript(transcript, provider, settings=settings(max_retries=4))

    assert stats.retries == 1
    assert provider.calls == [[1, 2], [1, 2]]
    assert all(cue.is_translated for cue in transcript.cues)


def test_failing_window_is_split_down_to_single_cues():
    transcript = make_transcript(3)
    provider = FakeProvider(fail_multi=True)

    stats = translate_transcript(transcript, provider, settings=settings(max_retries=4))

    assert [cue.text_translated for cue in transcript.cues] == [
        "[zh-CN]c1........",
        "[zh-CN]c2........",
        "[zh-CN]c3........",
    ]
    assert stats.retries >= 2
    assert any(len(ids) == 1 for ids in provider.seen_ids)
    assert stats.cues == 3  # counted once per cue, never twice


def test_permanent_single_cue_failure_raises_instead_of_leaving_it_blank():
    transcript = make_transcript(3)
    provider = FakeProvider(permanent=(2,))

    with pytest.raises(TranslationError) as excinfo:
        translate_transcript(transcript, provider, settings=settings(max_retries=2))

    assert "cue 2" in str(excinfo.value)
    assert not transcript.cues[1].is_translated
    assert all(cue.text_translated == "" for cue in transcript.cues if cue.id == 2)


def test_auth_error_is_not_retried():
    transcript = make_transcript(2)
    provider = FakeProvider(fail_first=5, fail_with=TranslationAuthError)

    with pytest.raises(TranslationAuthError):
        translate_transcript(transcript, provider, settings=settings(max_retries=4))

    assert provider.calls == [[1, 2]]


def test_window_error_callback_reports_the_window_index():
    transcript = make_transcript(4)
    provider = FakeProvider(fail_multi=True)
    errors: list[tuple[int, Exception]] = []

    translate_transcript(
        transcript,
        provider,
        settings=settings(window_chars=25, max_retries=4),
        on_window_error=lambda index, exc: errors.append((index, exc)),
    )

    assert errors
    assert {index for index, _ in errors} == {0, 1}
    assert all(isinstance(exc, TranslationError) for _, exc in errors)


# --------------------------------------------------------------------------- #
# cancellation
# --------------------------------------------------------------------------- #


def test_should_cancel_stops_between_windows():
    transcript = make_transcript(3)
    provider = FakeProvider()
    state = {"cancel": False}

    def progress(done: int, total: int) -> None:
        state["cancel"] = True

    with pytest.raises(TranslationCancelled):
        translate_transcript(
            transcript,
            provider,
            settings=settings(window_chars=25),
            progress=progress,
            should_cancel=lambda: state["cancel"],
        )

    assert provider.calls == [[1, 2]]
    assert not transcript.cues[2].is_translated
    assert isinstance(TranslationCancelled("x"), TranslationError)


def test_should_cancel_stops_between_retries():
    transcript = make_transcript(1)
    provider = FakeProvider(fail_first=99)
    calls = {"count": 0}

    def should_cancel() -> bool:
        # armed only once the first attempt is in flight, so the cancel must be
        # honoured by the retry path rather than before the window starts
        calls["count"] += 1
        return calls["count"] > 2

    with pytest.raises(TranslationCancelled):
        translate_transcript(
            transcript,
            provider,
            settings=settings(max_retries=6),
            should_cancel=should_cancel,
        )

    assert provider.calls == [[1]]
