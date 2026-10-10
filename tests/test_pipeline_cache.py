"""Transcript caching: changing translation settings must not re-run recognition.

The cache only inspects file identity and recognition parameters, so the unit tests
use plain files; the integration tests use real media through ffmpeg.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from dosubs.config import AppConfig
from dosubs.media.probe import have_ffmpeg
from dosubs.models import (
    AsrSegment,
    AsrSettings,
    JobSpec,
    JobStatus,
    OverwritePolicy,
    Word,
)
from dosubs.pipeline.cache import (
    CACHE_VERSION,
    cache_path,
    clear_cache,
    load_cached_transcript,
    store_transcript,
)
from dosubs.pipeline.orchestrator import Pipeline

requires_ffmpeg = pytest.mark.skipif(
    not have_ffmpeg(), reason="ffmpeg is required to generate media"
)


def _segments() -> list[AsrSegment]:
    return [
        AsrSegment(
            start_ms=500,
            end_ms=2500,
            text="Hello there, my friend.",
            words=(
                Word(start_ms=500, end_ms=1000, text="Hello", probability=0.9),
                Word(start_ms=1000, end_ms=1500, text="there,", probability=0.8),
                Word(start_ms=1500, end_ms=2500, text="my friend.", probability=0.85),
            ),
        )
    ]


def _store(media: Path, asr: AsrSettings, root: Path) -> Path:
    path = store_transcript(
        media,
        asr,
        segments=_segments(),
        language="en",
        language_probability=0.98,
        language_forced=False,
        duration_s=8.0,
        root=root,
    )
    assert path is not None
    return path


@pytest.fixture()
def media(tmp_path: Path) -> Path:
    """A stand-in media file: the cache fingerprints size and mtime, not content."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"pretend this is a video" * 100)
    return target


def test_round_trip_including_word_timings(media: Path, tmp_path: Path) -> None:
    asr = AsrSettings()
    _store(media, asr, tmp_path / "cache")

    hit = load_cached_transcript(media, asr, root=tmp_path / "cache")
    assert hit is not None
    assert hit.language == "en"
    assert hit.language_probability == pytest.approx(0.98)
    assert hit.duration_s == pytest.approx(8.0)
    assert len(hit.segments) == 1
    segment = hit.segments[0]
    assert (segment.start_ms, segment.end_ms, segment.text) == (
        500,
        2500,
        "Hello there, my friend.",
    )
    assert [word.text for word in segment.words] == ["Hello", "there,", "my friend."]
    assert segment.words[0].probability == pytest.approx(0.9)
    assert not list((tmp_path / "cache").glob("*.part"))


def test_unknown_media_is_a_miss(media: Path, tmp_path: Path) -> None:
    _store(media, AsrSettings(), tmp_path / "cache")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"different")
    assert load_cached_transcript(other, AsrSettings(), root=tmp_path / "cache") is None


def test_changed_content_is_a_miss(media: Path, tmp_path: Path) -> None:
    asr = AsrSettings()
    _store(media, asr, tmp_path / "cache")
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is not None

    with media.open("ab") as handle:
        handle.write(b"more bytes")
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None


def test_touched_file_is_a_miss(media: Path, tmp_path: Path) -> None:
    asr = AsrSettings()
    _store(media, asr, tmp_path / "cache")
    os.utime(media, (1_600_000_000, 1_600_000_000))
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model", "small"),
        ("language", "en"),
        ("beam_size", 1),
        ("vad_filter", False),
        ("initial_prompt", "Names: Alice"),
        ("batch_size", 8),
    ],
)
def test_recognition_changes_invalidate_the_entry(
    media: Path, tmp_path: Path, field_name: str, value: object
) -> None:
    asr = AsrSettings()
    _store(media, asr, tmp_path / "cache")
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is not None

    changed = AsrSettings()
    setattr(changed, field_name, value)
    assert load_cached_transcript(media, changed, root=tmp_path / "cache") is None


def test_execution_only_changes_still_hit(media: Path, tmp_path: Path) -> None:
    _store(media, AsrSettings(), tmp_path / "cache")
    presentation_only = AsrSettings()
    presentation_only.device = "cpu"
    presentation_only.compute_type = "int8"
    assert load_cached_transcript(media, presentation_only, root=tmp_path / "cache") is not None


def test_wordless_entry_cannot_satisfy_a_word_timing_request(media: Path, tmp_path: Path) -> None:
    asr = AsrSettings()
    wordless = [AsrSegment(start_ms=0, end_ms=1000, text="No words here.")]
    store_transcript(
        media,
        asr,
        segments=wordless,
        language="en",
        language_probability=0.9,
        language_forced=False,
        duration_s=1.0,
        root=tmp_path / "cache",
    )
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None

    no_words_wanted = AsrSettings()
    no_words_wanted.word_timestamps = False
    assert load_cached_transcript(media, no_words_wanted, root=tmp_path / "cache") is not None


def test_an_entry_with_words_also_satisfies_a_run_that_does_not_need_them(
    media: Path, tmp_path: Path
) -> None:
    _store(media, AsrSettings(), tmp_path / "cache")
    no_words_wanted = AsrSettings()
    no_words_wanted.word_timestamps = False
    hit = load_cached_transcript(media, no_words_wanted, root=tmp_path / "cache")
    assert hit is not None
    assert hit.segments[0].words


def test_corrupt_entry_is_a_miss_not_an_error(media: Path, tmp_path: Path) -> None:
    asr = AsrSettings()
    target = _store(media, asr, tmp_path / "cache")

    target.write_bytes(b"\x1f\x8bnot gzip at all")
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None

    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump({"version": CACHE_VERSION, "segments": [["bad"]]}, handle)
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None

    target.unlink()
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None


def test_version_mismatch_is_a_miss(media: Path, tmp_path: Path) -> None:
    asr = AsrSettings()
    target = _store(media, asr, tmp_path / "cache")
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["version"] = CACHE_VERSION + 1
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert load_cached_transcript(media, asr, root=tmp_path / "cache") is None


def test_clear_cache_removes_entries(media: Path, tmp_path: Path) -> None:
    _store(media, AsrSettings(), tmp_path / "cache")
    removed, freed = clear_cache(tmp_path / "cache")
    assert removed == 1
    assert freed > 0
    assert load_cached_transcript(media, AsrSettings(), root=tmp_path / "cache") is None
    assert clear_cache(tmp_path / "cache") == (0, 0)
    assert clear_cache(tmp_path / "missing") == (0, 0)


# --------------------------------------------------------------------------- #
# The requirement itself, through the real pipeline
# --------------------------------------------------------------------------- #


class _CountingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, media, *, duration_s=None, on_progress=None, should_cancel=None):
        from dosubs.asr.engine import AsrResult

        self.calls += 1
        return AsrResult(
            segments=_segments(),
            language="en",
            language_probability=0.97,
            duration_s=duration_s or 8.0,
            model="counting",
            device="cpu",
            compute_type="int8",
            elapsed_s=0.0,
        )

    def load_model(self):
        return None


class _FakeProvider:
    name = "fake"
    model = "fake"

    def translate_window(self, items, *, source_language, target_language):
        return {cue_id: f"译:{text}" for cue_id, text in items}

    def check_connection(self) -> str:
        return "ok"

    def close(self) -> None:
        pass


def _spec(media: Path, out: Path, **translation_overrides: object) -> JobSpec:
    config = AppConfig()
    config.export.output_dir = out
    config.export.overwrite = OverwritePolicy.OVERWRITE
    for key, value in translation_overrides.items():
        setattr(config.translation, key, value)
    return JobSpec(
        media=media, asr=config.asr, translation=config.translation, export=config.export
    )


@requires_ffmpeg
def test_changing_only_the_target_language_skips_recognition(
    sample_video: Path, tmp_path: Path
) -> None:
    engine = _CountingEngine()
    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: engine,
        provider_factory=lambda settings: _FakeProvider(),
        cache_root=tmp_path / "cache",
    )

    first = pipeline.run_job(_spec(sample_video, tmp_path / "out1"))
    assert first.status is JobStatus.DONE, first.error
    assert engine.calls == 1

    second = pipeline.run_job(_spec(sample_video, tmp_path / "out2", target_language="ja"))
    assert second.status is JobStatus.DONE, second.error
    assert engine.calls == 1, "recognition ran again even though the media did not change"
    assert (tmp_path / "out2" / "ExampleVideo.ja.srt").exists()
    assert (tmp_path / "out2" / "ExampleVideo.original.srt").exists()


@requires_ffmpeg
def test_disabling_the_cache_forces_recognition(sample_video: Path, tmp_path: Path) -> None:
    engine = _CountingEngine()
    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: engine,
        provider_factory=lambda settings: _FakeProvider(),
        cache_root=tmp_path / "cache",
        use_cache=False,
    )
    pipeline.run_job(_spec(sample_video, tmp_path / "out"))
    pipeline.run_job(_spec(sample_video, tmp_path / "out"))
    assert engine.calls == 2
    assert not list((tmp_path / "cache").glob("*.json.gz"))


@requires_ffmpeg
def test_recognition_settings_change_invalidates_through_the_pipeline(
    sample_video: Path, tmp_path: Path
) -> None:
    engine = _CountingEngine()
    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: engine,
        provider_factory=lambda settings: _FakeProvider(),
        cache_root=tmp_path / "cache",
    )
    first = pipeline.run_job(_spec(sample_video, tmp_path / "out"))
    assert first.status is JobStatus.DONE, first.error

    changed = _spec(sample_video, tmp_path / "out2")
    changed.asr.model = "small-variant"
    second = pipeline.run_job(changed)
    assert second.status is JobStatus.DONE, second.error
    assert engine.calls == 2
