"""End-to-end orchestration with a fake engine and a fake translation provider.

These tests exercise the real pipeline, the real exporter and the real verifier
against a real (generated) media file, while stubbing only the two costly external
dependencies: speech recognition and the translation API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from dosubs.config import AppConfig
from dosubs.media.probe import have_ffmpeg
from dosubs.models import (
    AsrSegment,
    ExportKind,
    JobSpec,
    JobStatus,
    OverwritePolicy,
    Stage,
    Word,
)
from dosubs.pipeline import JobCancelled, Pipeline
from dosubs.subs.io import read_subtitles
from dosubs.translate.base import TranslationError

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg is required to generate media")


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


def _words(text: str, start_ms: int, end_ms: int) -> tuple[Word, ...]:
    tokens = text.split()
    step = (end_ms - start_ms) / max(1, len(tokens))
    return tuple(
        Word(
            start_ms=int(start_ms + index * step),
            end_ms=int(start_ms + (index + 1) * step),
            text=token,
        )
        for index, token in enumerate(tokens)
    )


def _segments() -> list[AsrSegment]:
    first = "I never thought we would meet again."
    second = "What are you doing here?"
    return [
        AsrSegment(start_ms=1000, end_ms=4000, text=first, words=_words(first, 1000, 4000)),
        AsrSegment(start_ms=4500, end_ms=7000, text=second, words=_words(second, 4500, 7000)),
    ]


class FakeEngine:
    """Stands in for FasterWhisperEngine."""

    def __init__(self, segments: list[AsrSegment] | None = None) -> None:
        self.segments = _segments() if segments is None else segments
        self.calls = 0
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe(self, media, *, duration_s=None, on_progress=None, should_cancel=None):
        from dosubs.asr.engine import AsrCancelled, AsrResult

        self.calls += 1
        if should_cancel is not None and should_cancel():
            raise AsrCancelled("cancelled")
        for segment in self.segments:
            if on_progress is not None:
                on_progress(0.5, segment.text)
        return AsrResult(
            segments=list(self.segments),
            language="en",
            language_probability=0.98,
            duration_s=duration_s or 8.0,
            model="fake",
            device="cpu",
            compute_type="int8",
            elapsed_s=0.01,
        )

    def load_model(self):  # pragma: no cover - not used by the pipeline
        return None

    def close(self) -> None:
        pass


class FakeProvider:
    """Stands in for a real translation provider."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False
        self.requests: list[list[tuple[int, str]]] = []

    def translate_window(self, items, *, source_language, target_language):
        self.requests.append(list(items))
        if self.fail:
            raise TranslationError("provider exploded")
        return {cue_id: f"译:{text}" for cue_id, text in items}

    def check_connection(self) -> str:
        return "fake ok"

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def media(sample_video: Path) -> Path:
    return sample_video


def _spec(media: Path, out_dir: Path, **overrides) -> JobSpec:
    config = AppConfig()
    config.export.output_dir = out_dir
    config.export.overwrite = overrides.pop("overwrite", OverwritePolicy.OVERWRITE)
    for key, value in overrides.items():
        if hasattr(config.asr, key):
            setattr(config.asr, key, value)
        elif hasattr(config.translation, key):
            setattr(config.translation, key, value)
        elif hasattr(config.export, key):
            setattr(config.export, key, value)
    return JobSpec(
        media=media, asr=config.asr, translation=config.translation, export=config.export
    )


def _pipeline(out: list, **kwargs) -> Pipeline:
    return Pipeline(
        AppConfig(),
        on_progress=out.append,
        engine_factory=lambda settings: FakeEngine(),
        provider_factory=lambda settings: FakeProvider(**kwargs),
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_all_three_files_are_written_with_one_timeline(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _pipeline([]).run_job(_spec(media, out))

    assert result.status is JobStatus.DONE, result.error
    names = sorted(path.name for path in result.written)
    assert names == ["ExampleVideo.bilingual.srt", "ExampleVideo.original.srt", "ExampleVideo.zh-CN.srt"]

    timelines = {
        name: [(event.start, event.end) for event in read_subtitles(out / name)]
        for name in names
    }
    original = timelines["ExampleVideo.original.srt"]
    assert timelines["ExampleVideo.zh-CN.srt"] == original
    assert timelines["ExampleVideo.bilingual.srt"] == original
    assert original == [(1000, 4000), (4500, 7000)]


def test_export_content_is_correct(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pipeline([]).run_job(_spec(media, out))

    original = (out / "ExampleVideo.original.srt").read_text(encoding="utf-8")
    translated = (out / "ExampleVideo.zh-CN.srt").read_text(encoding="utf-8")
    bilingual = (out / "ExampleVideo.bilingual.srt").read_text(encoding="utf-8")

    def flattened(text: str) -> str:
        """Cue text with line wrapping removed."""
        return " ".join(text.split())

    assert "I never thought we would meet again." in original
    assert "译:" not in original
    assert "译:I never thought we would meet again." in flattened(translated)
    assert "I never thought we would meet again." not in translated
    assert "译:" in bilingual
    assert "I never thought we would meet again." in flattened(bilingual)

    blocks = bilingual.strip().split("\n\n")
    assert len(blocks) == 2
    first_lines = blocks[0].splitlines()
    assert first_lines[2] == "I never thought we would meet again."
    assert first_lines[3].startswith("译:I never thought")


def test_progress_events_cover_every_stage(media: Path, tmp_path: Path) -> None:
    events: list = []
    _pipeline(events).run_job(_spec(media, tmp_path / "out"))
    stages = [event.stage for event in events]
    for stage in (Stage.PROBE, Stage.TRANSCRIBE, Stage.SEGMENT, Stage.TRANSLATE, Stage.EXPORT):
        assert stage in stages, f"no progress reported for {stage}"
    assert events[-1].file_index == 1


# --------------------------------------------------------------------------- #
# Overwrite policy
# --------------------------------------------------------------------------- #


def test_existing_files_are_left_alone_under_the_ask_policy(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "ExampleVideo.original.srt"
    existing.write_text("keep me", encoding="utf-8")

    result = _pipeline([]).run_job(_spec(media, out, overwrite=OverwritePolicy.ASK))

    assert existing.read_text(encoding="utf-8") == "keep me"
    assert [path.name for path in result.skipped] == ["ExampleVideo.original.srt"]
    assert all(path.name != "ExampleVideo.original.srt" for path in result.written)
    assert "left 1 existing file(s) untouched" in (result.error or "")


def test_a_resolver_decision_is_honoured(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "ExampleVideo.original.srt"
    existing.write_text("stale", encoding="utf-8")

    seen: list[Path] = []

    def resolve(media_path: Path, conflicts: list[Path]) -> OverwritePolicy:
        seen.extend(conflicts)
        return OverwritePolicy.OVERWRITE

    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: FakeEngine(),
        provider_factory=lambda settings: FakeProvider(),
        resolve_conflicts=resolve,
    )
    result = pipeline.run_job(_spec(media, out, overwrite=OverwritePolicy.ASK))

    assert seen == [existing]
    assert "stale" not in existing.read_text(encoding="utf-8")


def test_rename_policy_keeps_both_copies(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "ExampleVideo.original.srt").write_text("stale", encoding="utf-8")

    result = _pipeline([]).run_job(_spec(media, out, overwrite=OverwritePolicy.RENAME))

    assert (out / "ExampleVideo.original (2).srt") in result.written
    assert (out / "ExampleVideo.original.srt").read_text(encoding="utf-8") == "stale"


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_translation_failure_keeps_the_original_subtitles(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _pipeline([], fail=True).run_job(_spec(media, out))

    assert result.status is JobStatus.FAILED
    assert "translation failed" in (result.error or "")
    assert [path.name for path in result.written] == ["ExampleVideo.original.srt"]
    assert not (out / "ExampleVideo.zh-CN.srt").exists()


def test_matching_source_and_target_skips_translation(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _pipeline([]).run_job(_spec(media, out, target_language="en-GB"))

    assert result.status is JobStatus.DONE
    assert [path.name for path in result.written] == ["ExampleVideo.original.srt"]
    assert "translation was skipped" in (result.error or "")


def test_translation_disabled_writes_only_the_original(media: Path, tmp_path: Path) -> None:
    result = _pipeline([]).run_job(_spec(media, tmp_path / "out", enabled=False))
    assert [path.name for path in result.written] == ["ExampleVideo.original.srt"]


def test_no_speech_writes_nothing(media: Path, tmp_path: Path) -> None:
    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: FakeEngine(segments=[]),
        provider_factory=lambda settings: FakeProvider(),
    )
    result = pipeline.run_job(_spec(media, tmp_path / "out"))
    assert result.status is JobStatus.SKIPPED
    assert result.written == []
    assert "no speech" in (result.error or "")


def test_missing_file_fails_without_stopping_the_batch(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    results = _pipeline([]).run(
        [_spec(tmp_path / "nope.mp4", out), _spec(media, out)]
    )
    assert [result.status for result in results] == [JobStatus.FAILED, JobStatus.DONE]
    assert "nope.mp4" in (results[0].error or "")


def test_cancellation_reports_cancelled_and_writes_nothing(media: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: FakeEngine(),
        provider_factory=lambda settings: FakeProvider(),
        should_cancel=lambda: True,
    )
    results = pipeline.run([_spec(media, out)])
    assert results[0].status is JobStatus.CANCELLED
    assert results[0].written == []
    assert not out.exists() or list(out.iterdir()) == []


def test_cancellation_between_jobs_is_respected(media: Path, tmp_path: Path) -> None:
    state = {"cancel": False}

    def on_progress(event) -> None:
        if event.stage is Stage.EXPORT and event.fraction == 1.0:
            state["cancel"] = True

    out = tmp_path / "out"
    pipeline = Pipeline(
        AppConfig(),
        on_progress=on_progress,
        engine_factory=lambda settings: FakeEngine(),
        provider_factory=lambda settings: FakeProvider(),
        should_cancel=lambda: state["cancel"],
    )
    results = pipeline.run([_spec(media, out), _spec(media, out)])
    assert results[0].status is JobStatus.DONE
    assert results[1].status is JobStatus.CANCELLED


def test_locale_and_unicode_paths_are_handled(
    tmp_path: Path, make_video_file: Callable[..., Path]
) -> None:
    source = make_video_file("我的 视频.mp4", 2, directory=tmp_path)
    out = tmp_path / "字幕"
    result = _pipeline([]).run_job(_spec(source, out))
    assert result.status is JobStatus.DONE, result.error
    assert sorted(path.name for path in result.written) == [
        "我的 视频.bilingual.srt",
        "我的 视频.original.srt",
        "我的 视频.zh-CN.srt",
    ]


def test_job_cancelled_from_the_provider_is_translated_to_a_cancel(media: Path, tmp_path: Path) -> None:
    from dosubs.translate.base import TranslationCancelled

    class CancellingProvider(FakeProvider):
        def translate_window(self, items, *, source_language, target_language):
            raise TranslationCancelled("user cancelled")

    pipeline = Pipeline(
        AppConfig(),
        engine_factory=lambda settings: FakeEngine(),
        provider_factory=lambda settings: CancellingProvider(),
    )
    with pytest.raises(JobCancelled):
        pipeline.run_job(_spec(media, tmp_path / "out"))
