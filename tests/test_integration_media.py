r"""End-to-end integration against real media and a real (local) HTTP server.

These tests run the actual speech recogniser and the actual HTTP client. They are
skipped unless a media file is provided, because no sample video can be generated
with synthetic speech:

    $env:DOSUBS_TEST_MEDIA = "D:\videos\clip.mp4"
    .\.venv\Scripts\python.exe -m pytest -m integration tests/test_integration_media.py

`DOSUBS_TEST_MODEL` (default ``tiny``) selects the model; translation is answered by a
local OpenAI-compatible stub so no API key or internet access is required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dosubs.config import AppConfig
from dosubs.models import JobSpec, JobStatus, OverwritePolicy, Stage
from dosubs.pipeline import Pipeline
from dosubs.subs.io import read_subtitles
from dosubs.subs.verify import check_srt_text

from conftest import STUB_TRANSLATION, _StubTranslationHandler

pytestmark = pytest.mark.integration

MODEL_ENV = "DOSUBS_TEST_MODEL"


def _spec(media: Path, out: Path, base_url: str) -> JobSpec:
    config = AppConfig()
    config.asr.model = os.environ.get(MODEL_ENV, "tiny")
    config.asr.device = "auto"
    config.translation.provider = "custom"
    config.translation.base_url = base_url
    config.translation.model = "stub"
    config.translation.target_language = "zh-CN"
    config.translation.max_retries = 2
    config.export.output_dir = out
    config.export.overwrite = OverwritePolicy.OVERWRITE
    return JobSpec(
        media=media, asr=config.asr, translation=config.translation, export=config.export
    )


def test_real_recognition_and_translation_produce_three_valid_files(
    real_media: Path, stub_server: str, tmp_path: Path
) -> None:
    media = real_media
    out = tmp_path / "out"
    events: list = []
    config = AppConfig()
    pipeline = Pipeline(config, on_progress=events.append, cache_root=tmp_path / "cache")

    result = pipeline.run_job(_spec(media, out, stub_server))

    assert result.status is JobStatus.DONE, result.error
    assert {path.name for path in result.written} == {
        f"{media.stem}.original.srt",
        f"{media.stem}.zh-CN.srt",
        f"{media.stem}.bilingual.srt",
    }

    originals = [(event.start, event.end) for event in read_subtitles(out / f"{media.stem}.original.srt")]
    translations = [
        (event.start, event.end) for event in read_subtitles(out / f"{media.stem}.zh-CN.srt")
    ]
    bilinguals = [
        (event.start, event.end) for event in read_subtitles(out / f"{media.stem}.bilingual.srt")
    ]

    assert originals, "no speech was recognised in the sample media"
    assert originals == translations == bilinguals, "the three files must share one timeline"
    assert all(start < end for start, end in originals)

    for path in result.written:
        problems = check_srt_text(path.read_text(encoding="utf-8"))
        assert problems == [], f"{path.name}: {problems}"

    translated_text = (out / f"{media.stem}.zh-CN.srt").read_text(encoding="utf-8")
    assert STUB_TRANSLATION in translated_text
    assert _StubTranslationHandler.seen_requests >= 1

    stages = {event.stage for event in events}
    assert stages == {
        Stage.PROBE,
        Stage.TRANSCRIBE,
        Stage.SEGMENT,
        Stage.TRANSLATE,
        Stage.EXPORT,
    }


def test_second_run_reuses_the_cached_transcript(
    real_media: Path, stub_server: str, tmp_path: Path
) -> None:
    config = AppConfig()
    pipeline = Pipeline(config, cache_root=tmp_path / "cache")

    first = pipeline.run_job(_spec(real_media, tmp_path / "a", stub_server))
    assert first.status is JobStatus.DONE, first.error

    stages: list = []
    second_pipeline = Pipeline(
        AppConfig(),
        on_progress=lambda event: stages.append(event),
        cache_root=tmp_path / "cache",
    )
    second = second_pipeline.run_job(_spec(real_media, tmp_path / "b", stub_server))
    assert second.status is JobStatus.DONE, second.error
    assert any("reusing cached transcript" in event.message for event in stages)


def test_cancelling_a_real_run_leaves_no_files(
    real_media: Path, stub_server: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    state = {"cancel": False}

    def on_progress(event) -> None:
        if event.stage is Stage.TRANSCRIBE:
            state["cancel"] = True

    pipeline = Pipeline(
        AppConfig(),
        on_progress=on_progress,
        should_cancel=lambda: state["cancel"],
        cache_root=tmp_path / "cache",
    )
    result = pipeline.run([_spec(real_media, out, stub_server)])[0]

    assert result.status is JobStatus.CANCELLED
    assert result.written == []
    assert not out.exists() or not list(out.glob("*.srt"))
