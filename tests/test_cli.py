"""Command line interface behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from dosubs import cli
from dosubs.media.probe import have_ffmpeg

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg is required to generate media")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep tests away from the real user configuration and log files."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for name in ("DOSUBS_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY", "DOSUBS_OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "roaming"


def test_selfcheck_reports_the_environment(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--selfcheck"]) == 0
    out = capsys.readouterr().out
    assert "ffmpeg" in out
    assert "device" in out
    assert "recommended" in out
    assert "api key" in out


def test_help_is_shown_without_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == cli.EXIT_USAGE
    assert "usage: dosubs" in capsys.readouterr().out


def test_unknown_provider_key_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--set-key", "not-a-provider"]) == cli.EXIT_USAGE
    assert "unknown provider" in capsys.readouterr().out


def test_storing_a_key_never_echoes_it(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stored: dict[str, str] = {}
    monkeypatch.setattr(cli, "set_api_key", lambda provider, key: stored.update({provider: key}))
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "sk-supersecret-1234567890")

    assert cli.main(["--set-key", "deepseek"]) == 0
    out = capsys.readouterr().out
    assert stored["deepseek"] == "sk-supersecret-1234567890"
    assert "sk-supersecret-1234567890" not in out
    assert "sk-s...7890" in out


def test_empty_key_input_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "   ")
    assert cli.main(["--set-key", "deepseek"]) == cli.EXIT_USAGE
    assert "nothing stored" in capsys.readouterr().out


def test_dry_run_lists_the_three_files(
    sample_video: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([str(sample_video), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ExampleVideo.original.srt" in out
    assert "ExampleVideo.zh-CN.srt" in out
    assert "ExampleVideo.bilingual.srt" in out


def test_dry_run_honours_only_original(
    sample_video: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([str(sample_video), "--dry-run", "--only", "original"]) == 0
    out = capsys.readouterr().out
    assert "ExampleVideo.original.srt" in out
    assert "zh-CN.srt" not in out


def test_dry_run_writes_nothing(sample_video: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert cli.main([str(sample_video), "--dry-run", "-o", str(out_dir)]) == 0
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_missing_media_file_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([str(tmp_path / "nope.mp4")]) == cli.EXIT_FAILED
    assert "input problem" in capsys.readouterr().out


def test_translation_without_a_key_is_a_usage_error(
    sample_video: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "get_api_key", lambda provider: None)
    assert cli.main([str(sample_video), "--provider", "deepseek"]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert "--set-key deepseek" in out


def test_no_translate_skips_the_key_requirement(
    sample_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "get_api_key", lambda provider: None)

    class _FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, specs):
            from dosubs.models import JobResult, JobStatus

            return [JobResult(media=specs[0].media, status=JobStatus.DONE, written=[])]

    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    assert cli.main([str(sample_video), "--no-translate"]) == cli.EXIT_OK


def test_provider_preset_overrides_configuration(
    sample_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching provider must carry the preset endpoint, not leave DeepSeek's."""
    captured: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, config, **kwargs) -> None:
            captured["translation"] = config.translation

        def run(self, specs):
            from dosubs.models import JobResult, JobStatus

            return [JobResult(media=specs[0].media, status=JobStatus.DONE, written=[])]

    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setenv("DOSUBS_OLLAMA_API_KEY", "unused")
    assert cli.main([str(sample_video), "--provider", "ollama"]) == cli.EXIT_OK

    translation = captured["translation"]
    assert translation.provider == "ollama"
    assert translation.base_url == "http://127.0.0.1:11434/v1"
    assert translation.model == "qwen3:8b"


def test_summary_reports_failures(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from dosubs.models import JobResult, JobStatus

    results = [
        JobResult(media=tmp_path / "a.mp4", status=JobStatus.DONE, written=[tmp_path / "a.srt"]),
        JobResult(media=tmp_path / "b.mp4", status=JobStatus.FAILED, error="boom"),
    ]
    assert cli._print_summary(results, cancelled=False) == cli.EXIT_FAILED
    out = capsys.readouterr().out
    assert "OK" in out and "FAILED" in out and "boom" in out
    assert "1 of 2 file(s) did not complete" in out


def test_summary_reports_cancellation(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from dosubs.models import JobResult, JobStatus

    results = [JobResult(media=tmp_path / "a.mp4", status=JobStatus.CANCELLED)]
    assert cli._print_summary(results, cancelled=True) == cli.EXIT_CANCELLED
    assert "cancelled" in capsys.readouterr().out
