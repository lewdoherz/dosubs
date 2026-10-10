"""Output naming and overwrite policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from dosubs.models import ExportKind, OverwritePolicy, SubtitleFormat
from dosubs.subs import naming


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("zh-cn", "zh-CN"),
        ("zh", "zh"),
        ("ZH-HANS", "zh-Hans"),
        ("en", "en"),
        ("pt-br", "pt-BR"),
    ],
)
def test_language_tag_normalisation(raw: str, expected: str) -> None:
    assert naming.normalize_language_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "zh CN", "../etc", "1", "zh/CN"])
def test_invalid_language_tags_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        naming.normalize_language_tag(raw)


def test_subtitle_paths_follow_the_specification(tmp_path: Path) -> None:
    media = tmp_path / "ExampleVideo.mp4"
    assert naming.subtitle_path(media, ExportKind.ORIGINAL).name == "ExampleVideo.original.srt"
    assert (
        naming.subtitle_path(media, ExportKind.TRANSLATED, target_language="zh-CN").name
        == "ExampleVideo.zh-CN.srt"
    )
    assert naming.subtitle_path(media, ExportKind.BILINGUAL).name == "ExampleVideo.bilingual.srt"


def test_subtitle_path_honours_output_dir_and_format(tmp_path: Path) -> None:
    path = naming.subtitle_path(
        tmp_path / "clip.mkv",
        ExportKind.TRANSLATED,
        target_language="zh-cn",
        fmt=SubtitleFormat.VTT,
        output_dir=tmp_path / "subs",
    )
    assert path == tmp_path / "subs" / "clip.zh-CN.vtt"


def test_unicode_and_spaced_filenames_survive(tmp_path: Path) -> None:
    path = naming.subtitle_path(tmp_path / "我的 视频 (final).mp4", ExportKind.BILINGUAL)
    assert path.name == "我的 视频 (final).bilingual.srt"


def test_plan_exports_without_existing_files(tmp_path: Path) -> None:
    plan = naming.plan_exports(
        tmp_path / "a.mp4", [ExportKind.ORIGINAL, ExportKind.TRANSLATED, ExportKind.BILINGUAL]
    )
    assert len(plan.targets) == 3
    assert plan.conflicts == ()
    assert plan.skipped == ()


def test_ask_policy_reports_conflicts_without_resolving_them(tmp_path: Path) -> None:
    existing = tmp_path / "a.original.srt"
    existing.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n", encoding="utf-8")

    plan = naming.plan_exports(
        tmp_path / "a.mp4", [ExportKind.ORIGINAL], policy=OverwritePolicy.ASK
    )
    assert plan.targets == ()
    assert plan.conflicts == (existing,)


def test_overwrite_policy_keeps_the_name(tmp_path: Path) -> None:
    (tmp_path / "a.original.srt").write_text("x", encoding="utf-8")
    plan = naming.plan_exports(
        tmp_path / "a.mp4", [ExportKind.ORIGINAL], policy=OverwritePolicy.OVERWRITE
    )
    assert plan.paths() == [tmp_path / "a.original.srt"]


def test_skip_policy_leaves_existing_files_alone(tmp_path: Path) -> None:
    (tmp_path / "a.original.srt").write_text("x", encoding="utf-8")
    plan = naming.plan_exports(
        tmp_path / "a.mp4", [ExportKind.ORIGINAL], policy=OverwritePolicy.SKIP
    )
    assert plan.targets == ()
    assert [t.path for t in plan.skipped] == [tmp_path / "a.original.srt"]


def test_rename_policy_picks_a_free_name(tmp_path: Path) -> None:
    (tmp_path / "a.original.srt").write_text("x", encoding="utf-8")
    (tmp_path / "a.original (2).srt").write_text("x", encoding="utf-8")
    plan = naming.plan_exports(
        tmp_path / "a.mp4", [ExportKind.ORIGINAL], policy=OverwritePolicy.RENAME
    )
    assert plan.paths() == [tmp_path / "a.original (3).srt"]


def test_the_three_exports_never_collide(tmp_path: Path) -> None:
    plan = naming.plan_exports(
        tmp_path / "a.mp4",
        [ExportKind.ORIGINAL, ExportKind.TRANSLATED, ExportKind.BILINGUAL],
        target_language="zh-CN",
    )
    assert sorted(p.name for p in plan.paths()) == sorted(
        ["a.original.srt", "a.zh-CN.srt", "a.bilingual.srt"]
    )


def test_duplicate_kinds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same file"):
        naming.plan_exports(tmp_path / "a.mp4", [ExportKind.ORIGINAL, ExportKind.ORIGINAL])
