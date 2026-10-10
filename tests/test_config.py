"""Configuration persistence, paths and credential handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dosubs import config as config_module
from dosubs.config import AppConfig, ConfigError, load_config, save_config
from dosubs.models import OverwritePolicy, SubtitleFormat, translation_settings_for


def test_missing_file_yields_defaults(tmp_path: Path) -> None:
    loaded = load_config(tmp_path / "does-not-exist.json")
    assert loaded == AppConfig()
    assert loaded.asr.model == "large-v3"
    assert loaded.translation.target_language == "zh-CN"
    assert loaded.export.overwrite is OverwritePolicy.ASK


def test_round_trip_preserves_settings(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    original = AppConfig()
    original.asr.model = "large-v3-turbo"
    original.asr.device = "cuda"
    original.asr.batch_size = 16
    original.asr.language = "en"
    original.translation.provider = "ollama"
    original.translation.base_url = "http://127.0.0.1:11434/v1"
    original.translation.model = "qwen3:8b"
    original.translation.enabled = False
    original.export.output_dir = tmp_path / "out"
    original.export.format = SubtitleFormat.VTT
    original.export.overwrite = OverwritePolicy.RENAME
    original.export.write_bilingual = False
    original.log_level = "DEBUG"

    save_config(original, target)
    loaded = load_config(target)

    assert loaded.asr.model == "large-v3-turbo"
    assert loaded.asr.batch_size == 16
    assert loaded.asr.language == "en"
    assert loaded.translation.provider == "ollama"
    assert loaded.translation.model == "qwen3:8b"
    assert loaded.translation.enabled is False
    assert loaded.export.output_dir == tmp_path / "out"
    assert loaded.export.format is SubtitleFormat.VTT
    assert loaded.export.overwrite is OverwritePolicy.RENAME
    assert loaded.export.write_bilingual is False
    assert loaded.log_level == "DEBUG"


def test_output_dir_round_trips_as_empty_for_none(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    save_config(AppConfig(), target)
    assert json.loads(target.read_text(encoding="utf-8"))["export"]["output_dir"] == ""
    assert load_config(target).export.output_dir is None


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    save_config(AppConfig(), target)
    assert target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_configuration_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(target)


def test_non_object_configuration_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(target)


def test_unknown_keys_and_bad_enums_fall_back(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text(
        json.dumps(
            {
                "asr": {"model": "small", "not_a_field": 1},
                "export": {"format": "docx", "overwrite": "explode", "output_dir": ""},
                "translation": {"unknown": True},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_config(target)
    assert loaded.asr.model == "small"
    assert loaded.export.format is SubtitleFormat.SRT
    assert loaded.export.overwrite is OverwritePolicy.ASK


def test_environment_variable_wins_over_stored_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOSUBS_DEEPSEEK_API_KEY", "sk-env-value-1234")
    assert config_module.get_api_key("deepseek") == "sk-env-value-1234"


def test_provider_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config_module.env_var_names("deepseek") == (
        "DOSUBS_DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY",
    )
    assert "DOSUBS_MY_HOST_API_KEY" in config_module.env_var_names("my-host")


def test_missing_key_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in config_module.env_var_names("deepseek"):
        monkeypatch.delenv(name, raising=False)

    class _FailingKeyring:
        @staticmethod
        def get_password(service: str, provider: str) -> None:
            return None

    monkeypatch.setattr(config_module, "_keyring", lambda: _FailingKeyring)
    assert config_module.get_api_key("deepseek") is None
    assert config_module.has_api_key("deepseek") is False


def test_mask_secret_never_reveals_the_middle() -> None:
    assert config_module.mask_secret("sk-abcdefghijklmnop") == "sk-a...mnop"
    assert config_module.mask_secret("short") == "***"
    assert config_module.mask_secret(None) == "(none)"


def test_paths_live_under_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert config_module.config_dir() == tmp_path / "roaming" / "DoSubs"
    assert config_module.config_path().name == "config.json"
    assert config_module.data_dir() == tmp_path / "local" / "DoSubs"
    assert config_module.logs_dir().parent == config_module.data_dir()


def test_translation_settings_for_applies_the_preset() -> None:
    settings = translation_settings_for("ollama")
    assert settings.provider == "ollama"
    assert settings.base_url == "http://127.0.0.1:11434/v1"
    assert settings.model == "qwen3:8b"
    assert settings.target_language == "zh-CN"


def test_translation_settings_for_allows_overrides() -> None:
    settings = translation_settings_for("deepseek", model="deepseek-v4-pro", temperature=0.0)
    assert settings.model == "deepseek-v4-pro"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.temperature == 0.0


def test_translation_settings_for_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown translation provider"):
        translation_settings_for("nope")
