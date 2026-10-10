"""Application paths, persisted settings and API credential storage.

Settings live in ``%APPDATA%\\DoSubs\\config.json``; logs, caches and downloaded
models live in ``%LOCALAPPDATA%\\DoSubs``. API keys are never written to the config
file: they are kept in the Windows credential store through ``keyring`` (DPAPI
backed) or supplied through an environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import APP_NAME
from .models import (
    AsrSettings,
    ExportSettings,
    OverwritePolicy,
    SubtitleFormat,
    TranslationSettings,
)

__all__ = [
    "ConfigError",
    "AppConfig",
    "config_dir",
    "data_dir",
    "logs_dir",
    "cache_dir",
    "models_dir",
    "config_path",
    "load_config",
    "save_config",
    "default_config",
    "set_api_key",
    "get_api_key",
    "delete_api_key",
    "has_api_key",
    "mask_secret",
    "env_var_names",
]

CONFIG_FILENAME = "config.json"
KEYRING_SERVICE = APP_NAME


class ConfigError(RuntimeError):
    """Raised when the configuration file exists but cannot be used."""


def _base_dir(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw) if raw else fallback


def config_dir() -> Path:
    return _base_dir("APPDATA", Path.home() / ".config") / APP_NAME


def data_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / APP_NAME
    return config_dir() / "data"


def logs_dir() -> Path:
    return data_dir() / "logs"


def cache_dir() -> Path:
    return data_dir() / "cache"


def models_dir() -> Path:
    return data_dir() / "models"


def config_path() -> Path:
    return config_dir() / CONFIG_FILENAME


@dataclass(slots=True)
class AppConfig:
    """Everything the application persists between runs."""

    asr: AsrSettings = field(default_factory=AsrSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    log_level: str = "INFO"
    cache_transcripts: bool = True
    last_input_dir: str = ""
    last_output_dir: str = ""


def default_config() -> AppConfig:
    return AppConfig()


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def _asr_to_dict(settings: AsrSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "device": settings.device,
        "compute_type": settings.compute_type,
        "language": settings.language,
        "beam_size": settings.beam_size,
        "vad_filter": settings.vad_filter,
        "word_timestamps": settings.word_timestamps,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "no_speech_threshold": settings.no_speech_threshold,
        "log_prob_threshold": settings.log_prob_threshold,
        "hallucination_silence_threshold": settings.hallucination_silence_threshold,
        "initial_prompt": settings.initial_prompt,
        "batch_size": settings.batch_size,
    }


def _translation_to_dict(settings: TranslationSettings) -> dict[str, Any]:
    return {
        "enabled": settings.enabled,
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
        "target_language": settings.target_language,
        "temperature": settings.temperature,
        "window_chars": settings.window_chars,
        "max_retries": settings.max_retries,
        "timeout_s": settings.timeout_s,
        "extra_instructions": settings.extra_instructions,
    }


def _export_to_dict(settings: ExportSettings) -> dict[str, Any]:
    return {
        "output_dir": str(settings.output_dir) if settings.output_dir else "",
        "format": settings.format.value,
        "write_original": settings.write_original,
        "write_translated": settings.write_translated,
        "write_bilingual": settings.write_bilingual,
        "overwrite": settings.overwrite.value,
        "encoding": settings.encoding,
        "bom": settings.bom,
    }


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "version": 1,
        "log_level": config.log_level,
        "cache_transcripts": config.cache_transcripts,
        "last_input_dir": config.last_input_dir,
        "last_output_dir": config.last_output_dir,
        "asr": _asr_to_dict(config.asr),
        "translation": _translation_to_dict(config.translation),
        "export": _export_to_dict(config.export),
    }


def _coerce(enum_type: type, value: Any, fallback: Any) -> Any:
    try:
        return enum_type(value)
    except ValueError:
        return fallback


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    """Build an :class:`AppConfig`, ignoring unknown keys and bad values."""
    config = AppConfig()
    asr_raw = raw.get("asr") or {}
    for key, value in asr_raw.items():
        if hasattr(config.asr, key):
            setattr(config.asr, key, value)

    translation_raw = raw.get("translation") or {}
    for key, value in translation_raw.items():
        if hasattr(config.translation, key):
            setattr(config.translation, key, value)

    export_raw = raw.get("export") or {}
    explicit = {}
    for key, value in export_raw.items():
        if not hasattr(config.export, key):
            continue
        explicit[key] = value
    output_dir = explicit.pop("output_dir", "")
    fmt = _coerce(SubtitleFormat, explicit.pop("format", "srt"), SubtitleFormat.SRT)
    overwrite = _coerce(
        OverwritePolicy, explicit.pop("overwrite", "ask"), OverwritePolicy.ASK
    )
    for key, value in explicit.items():
        setattr(config.export, key, value)
    config.export.format = fmt
    config.export.overwrite = overwrite
    config.export.output_dir = Path(output_dir) if output_dir else None

    config.log_level = str(raw.get("log_level", config.log_level))
    config.cache_transcripts = bool(raw.get("cache_transcripts", config.cache_transcripts))
    config.last_input_dir = str(raw.get("last_input_dir", ""))
    config.last_output_dir = str(raw.get("last_output_dir", ""))
    return config


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #


def load_config(path: Path | None = None) -> AppConfig:
    """Load settings, returning defaults when no file exists yet."""
    target = Path(path) if path is not None else config_path()
    if not target.exists():
        return default_config()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read configuration {target}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration {target} must contain a JSON object")
    return config_from_dict(raw)


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Write settings atomically; returns the file that was written."""
    target = Path(path) if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config_to_dict(config), indent=2, ensure_ascii=False)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)
    return target


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


def env_var_names(provider: str) -> tuple[str, ...]:
    """Environment variables checked for a provider key, in priority order."""
    upper = provider.upper().replace("-", "_").replace(".", "_")
    names = [f"DOSUBS_{upper}_API_KEY"]
    if upper == "DEEPSEEK":
        names.append("DEEPSEEK_API_KEY")
    elif upper == "OPENAI":
        names.append("OPENAI_API_KEY")
    else:
        names.append(f"{upper}_API_KEY")
    return tuple(names)


def _keyring():  # pragma: no cover - thin wrapper, exercised via the public API
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def get_api_key(provider: str) -> str | None:
    """Environment variable first, then the Windows credential store."""
    for name in env_var_names(provider):
        value = os.environ.get(name)
        if value:
            return value.strip()
    keyring = _keyring()
    if keyring is None:
        return None
    try:
        secret = keyring.get_password(KEYRING_SERVICE, provider)
    except Exception:
        return None
    return secret.strip() if secret else None


def set_api_key(provider: str, key: str) -> None:
    """Store a key in the OS credential store.

    Raises:
        RuntimeError: if ``keyring`` is unavailable.
    """
    keyring = _keyring()
    if keyring is None:
        raise RuntimeError(
            "the 'keyring' package is not installed, so API keys cannot be stored "
            "securely; set DOSUBS_<PROVIDER>_API_KEY instead"
        )
    keyring.set_password(KEYRING_SERVICE, provider, key.strip())


def delete_api_key(provider: str) -> None:
    keyring = _keyring()
    if keyring is None:
        return
    try:
        keyring.delete_password(KEYRING_SERVICE, provider)
    except Exception:
        pass


def has_api_key(provider: str) -> bool:
    return bool(get_api_key(provider))


def mask_secret(secret: str | None) -> str:
    """``sk-abcdef1234`` -> ``sk-a...1234``; safe for logs and UI."""
    if not secret:
        return "(none)"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"
