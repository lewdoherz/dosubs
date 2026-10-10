"""Command line interface for DoSubs.

The CLI drives exactly the same pipeline as the GUI, which makes every capability
scriptable and is what the end-to-end tests exercise.
"""

from __future__ import annotations

import argparse
import getpass
import signal
import sys
import threading
from pathlib import Path

from . import APP_NAME, __version__
from .config import (
    AppConfig,
    ConfigError,
    config_path,
    delete_api_key,
    get_api_key,
    load_config,
    logs_dir,
    mask_secret,
    save_config,
    set_api_key,
)
from .gpu import probe as gpu_probe
from .logging_setup import get_logger, register_secret, setup_logging
from .media.probe import MediaError, expand_inputs, ffmpeg_path, ffprobe_path, have_ffmpeg
from .models import (
    PROVIDER_PRESETS,
    ExportKind,
    ExportSettings,
    JobResult,
    JobSpec,
    JobStatus,
    OverwritePolicy,
    ProgressEvent,
    Stage,
    SubtitleFormat,
    TranslationSettings,
)
from .pipeline import JobCancelled, Pipeline
from .subs.naming import plan_exports
from .translate.base import TranslationError
from .translate.openai_compat import build_provider

log = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dosubs",
        description="Transcribe video/audio speech and translate it into subtitle files.",
        epilog="Outputs: <name>.original.srt, <name>.<target>.srt and <name>.bilingual.srt",
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="video/audio files or folders")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")

    job = parser.add_argument_group("output")
    job.add_argument("-o", "--output-dir", type=Path, default=None, help="where to write subtitles")
    job.add_argument(
        "--format", choices=[f.value for f in SubtitleFormat], default=None, help="subtitle format"
    )
    job.add_argument(
        "--only",
        action="append",
        choices=[k.value for k in ExportKind],
        help="write only these files (repeatable); default: all three",
    )
    job.add_argument(
        "--overwrite",
        choices=[p.value for p in OverwritePolicy],
        default=None,
        help="what to do when a subtitle file already exists (default: ask)",
    )
    job.add_argument("--dry-run", action="store_true", help="probe and plan, write nothing")

    asr = parser.add_argument_group("speech recognition")
    asr.add_argument("--model", default=None, help="whisper model (tiny ... large-v3, large-v3-turbo)")
    asr.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    asr.add_argument("--cpu", action="store_true", help="shorthand for --device cpu")
    asr.add_argument(
        "--compute-type",
        dest="compute_type",
        choices=["auto", "float16", "int8_float16", "int8", "float32"],
        default=None,
    )
    asr.add_argument("--language", default=None, help="source language code; default: detect")
    asr.add_argument("--no-vad", action="store_true", help="disable voice activity detection")
    asr.add_argument("--no-word-timestamps", action="store_true", help="skip word-level timestamps")
    asr.add_argument("--beam-size", type=int, default=None)
    asr.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=">1 enables batched inference (faster, coarser raw segmentation)",
    )

    mt = parser.add_argument_group("translation")
    mt.add_argument("--no-translate", action="store_true", help="skip translation entirely")
    mt.add_argument("--target-language", default=None, help="target language tag (default: zh-CN)")
    mt.add_argument("--provider", choices=sorted(PROVIDER_PRESETS), default=None)
    mt.add_argument("--provider-model", dest="provider_model", default=None)
    mt.add_argument("--base-url", dest="base_url", default=None)
    mt.add_argument("--temperature", type=float, default=None)
    mt.add_argument("--test-provider", action="store_true", help="check provider connectivity and exit")

    keys = parser.add_argument_group("credentials")
    keys.add_argument("--set-key", metavar="PROVIDER", help="store an API key in the Windows credential store")
    keys.add_argument("--delete-key", metavar="PROVIDER", help="remove a stored API key")

    other = parser.add_argument_group("diagnostics")
    other.add_argument("--selfcheck", action="store_true", help="report environment and exit")
    other.add_argument(
        "--deep",
        action="store_true",
        help="with --selfcheck: load the model on the resolved device and run 1 s of audio",
    )
    other.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and do not write cached transcripts (forces fresh recognition)",
    )
    other.add_argument("--clear-cache", action="store_true", help="delete cached transcripts and exit")
    other.add_argument("--save-config", action="store_true", help="persist the given settings")
    other.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def _selfcheck(deep: bool) -> int:
    from . import gpu

    print(f"{APP_NAME} {__version__}")
    print(f"python          : {sys.version.split()[0]} ({sys.executable})")
    print(f"config file     : {config_path()}")
    print(f"log folder      : {logs_dir()}")
    print(f"cpu             : {gpu.cpu_name()}")

    if have_ffmpeg():
        print(f"ffmpeg          : {ffmpeg_path()}")
        print(f"ffprobe         : {ffprobe_path()}")
    else:
        print("ffmpeg          : MISSING - install it (winget install Gyan.FFmpeg) or drop "
              "ffmpeg.exe into the DoSubs bin folder")

    result = gpu.probe()
    print(f"cuda runtime    : {len(result.runtime_dirs)} wheel directorie(s) registered")
    print(f"device          : {result.summary()}")
    print(f"recommended     : {result.recommended_device()} / {result.recommended_compute_type()}")
    if not result.cuda_available and result.problem:
        print(f"acceleration    : {result.problem}")

    config = load_config()
    provider = config.translation.provider
    key = get_api_key(provider)
    print(f"translation     : {provider} / {config.translation.model} -> "
          f"{config.translation.target_language}")
    print(f"api key ({provider}): {mask_secret(key) if key else 'not set (--set-key ' + provider + ')'}")

    if deep:
        return _deep_check(config)
    return EXIT_OK


def _deep_check(config: AppConfig) -> int:
    """Load the model on the resolved device and transcribe one second of silence."""
    import numpy as np

    from .asr.engine import AsrError, FasterWhisperEngine

    engine = FasterWhisperEngine(config.asr)
    print(f"loading model   : {config.asr.model} ({config.asr.device}) ...")
    try:
        engine.load_model()
    except AsrError as exc:
        print(f"model load      : FAILED {exc}")
        return EXIT_FAILED
    print(f"model load      : ok on {engine.device} / {engine.compute_type}")

    try:
        result = engine.transcribe_array(np.zeros(16000, dtype="float32"))
    except AsrError as exc:
        print(f"inference       : FAILED {exc}")
        return EXIT_FAILED
    print(
        f"inference       : ok on {result.device}/{result.compute_type} "
        f"({result.elapsed_s:.2f}s for 1 s of audio)"
    )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Credentials and providers
# --------------------------------------------------------------------------- #


def _store_key(provider: str) -> int:
    if provider not in PROVIDER_PRESETS:
        print(f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDER_PRESETS))}")
        return EXIT_USAGE
    key = getpass.getpass(f"{provider} API key (input hidden): ").strip()
    if not key:
        print("no key entered; nothing stored")
        return EXIT_USAGE
    try:
        set_api_key(provider, key)
    except RuntimeError as exc:
        print(f"could not store the key: {exc}")
        return EXIT_FAILED
    print(f"stored {provider} key {mask_secret(key)} in the Windows credential store")
    return EXIT_OK


def _remove_key(provider: str) -> int:
    delete_api_key(provider)
    print(f"removed any stored key for {provider}")
    return EXIT_OK


def _test_provider(settings: TranslationSettings) -> int:
    try:
        provider = build_provider(settings)
    except TranslationError as exc:
        print(f"provider setup failed: {exc}")
        return EXIT_FAILED
    try:
        print(provider.check_connection())
    except TranslationError as exc:
        print(f"provider check failed: {exc}")
        return EXIT_FAILED
    finally:
        provider.close()
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Settings from arguments
# --------------------------------------------------------------------------- #


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.model:
        config.asr.model = args.model
    if args.cpu:
        config.asr.device = "cpu"
    elif args.device:
        config.asr.device = args.device
    if args.compute_type:
        config.asr.compute_type = args.compute_type
    if args.language:
        config.asr.language = args.language
    if args.no_vad:
        config.asr.vad_filter = False
    if args.no_word_timestamps:
        config.asr.word_timestamps = False
    if args.beam_size is not None:
        config.asr.beam_size = max(1, args.beam_size)
    if args.batch_size is not None:
        config.asr.batch_size = max(0, args.batch_size)

    if args.no_translate:
        config.translation.enabled = False
    if args.target_language:
        config.translation.target_language = args.target_language
    if args.provider:
        # The dataclass defaults describe DeepSeek, so switching provider must
        # carry the preset endpoint/model across.
        preset = PROVIDER_PRESETS[args.provider]
        if not args.provider_model:
            config.translation.model = preset.model
        if not args.base_url:
            config.translation.base_url = preset.base_url
        config.translation.provider = preset.name
    if args.provider_model:
        config.translation.model = args.provider_model
    if args.base_url:
        config.translation.base_url = args.base_url
    if args.temperature is not None:
        config.translation.temperature = args.temperature

    if args.output_dir:
        config.export.output_dir = args.output_dir
    if args.format:
        config.export.format = SubtitleFormat(args.format)
    if args.overwrite:
        config.export.overwrite = OverwritePolicy(args.overwrite)
    if args.only:
        kinds = {ExportKind(kind) for kind in args.only}
        config.export.write_original = ExportKind.ORIGINAL in kinds
        config.export.write_translated = ExportKind.TRANSLATED in kinds
        config.export.write_bilingual = ExportKind.BILINGUAL in kinds
    return config


def _build_specs(media_files: list[Path], config: AppConfig) -> list[JobSpec]:
    return [
        JobSpec(
            media=path,
            asr=config.asr,
            translation=config.translation,
            export=config.export,
        )
        for path in media_files
    ]


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #


class ConsoleReporter:
    """Single-line progress in a terminal, one line per finished file."""

    def __init__(self) -> None:
        self._last_stage: Stage | None = None
        self._current_file: Path | None = None

    def __call__(self, event: ProgressEvent) -> None:
        if self._current_file != event.message and event.stage is Stage.PROBE:
            self._current_file = None
        if event.stage is not self._last_stage:
            if self._last_stage is not None:
                sys.stdout.write("\n")
            self._last_stage = event.stage
        prefix = f"[{event.file_index}/{event.file_total}] {event.stage.label}"
        percent = f"{event.percent:5.1f}%" if event.percent is not None else "  ..."
        text = event.message if event.stage is Stage.TRANSCRIBE else ""
        line = f"\r{prefix}: {percent} {text[:60]:<60}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self) -> None:
        if self._last_stage is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._last_stage = None


def _prompt_overwrite(media: Path, conflicts: list[Path]) -> OverwritePolicy:
    print(f"\n{media.name}: {len(conflicts)} subtitle file(s) already exist, e.g. {conflicts[0].name}")
    while True:
        answer = input("  [o]verwrite, [s]kip, [r]ename, [a]bort? ").strip().lower()
        if answer in ("o", "overwrite"):
            return OverwritePolicy.OVERWRITE
        if answer in ("s", "skip", ""):
            return OverwritePolicy.SKIP
        if answer in ("r", "rename"):
            return OverwritePolicy.RENAME
        if answer in ("a", "abort"):
            raise KeyboardInterrupt


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")

    if args.set_key:
        return _store_key(args.set_key)
    if args.delete_key:
        return _remove_key(args.delete_key)
    if args.clear_cache:
        from .pipeline.cache import clear_cache

        removed, freed = clear_cache()
        print(f"removed {removed} cached transcript(s), reclaimed {freed / 1_048_576:.1f} MB")
        return EXIT_OK

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"configuration problem: {exc}")
        return EXIT_FAILED
    config = _apply_overrides(config, args)

    if args.selfcheck:
        return _selfcheck(args.deep)
    if args.test_provider:
        return _test_provider(config.translation)

    if not args.inputs:
        parser.print_help()
        return EXIT_USAGE

    try:
        media_files = expand_inputs(args.inputs)
    except MediaError as exc:
        print(f"input problem: {exc}")
        return EXIT_FAILED
    if not media_files:
        print("no supported media files found in the given inputs")
        return EXIT_USAGE

    if args.dry_run:
        # A dry run performs no recognition and no translation, so it must not
        # require a translation provider key.
        for path in media_files:
            plan = plan_exports(
                path,
                config.export.wanted_kinds(),
                target_language=config.translation.target_language,
                fmt=config.export.format,
                output_dir=config.export.output_dir,
                policy=OverwritePolicy.OVERWRITE,
            )
            print(f"  {path.name} -> " + ", ".join(t.path.name for t in plan.targets))
        return EXIT_OK

    if config.translation.enabled:
        key = get_api_key(config.translation.provider)
        if key:
            register_secret(key)
        needs_key = PROVIDER_PRESETS.get(config.translation.provider)
        if needs_key and needs_key.needs_key and not key:
            print(
                f"translation provider {config.translation.provider!r} needs an API key: run "
                f"'dosubs --set-key {config.translation.provider}' or pass --no-translate"
            )
            return EXIT_USAGE

    print(
        f"{APP_NAME} {__version__}: {len(media_files)} file(s), model {config.asr.model} "
        f"on {config.asr.device}, target {config.translation.target_language}"
    )

    if args.save_config:
        print(f"saved settings to {save_config(config)}")

    stop = threading.Event()

    def _handle_interrupt(signum, frame):  # noqa: ANN001 - signal handler signature
        if not stop.is_set():
            print("\ncancelling - finishing the current step ...")
        stop.set()

    try:
        signal.signal(signal.SIGINT, _handle_interrupt)
    except ValueError:  # pragma: no cover - not on the main thread
        pass

    reporter = ConsoleReporter()
    pipeline = Pipeline(
        config,
        on_progress=reporter,
        should_cancel=stop.is_set,
        resolve_conflicts=_prompt_overwrite if config.export.overwrite is OverwritePolicy.ASK else None,
        use_cache=not args.no_cache,
    )
    specs = _build_specs(media_files, config)
    try:
        results = pipeline.run(specs)
    except JobCancelled:
        reporter.finish()
        print("cancelled")
        return EXIT_CANCELLED
    except KeyboardInterrupt:
        reporter.finish()
        print("aborted")
        return EXIT_CANCELLED
    reporter.finish()

    return _print_summary(results, stop.is_set())


def _print_summary(results: list[JobResult], cancelled: bool) -> int:
    failures = 0
    for result in results:
        if result.status is JobStatus.DONE:
            print(f"OK      {result.media.name}: {len(result.written)} file(s)")
            for path in result.written:
                print(f"          {path}")
            if result.error:
                print(f"          note: {result.error}")
        elif result.status is JobStatus.SKIPPED:
            print(f"SKIP    {result.media.name}: {result.error or 'nothing to do'}")
        elif result.status is JobStatus.CANCELLED:
            print(f"CANCEL  {result.media.name}")
            failures += 1
        else:
            failures += 1
            print(f"FAILED  {result.media.name}: {result.error}")
            for path in result.written:
                print(f"          (kept) {path}")

    if cancelled:
        print("run cancelled; completed files were kept")
        return EXIT_CANCELLED
    if failures:
        print(f"{failures} of {len(results)} file(s) did not complete")
        return EXIT_FAILED
    print(f"{len(results)} file(s) processed successfully")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
