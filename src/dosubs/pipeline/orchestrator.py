"""Processing orchestration: probe, recognise, segment, translate, export.

The orchestrator owns the job lifecycle and nothing else: engines, segmenters,
providers and exporters are all injected or imported behind their own interfaces, so
the GUI and the CLI drive the same code path.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from ..asr.engine import AsrCancelled, AsrError, AsrResult, FasterWhisperEngine
from ..config import AppConfig
from ..logging_setup import get_logger
from ..media.probe import MediaError, probe_media
from ..models import (
    AsrSettings,
    ExportKind,
    JobResult,
    JobSpec,
    JobStatus,
    OverwritePolicy,
    ProgressEvent,
    Stage,
    Transcript,
    TranslationSettings,
)
from ..segment.cue_builder import build_cues
from ..segment.profiles import profile_for
from ..subs.io import write_subtitle_file
from ..subs.naming import plan_exports
from ..subs.verify import verify_written_file
from ..translate.base import TranslationCancelled, TranslationError
from ..translate.batch import translate_transcript
from ..translate.openai_compat import build_provider
from .cache import load_cached_transcript, store_transcript

log = get_logger(__name__)

__all__ = ["JobCancelled", "Pipeline", "ProgressCallback", "ConflictResolver"]

ProgressCallback = Callable[[ProgressEvent], None]
CancelCallback = Callable[[], bool]
ConflictResolver = Callable[[Path, list[Path]], OverwritePolicy]

#: Below this automatic language-detection confidence we warn the user.
LANGUAGE_CONFIDENCE_WARNING = 0.7


class JobCancelled(RuntimeError):
    """Raised when a run was cancelled by the user."""


class Pipeline:
    """Runs jobs described by :class:`~dosubs.models.JobSpec`."""

    def __init__(
        self,
        config: AppConfig,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
        resolve_conflicts: ConflictResolver | None = None,
        verify_writes: bool = True,
        use_cache: bool = True,
        cache_root: Path | None = None,
        engine_factory: Callable[[AsrSettings], FasterWhisperEngine] | None = None,
        provider_factory: Callable[[TranslationSettings], object] | None = None,
    ) -> None:
        self.config = config
        self.on_progress = on_progress
        self.should_cancel = should_cancel
        self.resolve_conflicts = resolve_conflicts
        self.verify_writes = verify_writes
        self.use_cache = use_cache
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._engine_factory = engine_factory or FasterWhisperEngine
        self._provider_factory = provider_factory or build_provider
        self._engines: dict[tuple[object, ...], FasterWhisperEngine] = {}

    # -- helpers ------------------------------------------------------------ #

    def _emit(
        self,
        stage: Stage,
        message: str,
        fraction: float | None = None,
        file_index: int = 1,
        file_total: int = 1,
    ) -> None:
        if self.on_progress is None:
            return
        self.on_progress(
            ProgressEvent(
                stage=stage,
                message=message,
                fraction=fraction,
                file_index=file_index,
                file_total=file_total,
            )
        )

    def _cancelled(self) -> bool:
        return bool(self.should_cancel and self.should_cancel())

    def _engine(self, settings: AsrSettings) -> FasterWhisperEngine:
        key = (
            settings.model,
            settings.device,
            settings.compute_type,
            settings.batch_size,
            settings.language,
        )
        engine = self._engines.get(key)
        if engine is None:
            engine = self._engine_factory(settings)
            self._engines[key] = engine
        return engine

    # -- job execution ------------------------------------------------------ #

    def run(self, specs: Sequence[JobSpec]) -> list[JobResult]:
        """Run every job in order, isolating failures from each other."""
        results: list[JobResult] = []
        total = len(specs)
        for index, spec in enumerate(specs, start=1):
            if self._cancelled():
                results.append(
                    JobResult(media=spec.media, status=JobStatus.CANCELLED, error="cancelled")
                )
                continue
            self._emit(Stage.PROBE, f"[{index}/{total}] {spec.media.name}", 0.0, index, total)
            try:
                result = self.run_job(spec, file_index=index, file_total=total)
            except JobCancelled:
                result = JobResult(
                    media=spec.media, status=JobStatus.CANCELLED, error="cancelled by user"
                )
            except Exception as exc:  # never let one file kill the batch
                log.exception("unexpected failure while processing %s", spec.media)
                result = JobResult(
                    media=spec.media,
                    status=JobStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
        return results

    def run_job(
        self, spec: JobSpec, *, file_index: int = 1, file_total: int = 1
    ) -> JobResult:
        """Process one file end to end."""
        media = Path(spec.media)
        result = JobResult(media=media, status=JobStatus.RUNNING)

        # 1. probe ---------------------------------------------------------- #
        self._emit(Stage.PROBE, f"probing {media.name}", 0.05, file_index, file_total)
        try:
            info = probe_media(media)
        except MediaError as exc:
            result.status = JobStatus.FAILED
            result.error = str(exc)
            return result
        if not info.has_audio:
            result.status = JobStatus.FAILED
            result.error = f"{media.name} has no audio stream to transcribe"
            return result

        # 2. transcribe ----------------------------------------------------- #
        engine = self._engine(spec.asr)
        cached = (
            load_cached_transcript(media, spec.asr, root=self.cache_root)
            if self.use_cache
            else None
        )
        if cached is not None:
            log.info(
                "reusing the cached transcript for %s (%d segments); recognition skipped",
                media.name,
                len(cached.segments),
            )
            self._emit(
                Stage.TRANSCRIBE,
                f"reusing cached transcript for {media.name}",
                0.65,
                file_index,
                file_total,
            )
            asr = AsrResult(
                segments=list(cached.segments),
                language=cached.language,
                language_probability=cached.language_probability,
                language_forced=cached.language_forced,
                duration_s=cached.duration_s or info.duration_s,
                model=spec.asr.model,
                device="cache",
                compute_type="",
                elapsed_s=0.0,
            )
        else:
            self._emit(
                Stage.TRANSCRIBE,
                f"recognising speech in {media.name}",
                0.1,
                file_index,
                file_total,
            )
            try:
                asr = engine.transcribe(
                    media,
                    duration_s=info.duration_s,
                    on_progress=lambda fraction, text: self._emit(
                        Stage.TRANSCRIBE,
                        text,
                        0.1 + 0.55 * fraction,
                        file_index,
                        file_total,
                    ),
                    should_cancel=self.should_cancel,
                )
            except AsrCancelled:
                raise JobCancelled("transcription cancelled") from None
            except AsrError as exc:
                result.status = JobStatus.FAILED
                result.error = str(exc)
                return result
            if self.use_cache:
                store_transcript(
                    media,
                    spec.asr,
                    segments=asr.segments,
                    language=asr.language,
                    language_probability=asr.language_probability,
                    language_forced=asr.language_forced,
                    duration_s=asr.duration_s,
                    root=self.cache_root,
                )

        # 3. segment -------------------------------------------------------- #
        self._emit(Stage.SEGMENT, "building subtitle cues", 0.7, file_index, file_total)
        effective_language = spec.asr.language or asr.language
        profile = profile_for(effective_language)
        if ExportKind.BILINGUAL in spec.export.wanted_kinds():
            # One line per language keeps the bilingual file to two lines per cue.
            profile = replace(
                profile, max_lines=1, max_chars_per_cue=profile.max_chars_per_line
            )

        cues = build_cues(asr.segments, profile=profile)
        transcript = Transcript(
            media=info,
            cues=cues,
            source_language=effective_language,
            language_probability=asr.language_probability,
            language_forced=asr.language_forced,
            asr_params=spec.asr.cache_key(),
        )
        if (
            not asr.language_forced
            and asr.language_probability is not None
            and asr.language_probability < LANGUAGE_CONFIDENCE_WARNING
        ):
            transcript.warnings.append(
                f"language detection was uncertain ({asr.language} at "
                f"{asr.language_probability:.0%}); set the source language explicitly if the "
                "transcript looks wrong"
            )
        for problem in transcript.validate_timeline():
            transcript.warnings.append(f"timeline: {problem}")
            log.warning("%s: %s", media.name, problem)

        if not transcript.cues:
            result.status = JobStatus.SKIPPED
            result.error = "no speech detected; nothing was written"
            self._emit(Stage.SEGMENT, result.error, 1.0, file_index, file_total)
            return result

        # 4. translate ------------------------------------------------------ #
        wanted = list(spec.export.wanted_kinds())
        translation_error: str | None = None
        translation_stats = None
        if spec.translation.enabled:
            if _same_language(effective_language, spec.translation.target_language):
                transcript.warnings.append(
                    f"source and target language are both {spec.translation.target_language}; "
                    "translation was skipped"
                )
                wanted = [kind for kind in wanted if kind is ExportKind.ORIGINAL]
            else:
                self._emit(Stage.TRANSLATE, "translating subtitles", 0.75, file_index, file_total)
                provider = None
                try:
                    provider = self._provider_factory(spec.translation)
                    translation_stats = translate_transcript(
                        transcript,
                        provider,
                        settings=spec.translation,
                        progress=lambda done, count: self._emit(
                            Stage.TRANSLATE,
                            f"translated {done}/{count} cues",
                            0.75 + 0.15 * (done / max(1, count)),
                            file_index,
                            file_total,
                        ),
                        should_cancel=self.should_cancel,
                        on_window_error=lambda index, exc: log.warning(
                            "translation window %d failed: %s", index, exc
                        ),
                    )
                    missing = transcript.untranslated_ids()
                    if missing:
                        translation_error = (
                            f"{len(missing)} cue(s) have no translation "
                            f"(first: {missing[0]})"
                        )
                except TranslationCancelled:
                    raise JobCancelled("translation cancelled") from None
                except TranslationError as exc:
                    translation_error = str(exc)
                    log.error("translation failed for %s: %s", media.name, exc)
                finally:
                    if provider is not None:
                        try:
                            provider.close()
                        except Exception:  # pragma: no cover - closing is best effort
                            pass
                if translation_error:
                    # Keep the original subtitles: partial results are still useful.
                    wanted = [kind for kind in wanted if kind is ExportKind.ORIGINAL]
        else:
            wanted = [kind for kind in wanted if kind is ExportKind.ORIGINAL]

        # 5. export --------------------------------------------------------- #
        plan = plan_exports(
            media,
            wanted,
            target_language=spec.translation.target_language,
            fmt=spec.export.format,
            output_dir=spec.export.output_dir,
            policy=spec.export.overwrite,
        )
        if plan.conflicts:
            policy = (
                self.resolve_conflicts(media, list(plan.conflicts))
                if self.resolve_conflicts
                else OverwritePolicy.SKIP
            )
            if policy is OverwritePolicy.ASK:
                policy = OverwritePolicy.SKIP
            plan = plan_exports(
                media,
                wanted,
                target_language=spec.translation.target_language,
                fmt=spec.export.format,
                output_dir=spec.export.output_dir,
                policy=policy,
            )
            if plan.skipped:
                transcript.warnings.append(
                    f"left {len(plan.skipped)} existing file(s) untouched (overwrite policy: "
                    f"{policy.value})"
                )

        self._emit(Stage.EXPORT, "writing subtitle files", 0.95, file_index, file_total)
        verification_problems: list[str] = []
        for target in plan.targets:
            kind = target.kind
            path = write_subtitle_file(
                transcript,
                target.path,
                kind,
                target_language=spec.translation.target_language,
                fmt=spec.export.format,
                encoding=spec.export.encoding,
                bom=spec.export.bom,
            )
            result.written.append(path)
            if self.verify_writes:
                problems = verify_written_file(
                    path,
                    transcript,
                    kind,
                    expect_translation=kind in (ExportKind.TRANSLATED, ExportKind.BILINGUAL),
                )
                for problem in problems:
                    log.error("%s", problem)
                verification_problems.extend(problems)
        result.skipped.extend(target.path for target in plan.skipped)

        # 6. verdict -------------------------------------------------------- #
        problems = list(transcript.warnings)
        if verification_problems:
            problems = verification_problems + problems
        if translation_error:
            problems.insert(0, f"translation failed: {translation_error}")
        result.status = (
            JobStatus.FAILED if (translation_error or verification_problems) else JobStatus.DONE
        )
        result.error = "; ".join(problems) if problems else None
        if translation_stats is not None:
            log.info(
                "%s: %d cues, %d translation windows%s",
                media.name,
                len(transcript.cues),
                translation_stats.windows,
                f", {len(problems)} warning(s)" if problems else "",
            )
        self._emit(Stage.EXPORT, f"finished {media.name}", 1.0, file_index, file_total)
        return result


def _same_language(source: str | None, target: str | None) -> bool:
    if not source or not target:
        return False
    return source.split("-")[0].lower() == target.split("-")[0].lower()
