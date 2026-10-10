"""faster-whisper speech recognition engine.

Wraps model loading, CUDA bootstrap, voice activity detection, language detection,
word-level timestamps, progress reporting and cancellation. The engine hands raw
:class:`~dosubs.models.AsrSegment` objects to the segmenter, so nothing downstream
depends on faster-whisper types.

Models are cached per ``(model, device, compute_type)`` for the lifetime of the
process: a batch of twenty videos loads the weights once, and the cache is the
reason batch runs do not pay the ~4-8 s load cost per file.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .. import gpu
from ..logging_setup import get_logger
from ..models import AsrSegment, AsrSettings, Word

log = get_logger(__name__)

__all__ = [
    "AsrError",
    "AsrCancelled",
    "AsrResult",
    "FasterWhisperEngine",
    "engine_for",
    "unload_models",
]

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


class AsrError(RuntimeError):
    """Speech recognition failed for a reason the user should see."""


class AsrCancelled(AsrError):
    """The caller asked to stop; no partial transcript is returned."""


@dataclass(slots=True)
class AsrResult:
    """Raw recogniser output for one media file."""

    segments: list[AsrSegment] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    language_forced: bool = False
    duration_s: float = 0.0
    model: str = ""
    device: str = ""
    compute_type: str = ""
    elapsed_s: float = 0.0


_cache_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str], object] = {}


def unload_models() -> None:
    """Drop cached models (used by tests and by an explicit GUI 'unload')."""
    with _cache_lock:
        _model_cache.clear()


class FasterWhisperEngine:
    """Transcribe media with faster-whisper, one engine per configuration."""

    def __init__(self, settings: AsrSettings, *, probe: gpu.GpuProbe | None = None) -> None:
        self.settings = settings
        self._probe = probe
        self.device = ""
        self.compute_type = ""
        self._cuda_disabled = False

    # -- model lifecycle ---------------------------------------------------- #

    def _device_and_compute(self) -> tuple[str, str]:
        if self._cuda_disabled:
            return "cpu", "int8"
        probe = self._probe or gpu.probe()
        device = gpu.resolve_device(self.settings.device, probe)
        compute = gpu.resolve_compute_type(self.settings.compute_type, device, probe)
        return device, compute

    def load_model(self):
        """Load (or reuse) the underlying model and return it."""
        from faster_whisper import WhisperModel  # imported late: CUDA bootstrap first

        gpu.bootstrap()
        device, compute = self._device_and_compute()
        self.device, self.compute_type = device, compute
        key = (self.settings.model, device, compute)
        with _cache_lock:
            cached = _model_cache.get(key)
            if cached is not None:
                log.debug("reusing loaded model %s on %s (%s)", *key)
                return cached

        log.info(
            "loading faster-whisper model %r on %s (%s)", self.settings.model, device, compute
        )
        try:
            model = WhisperModel(self.settings.model, device=device, compute_type=compute)
        except Exception as exc:  # surfaced to the user with context
            raise AsrError(
                f"could not load the {self.settings.model!r} model on {device} ({compute}): {exc}"
            ) from exc
        with _cache_lock:
            _model_cache[key] = model
        return model

    def close(self) -> None:
        """Release this engine's cached model."""
        device, compute = self.device, self.compute_type
        if not device:
            return
        with _cache_lock:
            _model_cache.pop((self.settings.model, device, compute), None)

    # -- transcription ------------------------------------------------------ #

    def transcribe(
        self,
        media: Path,
        *,
        duration_s: float | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AsrResult:
        """Transcribe ``media`` and return raw segments.

        ``should_cancel`` is polled once per decoded segment; cancelling raises
        :class:`AsrCancelled` and returns no partial result, so the caller never
        writes a truncated subtitle file.
        """
        return self._run(
            str(media),
            label=Path(media).name,
            duration_s=duration_s,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    def transcribe_array(
        self,
        audio: "object",
        *,
        sample_rate: int = 16000,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AsrResult:
        """Transcribe an in-memory float32 mono signal.

        Used by ``dosubs --selfcheck --deep`` to prove that the resolved device can
        actually run inference, rather than only that a library reported a device.
        """
        import numpy as np

        samples = np.asarray(audio, dtype="float32")
        return self._run(
            samples,
            label="<memory>",
            duration_s=samples.size / float(sample_rate),
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    def _run(
        self,
        source: object,
        *,
        label: str,
        duration_s: float | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AsrResult:
        """Run once, and on ``auto`` retry once on the CPU if CUDA fails at runtime.

        A device count is not proof that the CUDA runtime is usable: on Windows the
        libraries may be missing (a packaged build without the CUDA component) or the
        driver may be too old for the built kernels. Since the user asked for
        ``auto``, that must degrade to CPU work rather than fail the job; an explicit
        ``cuda`` request still fails loudly.
        """
        try:
            return self._call_once(
                source,
                label=label,
                duration_s=duration_s,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
        except AsrCancelled:
            raise
        except AsrError as exc:
            if not self._can_fall_back():
                raise
            log.warning(
                "%s/%s failed (%s: %s); retrying on the CPU. Install the CUDA component or "
                "'pip install -e .[cuda]' to use the GPU.",
                self.device,
                self.compute_type,
                type(exc).__name__,
                exc,
            )
            self._disable_cuda()
            return self._call_once(
                source,
                label=label,
                duration_s=duration_s,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )

    def _call_once(
        self,
        source: object,
        *,
        label: str,
        duration_s: float | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AsrResult:
        """One attempt, with engine-level failures turned into a user-facing error."""
        try:
            return self._run_once(
                source,
                label=label,
                duration_s=duration_s,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
        except AsrCancelled:
            raise
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError(
                f"transcription failed on {self.device} ({self.compute_type}): {exc}"
            ) from exc

    def _can_fall_back(self) -> bool:
        """Only an ``auto`` device may degrade; ``cuda`` must fail loudly."""
        return self.settings.device.lower() == "auto" and self.device == "cuda"

    def _disable_cuda(self) -> None:
        """Pin this engine to the CPU for the rest of the run."""
        self.device, self.compute_type = "cpu", "int8"
        self._cuda_disabled = True

    def _run_once(
        self,
        source: object,
        *,
        label: str,
        duration_s: float | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AsrResult:
        model = self.load_model()
        settings = self.settings
        started = time.monotonic()

        kwargs: dict[str, object] = {
            "beam_size": settings.beam_size,
            "vad_filter": settings.vad_filter,
            "condition_on_previous_text": settings.condition_on_previous_text,
            "no_speech_threshold": settings.no_speech_threshold,
            "log_prob_threshold": settings.log_prob_threshold,
            "word_timestamps": settings.word_timestamps,
            "language": settings.language,
        }
        if settings.initial_prompt:
            kwargs["initial_prompt"] = settings.initial_prompt
        if settings.hallucination_silence_threshold is not None:
            kwargs["hallucination_silence_threshold"] = settings.hallucination_silence_threshold

        if settings.batch_size and settings.batch_size > 1:
            from faster_whisper import BatchedInferencePipeline

            pipeline = BatchedInferencePipeline(model=model)
            kwargs["batch_size"] = settings.batch_size
            log.info("using batched inference with batch_size=%d", settings.batch_size)
            segments_iter, info = pipeline.transcribe(source, **kwargs)
        else:
            segments_iter, info = model.transcribe(source, **kwargs)

        total_s = float(duration_s or getattr(info, "duration", 0.0) or 0.0)
        collected: list[AsrSegment] = []
        try:
            for segment in segments_iter:
                if should_cancel is not None and should_cancel():
                    raise AsrCancelled("transcription cancelled")
                collected.append(_to_asr_segment(segment))
                if on_progress is not None:
                    end_s = float(getattr(segment, "end", 0.0) or 0.0)
                    fraction = min(0.999, end_s / total_s) if total_s > 0 else None
                    on_progress(fraction if fraction is not None else 0.0, segment.text.strip())
        finally:
            close = getattr(segments_iter, "close", None)
            if callable(close):  # release the decoder even when we bail out early
                try:
                    close()
                except Exception:  # pragma: no cover - generator close is best effort
                    pass

        elapsed = time.monotonic() - started
        result = AsrResult(
            segments=collected,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            language_forced=bool(settings.language),
            duration_s=total_s,
            model=settings.model,
            device=self.device,
            compute_type=self.compute_type,
            elapsed_s=elapsed,
        )
        log.info(
            "transcribed %s in %.1fs: %d segments, language %s (%.2f), device %s/%s",
            label,
            elapsed,
            len(collected),
            result.language,
            result.language_probability or 0.0,
            self.device,
            self.compute_type,
        )
        return result


def _to_asr_segment(segment: object) -> AsrSegment:
    """Convert a faster-whisper segment into our engine-agnostic form."""
    raw_words: Iterable[object] = getattr(segment, "words", None) or ()
    words: list[Word] = []
    for word in raw_words:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        text = getattr(word, "word", "") or ""
        if start is None or end is None:
            continue
        words.append(
            Word(
                start_ms=int(round(float(start) * 1000)),
                end_ms=int(round(float(end) * 1000)),
                text=str(text),
                probability=float(getattr(word, "probability", 1.0) or 1.0),
            )
        )
    return AsrSegment(
        start_ms=int(round(float(getattr(segment, "start", 0.0)) * 1000)),
        end_ms=int(round(float(getattr(segment, "end", 0.0)) * 1000)),
        text=str(getattr(segment, "text", "") or ""),
        words=tuple(words),
        no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
        avg_logprob=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
    )


def engine_for(settings: AsrSettings) -> FasterWhisperEngine:
    return FasterWhisperEngine(settings)
