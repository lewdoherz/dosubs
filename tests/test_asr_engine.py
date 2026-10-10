"""Device selection and graceful degradation when CUDA is present but unusable.

A CUDA device count is not proof that inference will work: the runtime DLLs may be
missing (a packaged build without the CUDA component) or the driver may be too old for
the kernels that were built. ``--device auto`` must degrade to the CPU in that case;
an explicit ``--device cuda`` must fail loudly instead of silently running 20x slower.
"""

from __future__ import annotations

import numpy as np
import pytest

from dosubs.asr.engine import AsrError, AsrResult, FasterWhisperEngine
from dosubs.models import AsrSettings, Word

_CUBLAS_MESSAGE = "Library cublas64_12.dll is not found or cannot be loaded"


def _engine(device: str = "auto") -> FasterWhisperEngine:
    return FasterWhisperEngine(AsrSettings(model="tiny", device=device, word_timestamps=False))


def test_auto_device_falls_back_to_cpu_when_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine("auto")
    engine.device, engine.compute_type = "cuda", "float16"
    attempts: list[tuple[str, str]] = []

    def fake_run_once(source, *, label, duration_s=None, on_progress=None, should_cancel=None):
        attempts.append((engine.device, engine.compute_type))
        if len(attempts) == 1:
            raise RuntimeError(_CUBLAS_MESSAGE)
        return AsrResult(
            segments=[],
            language="en",
            device=engine.device,
            compute_type=engine.compute_type,
        )

    monkeypatch.setattr(engine, "_run_once", fake_run_once)
    result = engine._run(object(), label="fake")

    assert attempts == [("cuda", "float16"), ("cpu", "int8")]
    assert result.device == "cpu"
    assert result.compute_type == "int8"


def test_the_fallback_sticks_for_the_rest_of_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine("auto")
    engine.device, engine.compute_type = "cuda", "float16"
    monkeypatch.setattr(
        engine,
        "_run_once",
        lambda source, **kwargs: (_ for _ in ()).throw(RuntimeError(_CUBLAS_MESSAGE)),
    )
    with pytest.raises(AsrError):
        engine._run(object(), label="fake")

    assert engine._cuda_disabled is True
    assert engine._device_and_compute() == ("cpu", "int8")


def test_explicit_cuda_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine("cuda")
    engine.device, engine.compute_type = "cuda", "float16"
    calls = 0

    def fake_run_once(source, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(_CUBLAS_MESSAGE)

    monkeypatch.setattr(engine, "_run_once", fake_run_once)
    with pytest.raises(AsrError) as info:
        engine._run(object(), label="fake")

    assert calls == 1, "an explicit cuda request must not be retried on the CPU"
    assert "cuda" in str(info.value).lower()


def test_cpu_runs_never_try_to_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine("cpu")
    engine.device, engine.compute_type = "cpu", "int8"
    monkeypatch.setattr(
        engine, "_run_once", lambda source, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(AsrError, match="boom"):
        engine._run(object(), label="fake")


def test_asr_error_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine("auto")
    engine.device, engine.compute_type = "cpu", "int8"

    def fail(source, **kwargs):
        raise AsrError("model file is corrupt")

    monkeypatch.setattr(engine, "_run_once", fail)
    with pytest.raises(AsrError, match="model file is corrupt"):
        engine._run(object(), label="fake")


def test_cancellation_is_never_converted_into_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from dosubs.asr.engine import AsrCancelled

    engine = _engine("auto")
    engine.device, engine.compute_type = "cuda", "float16"
    calls = 0

    def fail(source, **kwargs):
        nonlocal calls
        calls += 1
        raise AsrCancelled("cancelled")

    monkeypatch.setattr(engine, "_run_once", fail)
    with pytest.raises(AsrCancelled):
        engine._run(object(), label="fake")
    assert calls == 1


def test_real_cpu_inference_still_works() -> None:
    """The refactor must not break the ordinary path."""
    engine = _engine("cpu")
    result = engine.transcribe_array(np.zeros(16000, dtype="float32"))
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert result.elapsed_s > 0


def test_word_conversion_keeps_timings_and_probabilities() -> None:
    from dosubs.asr.engine import _to_asr_segment

    class _Word:
        def __init__(self, word: str, start: float, end: float, probability: float) -> None:
            self.word, self.start, self.end, self.probability = word, start, end, probability

    class _Segment:
        start, end, text = 1.5, 3.25, " Hello there."
        words = (_Word(" Hello", 1.5, 2.0, 0.9), _Word(" there.", 2.0, 3.25, 0.8))
        no_speech_prob, avg_logprob = 0.01, -0.2

    segment = _to_asr_segment(_Segment())
    assert (segment.start_ms, segment.end_ms) == (1500, 3250)
    assert segment.text == " Hello there."
    assert segment.words == (
        Word(start_ms=1500, end_ms=2000, text=" Hello", probability=0.9),
        Word(start_ms=2000, end_ms=3250, text=" there.", probability=0.8),
    )
    assert segment.no_speech_prob == pytest.approx(0.01)
