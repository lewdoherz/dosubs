"""CUDA runtime bootstrap and device selection.

The bootstrap behaviour is asserted here because it is the difference between a
working GPU path and a cryptic ``cublas64_12.dll is not found`` error: on Windows,
CTranslate2 adds only its own package directory to the DLL search path, so the
CUDA libraries that ship in the ``nvidia-*-cu12`` wheels have to be made findable
by the application.
"""

from __future__ import annotations

import os

import pytest

from dosubs import gpu
from dosubs.models import AsrSettings

requires_cuda = pytest.mark.skipif(gpu.probe().cuda_available is False, reason="CUDA is unavailable")


def test_wheel_runtime_directories_are_discovered() -> None:
    dirs = gpu.nvidia_runtime_dirs()
    assert dirs, "expected the nvidia-*-cu12 wheels to be installed in the dev venv"
    components = {path.parent.name for path in dirs}
    assert "cublas" in components or "cudnn" in components


def test_bootstrap_is_idempotent_and_puts_dirs_on_path() -> None:
    first = gpu.bootstrap()
    second = gpu.bootstrap()
    assert first == second
    if first:
        assert str(first[0]) in (os.environ.get("PATH") or "")


def test_app_local_cuda_folder_is_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A packaged build finds the optional _cuda folder through DOSUBS_CUDA_DIRS."""
    flat = tmp_path / "_cuda"
    (flat / "cublas" / "bin").mkdir(parents=True)
    (flat / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"stub")
    (flat / "cudnn" / "bin").mkdir(parents=True)
    (flat / "cudnn" / "bin" / "cudnn64_9.dll").write_bytes(b"stub")

    monkeypatch.setenv("DOSUBS_CUDA_DIRS", str(flat))
    dirs = gpu.nvidia_runtime_dirs()
    assert flat / "cublas" / "bin" in dirs
    assert flat / "cudnn" / "bin" in dirs

    # An empty directory must not be reported as a usable runtime.
    (flat / "cublas" / "bin" / "cublas64_12.dll").unlink()
    assert flat / "cublas" / "bin" not in gpu.nvidia_runtime_dirs()


def test_multiple_cuda_dirs_can_be_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    for root in (first, second):
        (root / "cublas" / "bin").mkdir(parents=True)
        (root / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"stub")
    monkeypatch.setenv("DOSUBS_CUDA_DIRS", os.pathsep.join([str(first), str(second)]))
    dirs = gpu.nvidia_runtime_dirs()
    assert first / "cublas" / "bin" in dirs
    assert second / "cublas" / "bin" in dirs


def test_resolve_device_prefers_cpu_when_asked() -> None:
    assert gpu.resolve_device("cpu") == "cpu"


def test_resolve_device_auto_degrades_to_cpu_without_cuda() -> None:
    probe = gpu.GpuProbe(cuda_available=False, problem="no driver")
    assert gpu.resolve_device("auto", probe) == "cpu"


def test_resolve_device_auto_uses_cuda_when_available() -> None:
    probe = gpu.GpuProbe(cuda_available=True, device_count=1, compute_types=("float16",))
    assert gpu.resolve_device("auto", probe) == "cuda"


def test_explicit_cuda_failure_is_actionable() -> None:
    probe = gpu.GpuProbe(cuda_available=False, problem="no CUDA device reported")
    with pytest.raises(gpu.AccelerationUnavailable) as info:
        gpu.resolve_device("cuda", probe)
    assert "no CUDA device reported" in str(info.value)
    assert "CUDA" in str(info.value)


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        gpu.resolve_device("tpu", gpu.GpuProbe(cuda_available=False))


def test_compute_type_resolution_for_cpu() -> None:
    assert gpu.resolve_compute_type("auto", "cpu") == "int8"
    assert gpu.resolve_compute_type("float32", "cpu") == "float32"
    assert gpu.resolve_compute_type("float16", "cpu") == "int8"


def test_compute_type_resolution_for_cuda() -> None:
    probe = gpu.GpuProbe(cuda_available=True, compute_types=("float16", "int8_float16"))
    assert gpu.resolve_compute_type("auto", "cuda", probe) == "float16"
    assert gpu.resolve_compute_type("int8_float16", "cuda", probe) == "int8_float16"
    # An unsupported request degrades to something the device can actually run.
    assert gpu.resolve_compute_type("float32", "cuda", probe) == "float16"


def test_probe_reports_consistent_state() -> None:
    result = gpu.probe()
    assert result.recommended_device() in ("cuda", "cpu")
    if result.cuda_available:
        assert result.compute_types
        assert "CUDA available" in result.summary()
    else:
        assert result.recommended_device() == "cpu"
        assert "CPU only" in result.summary()


@requires_cuda
@pytest.mark.integration
def test_cuda_is_actually_usable_on_this_workstation() -> None:
    """A device count is not proof; this runs real inference (marked integration)."""
    import numpy as np

    from dosubs.asr.engine import FasterWhisperEngine

    engine = FasterWhisperEngine(AsrSettings(model="tiny", device="cuda", word_timestamps=False))
    result = engine.transcribe_array(np.zeros(16000, dtype="float32"))
    assert engine.device == "cuda"
    assert result.elapsed_s > 0
