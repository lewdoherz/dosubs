"""CUDA runtime bootstrap and device discovery.

CTranslate2 4.8.2 on Windows only adds its own package directory to the DLL search
path, so the CUDA libraries that ship in the ``nvidia-*-cu12`` wheels are invisible
until the process makes them findable. Measured on this workstation (RTX 5080,
CTranslate2 4.8.2, Python 3.14):

* ``os.add_dll_directory`` alone -> ``Library cublas64_12.dll is not found``
* prepending the wheel ``bin`` directories to ``PATH`` -> works
* ``add_dll_directory`` plus an explicit ``ctypes.CDLL`` preload -> works

:func:`bootstrap` therefore does all three, and must run *before* ctranslate2 or
faster_whisper is imported.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "AccelerationUnavailable",
    "DeviceInfo",
    "GpuProbe",
    "nvidia_runtime_dirs",
    "bootstrap",
    "probe",
    "cpu_name",
    "resolve_device",
]

#: Wheel layout: ``site-packages/nvidia/<component>/bin/*.dll``.
_NVIDIA_SUBDIRS = (
    ("cublas", "bin"),
    ("cudnn", "bin"),
    ("cuda_nvrtc", "bin"),
    ("cuda_runtime", "bin"),
)

#: Load order matters: cuBLAS/Lt before cuDNN, both after the CUDA runtime.
_PRELOAD_ORDER = (
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "nvrtc64_120_0.dll",
    "cudnn64_9.dll",
)

_bootstrapped: list[Path] = []


class AccelerationUnavailable(RuntimeError):
    """Raised when CUDA was requested but cannot be used on this machine."""


def _extra_runtime_roots() -> list[Path]:
    """Locations that hold CUDA libraries outside ``sys.path``.

    A packaged build cannot reach the site-packages wheel layout, so the CUDA
    runtime is shipped as an optional ``_cuda`` folder next to the executable, and
    ``DOSUBS_CUDA_DIRS`` lets a user point at a system CUDA install instead.
    """
    roots: list[Path] = []
    env = os.environ.get("DOSUBS_CUDA_DIRS", "")
    for entry in env.split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry.strip()))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent / "_cuda")
    return roots


def nvidia_runtime_dirs() -> list[Path]:
    """Return existing CUDA runtime directories.

    Two layouts are covered: the ``site-packages/nvidia/<component>/bin`` layout of the
    ``nvidia-*-cu12`` wheels, and a flat ``<root>/<component>/bin`` layout used by the
    packaged app's optional CUDA folder.
    """
    candidates: list[Path] = []
    for entry in sys.path:
        if entry:
            candidates.append(Path(entry) / "nvidia")
    for root in _extra_runtime_roots():
        candidates.append(root)
        candidates.append(root / "nvidia")

    found: list[Path] = []
    seen: set[str] = set()
    for base in candidates:
        for component, sub in _NVIDIA_SUBDIRS:
            candidate = base / component / sub
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_dir() and any(candidate.glob("*.dll")):
                found.append(candidate)
    return found


def bootstrap() -> list[Path]:
    """Make the pip-installed CUDA runtime loadable. Idempotent."""
    global _bootstrapped
    if _bootstrapped:
        return _bootstrapped

    dirs = nvidia_runtime_dirs()
    if not dirs:
        return []

    if sys.platform == "win32":
        for directory in dirs:
            # PATH is what CTranslate2's implicit LoadLibrary ends up using.
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass
        for name in _PRELOAD_ORDER:
            try:
                ctypes.CDLL(name)
            except OSError:
                pass

    _bootstrapped = dirs
    return dirs


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One usable execution device."""

    name: str
    device: str  # "cuda" | "cpu"
    compute_types: tuple[str, ...] = ()

    def supports(self, compute_type: str) -> bool:
        return compute_type in self.compute_types


@dataclass(frozen=True, slots=True)
class GpuProbe:
    """Result of looking for usable acceleration."""

    cuda_available: bool
    device_count: int = 0
    device_names: tuple[str, ...] = ()
    compute_types: tuple[str, ...] = ()
    runtime_dirs: tuple[Path, ...] = ()
    detail: str = ""
    problem: str | None = None

    def recommended_device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"

    def recommended_compute_type(self) -> str:
        if not self.cuda_available:
            return "int8"
        for candidate in ("float16", "int8_float16", "float32"):
            if candidate in self.compute_types:
                return candidate
        return "float16"

    def summary(self) -> str:
        if self.cuda_available:
            names = ", ".join(self.device_names) or f"{self.device_count} CUDA device(s)"
            return f"CUDA available: {names} [{', '.join(self.compute_types)}]"
        return f"CUDA unavailable, CPU only{': ' + self.problem if self.problem else ''}"


def _smi_gpu_names() -> tuple[str, ...]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ()
    try:
        completed = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def cpu_name() -> str:
    return f"{platform.processor() or 'CPU'} ({os.cpu_count()} threads)"


def probe() -> GpuProbe:
    """Detect usable devices. Never raises: failures are reported, not thrown."""
    runtime_dirs = bootstrap()
    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - broken install
        return GpuProbe(
            cuda_available=False,
            runtime_dirs=tuple(runtime_dirs),
            detail="ctranslate2 is not importable",
            problem=f"{type(exc).__name__}: {exc}",
        )

    compute_types: tuple[str, ...] = ()
    problem: str | None = None
    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        count = 0
        problem = f"{type(exc).__name__}: {exc}"

    if count:
        try:
            supported = tuple(ctranslate2.get_supported_compute_types("cuda"))
            compute_types = tuple(
                name for name in ("float16", "int8_float16", "int8", "float32") if name in supported
            )
        except Exception as exc:
            count = 0
            problem = f"{type(exc).__name__}: {exc}"

    if count == 0 and problem is None:
        problem = explain_missing_cuda()

    return GpuProbe(
        cuda_available=count > 0,
        device_count=count,
        device_names=_smi_gpu_names(),
        compute_types=compute_types,
        runtime_dirs=tuple(runtime_dirs),
        detail="ctranslate2 CUDA backend" if count else "no CUDA device reported",
        problem=problem,
    )


def explain_missing_cuda() -> str:
    if not shutil.which("nvidia-smi") and not nvidia_runtime_dirs():
        return (
            "no NVIDIA driver and no CUDA runtime libraries were found; "
            "transcription will run on the CPU"
        )
    return (
        "an NVIDIA driver or CUDA device is present but CTranslate2 could not use it; "
        "install the CUDA extras (pip install -e .[cuda]) and check that the driver "
        "reports a device via nvidia-smi"
    )


def resolve_device(requested: str, probe_result: GpuProbe | None = None) -> str:
    """Turn a user device preference into a concrete device, with a clear error.

    ``"auto"`` degrades to CPU when CUDA is unusable; an explicit ``"cuda"`` fails
    loudly instead of silently running 20x slower.
    """
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    result = probe_result or probe()
    if requested == "cuda":
        if not result.cuda_available:
            raise AccelerationUnavailable(
                "CUDA was requested but is unavailable: "
                + (result.problem or "no CUDA device reported")
            )
        return "cuda"
    if requested not in ("auto",):
        raise ValueError(f"unknown device {requested!r}; expected auto, cuda or cpu")
    return result.recommended_device()


def resolve_compute_type(requested: str, device: str, probe_result: GpuProbe | None = None) -> str:
    """Pick a compute type that the chosen device actually supports."""
    requested = (requested or "auto").lower()
    if device == "cpu":
        if requested in ("auto", "int8", "float32"):
            return "int8" if requested == "auto" else requested
        return "int8"
    result = probe_result or probe()
    available = result.compute_types or ("float16",)
    if requested == "auto":
        return result.recommended_compute_type()
    if requested in available:
        return requested
    fallback = result.recommended_compute_type()
    return fallback
