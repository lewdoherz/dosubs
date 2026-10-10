# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DoSubs.

Built as a one-directory bundle on purpose:

* no UPX - compressed Qt/CTranslate2 binaries are fragile and slow to start;
* the CUDA runtime is *not* bundled (about 1.9 GB of DLLs, one of which is 638 MB);
  it ships as a separate optional component installed into ``_cuda`` next to the
  executable, which ``dosubs.gpu`` discovers on start;
* speech models are never bundled - they are downloaded on first use into the user's
  Hugging Face cache.

Build with:  packaging/build.ps1
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

project_root = Path(SPECPATH).resolve().parent
src = project_root / "src"

# faster-whisper ships its Silero VAD model as package data
# (faster_whisper/assets/silero_vad_v6.onnx). Without it every run with voice
# activity detection fails at start-up with an ONNXRuntime NO_SUCHFILE error.
data_files = collect_data_files("faster_whisper")

# Qt modules this app never touches. Excluding them keeps the bundle several hundred
# MB smaller; anything left out that is genuinely needed is reported at launch.
excluded_qt = [
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNfc",
    "PySide6.QtOpenGL",
    "PySide6.QtPdf",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtSvg",
    "PySide6.QtTest",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
]

excluded_unused = [
    "torch",
    "torchaudio",
    "torchvision",
    "whisperx",
    "pyannote",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "tkinter",
    "pytest",
    "PyQt5",
    "PyQt6",
    "argostranslate",
]

a = Analysis(
    [str(project_root / "packaging" / "entry.py")],
    pathex=[str(src)],
    binaries=[],
    datas=data_files,
    # keyring needs its platform backend to be importable by name.
    hiddenimports=["keyring.backends.Windows"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_qt + excluded_unused,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DoSubs",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DoSubs",
)
