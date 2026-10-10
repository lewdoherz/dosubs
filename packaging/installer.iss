; Inno Setup script for DoSubs.
;
;   iscc /DAppVersion=0.1.0 packaging\installer.iss
;   iscc /DAppVersion=0.1.0 /DIncludeCuda packaging\installer.iss   ; bundle the 2 GB CUDA runtime
;
; The CUDA runtime is compile-time optional because compressing ~2 GB of NVIDIA DLLs into the
; installer takes a very long time; the default build expects users to unzip
; DoSubs-cuda-<version>-windows-x64.zip into "_cuda" next to DoSubs.exe (the app discovers it).
; Pass /DIncludeCuda to produce a single self-contained installer instead.
;
; Per-user install (no administrator prompt). Speech models are not shipped; they download on
; first use. User data (models, cache, logs, config) lives in %LOCALAPPDATA%\DoSubs and
; %APPDATA%\DoSubs and is deliberately left in place on uninstall.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define AppName "DoSubs"
#define AppPublisher "DoSubs"
#define AppExeName "DoSubs.exe"

[Setup]
AppId={{8F1C4C1E-2E52-4C1B-9A6E-4C9C2F1B7D31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=DoSubs-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
; Chinese Simplified is not one of the 29 translations Inno Setup ships. To use it, download
; ChineseSimplified.isl from the Inno Setup translations repository into the compiler's
; Languages folder and add:  Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Types]
Name: "full"; Description: "Application (CPU transcription, GPU optional later)"
#ifdef IncludeCuda
Name: "gpu"; Description: "Application with NVIDIA CUDA acceleration (adds about 2 GB)"
#endif
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "app"; Description: "DoSubs application"; Types: full custom; Flags: fixed
#ifdef IncludeCuda
Name: "cuda"; Description: "NVIDIA CUDA runtime (needed for GPU transcription)"; Types: full gpu custom
#endif

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\DoSubs\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: app
#ifdef IncludeCuda
Source: "..\dist\DoSubs-cuda\*"; DestDir: "{app}\_cuda"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: cuda
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
