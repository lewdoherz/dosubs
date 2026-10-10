; Inno Setup script for DoSubs.
;
;   packaging\build.ps1 -IncludeCuda -Installer
; or manually:
;   iscc /DAppVersion=0.1.0 packaging\installer.iss
;
; Per-user install (no administrator prompt). The CUDA runtime is a separate optional
; component: it is about 2 GB unpacked, and a machine without an NVIDIA GPU never
; needs it. Speech models are not shipped; they download on first use. User data
; (models, cache, logs, config) lives in %LOCALAPPDATA%\DoSubs and %APPDATA%\DoSubs
; and is deliberately left in place on uninstall.

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
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Types]
Name: "full"; Description: "Application and CPU support"
Name: "gpu"; Description: "Application with NVIDIA CUDA acceleration (adds about 2 GB)"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "app"; Description: "DoSubs application"; Types: full gpu custom; Flags: fixed
Name: "cuda"; Description: "NVIDIA CUDA runtime (needed for GPU transcription)"; Types: gpu

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\DoSubs\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: app
Source: "..\dist\DoSubs-cuda\*"; DestDir: "{app}\_cuda"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: cuda

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
