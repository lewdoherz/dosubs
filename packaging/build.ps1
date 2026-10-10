# Builds the Windows application: PyInstaller one-directory bundle, then optionally
# the CUDA runtime component and the installer.
#
#   packaging\build.ps1                        # app bundle only (CPU-capable)
#   packaging\build.ps1 -IncludeCuda           # also collect the CUDA DLLs (about 2 GB)
#   packaging\build.ps1 -IncludeCuda -Installer # also compile installer.iss (needs Inno Setup)
[CmdletBinding()]
param(
    [switch]$IncludeCuda,
    [switch]$Installer,
    [string]$AppVersion = "0.1.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found at $python. Create it first (see README)."
}

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in the virtual environment. Run: $python -m pip install -e `".[packaging]`""
}

Write-Host "Building the application with PyInstaller ..." -ForegroundColor Cyan
Push-Location $root
try {
    & $python -m PyInstaller "packaging\dosubs.spec" --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

$appDir = Join-Path $root "dist\DoSubs"
if (-not (Test-Path $appDir)) { throw "Expected bundle at $appDir was not produced." }
$sizeMb = [math]::Round((Get-ChildItem $appDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Bundle: $appDir ($sizeMb MB)" -ForegroundColor Green

if ($IncludeCuda) {
    Write-Host "Collecting the CUDA runtime ..." -ForegroundColor Cyan
    $sitePackages = & $python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    $nvidiaRoot = Join-Path $sitePackages "nvidia"
    if (-not (Test-Path $nvidiaRoot)) {
        throw "No nvidia wheels found in $sitePackages. Install them with: $python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
    }

    $cudaDir = Join-Path $root "dist\DoSubs-cuda"
    if (Test-Path $cudaDir) { Remove-Item $cudaDir -Recurse -Force }

    foreach ($component in @("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")) {
        $source = Join-Path $nvidiaRoot "$component\bin"
        if (-not (Test-Path $source)) { continue }
        $target = Join-Path $cudaDir "$component\bin"
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Copy-Item (Join-Path $source "*.dll") $target -Force
        Write-Host "  $component -> $target"
    }

    $cudaMb = [math]::Round((Get-ChildItem $cudaDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
    Write-Host "CUDA component: $cudaDir ($cudaMb MB)" -ForegroundColor Green
}

if ($Installer) {
    $iscc = $null
    $command = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($command) { $iscc = $command.Source }
    if (-not $iscc) {
        # The winget package installs Inno Setup per-user, which is not on PATH.
        foreach ($candidate in @(
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )) {
            if (Test-Path $candidate) { $iscc = $candidate; break }
        }
    }
    if (-not $iscc) {
        throw "Inno Setup (ISCC.exe) was not found. Install it with: winget install --id JRSoftware.InnoSetup -e"
    }

    Write-Host "Compiling the installer with $iscc ..." -ForegroundColor Cyan
    $defines = @("/DAppVersion=$AppVersion")
    if ($IncludeCuda) { $defines += "/DIncludeCuda" }   # self-contained installer, ~1.3 GB output
    & $iscc @defines (Join-Path $root "packaging\installer.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }
    Write-Host "Installer written to $root\dist" -ForegroundColor Green
}
