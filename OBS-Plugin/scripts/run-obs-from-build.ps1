# Start OBS from the source build so the Face Captions plugin (and its DLLs in bin\64bit) loads.
# Run from OBS-Plugin: .\scripts\run-obs-from-build.ps1
#
# IMPORTANT: You MUST start OBS with this script. If you start OBS from the Start Menu, desktop
# shortcut, or by double-clicking obs64.exe, Windows uses the INSTALLED OBS (Program Files), which
# does NOT have the plugin dependencies in its folder — you will get "Plugin Load Error" (126).
#
# If you get "Failed to initialize video": do NOT run PowerShell as Administrator.

param(
    [string]$Rundir = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo'
)

$obsExe = Join-Path $Rundir 'bin\64bit\obs64.exe'
if (-not (Test-Path $obsExe)) {
    Write-Error ('OBS not found: ' + $obsExe + '. Build OBS from source first (see README Step 3).')
    exit 1
}

$bin64 = Join-Path $Rundir 'bin\64bit'
$pluginDll = Join-Path $Rundir 'obs-plugins\64bit\obs-face-captions.dll'
if (-not (Test-Path $pluginDll)) {
    Write-Host 'WARNING: obs-face-captions.dll not found in build. Run .\scripts\deploy-to-obs.ps1 -VoskRoot "C:\path\to\vosk-win64-0.3.45" first.' -ForegroundColor Yellow
}
if (-not (Test-Path (Join-Path $bin64 'libvosk.dll'))) {
    Write-Host 'WARNING: libvosk.dll not in bin\64bit. Deploy with -VoskRoot so the plugin can load.' -ForegroundColor Yellow
}

Write-Host 'Starting OBS from BUILD (required for Face Captions). Do NOT use Start Menu or other shortcuts.' -ForegroundColor Cyan
# Prepend bin\64bit to PATH so the loader can find plugin dependencies (OBS may use SetDllDirectory when loading plugins).
$env:PATH = $bin64 + ';' + $env:PATH
# Use bin\64bit as CWD so GPU/video init works. Plugin dependency DLLs must be in this folder (deploy copies them here).
Set-Location $bin64
& $obsExe
