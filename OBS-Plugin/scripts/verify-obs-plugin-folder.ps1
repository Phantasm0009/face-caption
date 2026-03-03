# Verify that the OBS exe folder has the Face Captions dependencies (fixes error 126).
# Run from OBS-Plugin: .\scripts\verify-obs-plugin-folder.ps1

$Bin64 = "C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit"
# libvosk.dll needs these MinGW runtime DLLs in the same folder
$required = @("obs64.exe", "obs.dll", "libvosk.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll", "opencv_core4.dll", "vcruntime140.dll")
$missing = @()
foreach ($name in $required) {
    $path = Join-Path $Bin64 $name
    if (Test-Path $path) {
        Write-Host "  OK  $name" -ForegroundColor Green
    } else {
        Write-Host "  MISSING  $name" -ForegroundColor Red
        $missing += $name
    }
}
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "The folder that must contain these files is:" -ForegroundColor Yellow
    Write-Host "  $Bin64" -ForegroundColor Cyan
    Write-Host "Re-run deploy with Vosk so dependencies are copied there:" -ForegroundColor Yellow
    Write-Host '  .\scripts\deploy-to-obs.ps1 -VoskRoot "C:\Users\Pramod Tiwari\Downloads\vosk-win64-0.3.45"' -ForegroundColor White
    Write-Host "Then start OBS by double-clicking obs64.exe in that folder (or run .\scripts\run-obs-from-build.ps1)." -ForegroundColor Yellow
    exit 1
}
Write-Host ""
Write-Host "All required files are in the exe folder. Start OBS from there (double-click obs64.exe or run-obs-from-build.ps1)." -ForegroundColor Green
exit 0
