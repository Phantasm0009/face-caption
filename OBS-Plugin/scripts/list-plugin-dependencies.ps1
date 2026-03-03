# List DLL dependencies of obs-face-captions.dll to find what's missing (error 126).
# Run from OBS-Plugin: .\scripts\list-plugin-dependencies.ps1
# Requires: Visual Studio (dumpbin) and the plugin deployed to OBS.

param(
    [string]$PluginDll = "C:\obs-studio\build_x64\rundir\RelWithDebInfo\obs-plugins\64bit\obs-face-captions.dll",
    [string]$SearchDir = "C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit"
)

$dumpbin = $null
$vsroots = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
)
foreach ($root in $vsroots) {
    $msvc = Join-Path $root "VC\Tools\MSVC"
    if (-not (Test-Path $msvc)) { continue }
    $ver = Get-ChildItem -Path $msvc -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ver) { continue }
    $exe = Join-Path $ver.FullName "bin\Hostx64\x64\dumpbin.exe"
    if (Test-Path $exe) { $dumpbin = $exe; break }
}
if (-not $dumpbin) {
    Write-Host "dumpbin not found. Open 'Developer PowerShell for VS' and run: dumpbin /dependents `"$PluginDll`"" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $PluginDll)) {
    Write-Host "Plugin DLL not found: $PluginDll" -ForegroundColor Red
    Write-Host "Deploy first: .\scripts\deploy-to-obs.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Dependencies of $PluginDll" -ForegroundColor Cyan
Write-Host "---"
$out = & $dumpbin /dependents $PluginDll 2>&1
$deps = $out | Where-Object { $_ -match "^\s+(\S+\.dll)\s*$" } | ForEach-Object { $matches[1] }
if ($deps.Count -eq 0) {
    Write-Host $out
    exit 0
}

# Windows/system DLLs: loader finds these in System32 or via API sets; don't report as missing
$systemDlls = @("KERNEL32.dll", "KERNELBASE.dll", "USER32.dll", "GDI32.dll", "ADVAPI32.dll", "msvcrt.dll", "SHELL32.dll", "ole32.dll", "OLEAUT32.dll", "RPCRT4.dll", "VCRUNTIME140.dll", "api-ms-win-crt-runtime-l1-1-0.dll", "api-ms-win-crt-heap-l1-1-0.dll", "api-ms-win-crt-math-l1-1-0.dll", "api-ms-win-crt-stdio-l1-1-0.dll", "api-ms-win-crt-string-l1-1-0.dll", "api-ms-win-crt-convert-l1-1-0.dll", "api-ms-win-crt-environment-l1-1-0.dll", "api-ms-win-crt-filesystem-l1-1-0.dll", "api-ms-win-crt-time-l1-1-0.dll", "api-ms-win-crt-utility-l1-1-0.dll")
$isSystem = { param($n) $systemDlls -contains $n -or $n -match "^api-ms-win-" }

Write-Host "Checking which exist in: $SearchDir"
Write-Host "---"
$missing = @()
foreach ($d in $deps) {
    $path = Join-Path $SearchDir $d
    if (Test-Path $path) {
        Write-Host "  OK  $d" -ForegroundColor Green
    } elseif (& $isSystem $d) {
        Write-Host "  OK  $d (system)" -ForegroundColor DarkGray
    } else {
        Write-Host "  MISSING  $d" -ForegroundColor Red
        $missing += $d
    }
}
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing DLLs (likely cause of error 126):" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
    Write-Host "Copy these into: $SearchDir" -ForegroundColor Yellow
    if ($missing -match "libvosk|vosk\.dll") {
        Write-Host "Vosk: run deploy with -VoskRoot `"C:\path\to\vosk-api`" or copy all DLLs from your Vosk package into the folder above." -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "All copyable dependencies are present. If OBS still shows error 126, ensure Vosk DLLs (libvosk.dll, etc.) are in the folder above (run deploy with -VoskRoot if needed)." -ForegroundColor Green
}
exit 0
