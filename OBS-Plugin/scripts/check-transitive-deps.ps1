# Check transitive dependencies of libvosk.dll and opencv_core4.dll in bin\64bit (find missing DLLs for error 126).
# Run from OBS-Plugin: .\scripts\check-transitive-deps.ps1

param(
    [string]$Bin64 = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit'
)

$dumpbin = $null
foreach ($root in @(
    'C:\Program Files\Microsoft Visual Studio\2022\Community',
    'C:\Program Files\Microsoft Visual Studio\2022\Professional',
    'C:\Program Files\Microsoft Visual Studio\2022\Enterprise'
)) {
    $msvc = Join-Path $root 'VC\Tools\MSVC'
    if (-not (Test-Path $msvc)) { continue }
    $ver = Get-ChildItem -Path $msvc -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ver) { continue }
    $exe = Join-Path $ver.FullName 'bin\Hostx64\x64\dumpbin.exe'
    if (Test-Path $exe) { $dumpbin = $exe; break }
}
if (-not $dumpbin) {
    Write-Host 'dumpbin not found. Install Visual Studio or run from Developer PowerShell.' -ForegroundColor Yellow
    exit 1
}

$systemDlls = @(
    'KERNEL32.dll', 'KERNELBASE.dll', 'USER32.dll', 'GDI32.dll', 'ADVAPI32.dll', 'msvcrt.dll',
    'SHELL32.dll', 'ole32.dll', 'OLEAUT32.dll', 'RPCRT4.dll', 'VCRUNTIME140.dll',
    'api-ms-win-crt-runtime-l1-1-0.dll', 'api-ms-win-crt-heap-l1-1-0.dll', 'api-ms-win-crt-math-l1-1-0.dll',
    'api-ms-win-crt-stdio-l1-1-0.dll', 'api-ms-win-crt-string-l1-1-0.dll', 'api-ms-win-crt-convert-l1-1-0.dll',
    'api-ms-win-crt-environment-l1-1-0.dll', 'api-ms-win-crt-filesystem-l1-1-0.dll', 'api-ms-win-crt-time-l1-1-0.dll',
    'api-ms-win-crt-utility-l1-1-0.dll', 'MSVCP140.dll', 'VCRUNTIME140_1.dll'
)

$toCheck = @(
    (Join-Path $Bin64 'libvosk.dll'),
    (Join-Path $Bin64 'opencv_core4.dll')
)

$allMissing = @{}
foreach ($dll in $toCheck) {
    $name = Split-Path $dll -Leaf
    if (-not (Test-Path $dll)) {
        Write-Host "Skip (not found): $dll" -ForegroundColor DarkGray
        continue
    }
    Write-Host "Dependencies of $name" -ForegroundColor Cyan
    $out = & $dumpbin /dependents $dll 2>&1
    $deps = $out | Where-Object { $_ -match '^\s+(\S+\.dll)\s*$' } | ForEach-Object { $matches[1] }
    foreach ($dep in $deps) {
        $path = Join-Path $Bin64 $dep
        $isSystem = ($systemDlls -contains $dep) -or ($dep -match '^api-ms-win-')
        if (Test-Path $path) {
            Write-Host "  OK   $dep" -ForegroundColor Green
        } elseif ($isSystem) {
            Write-Host "  OK   $dep (system)" -ForegroundColor DarkGray
        } else {
            Write-Host "  MISSING  $dep" -ForegroundColor Red
            $allMissing[$dep] = $true
        }
    }
    Write-Host ''
}

if ($allMissing.Count -gt 0) {
    Write-Host 'Missing DLLs (likely cause of error 126). Copy them into:' -ForegroundColor Yellow
    Write-Host "  $Bin64" -ForegroundColor Cyan
    $allMissing.Keys | Sort-Object | ForEach-Object { Write-Host "    $_" }
}
