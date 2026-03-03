# Deep recursive dependency check for obs-face-captions.dll
# Finds ALL missing DLLs in the dependency tree

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
if (-not $dumpbin) { Write-Host 'dumpbin not found'; exit 1 }

$bin64 = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit'
$pluginDir = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\obs-plugins\64bit'
$searchDirs = @($bin64, $pluginDir)

$systemPatterns = @(
    '^api-ms-win-', '^ext-ms-', '^KERNEL32', '^KERNELBASE', '^ntdll',
    '^USER32', '^GDI32', '^ADVAPI32', '^msvcrt\.dll', '^SHELL32', '^ole32',
    '^OLEAUT32', '^RPCRT4', '^SHLWAPI', '^COMCTL32', '^COMDLG32', '^IMM32',
    '^WS2_32', '^CRYPT32', '^WINTRUST', '^SETUPAPI', '^CFGMGR32', '^bcrypt',
    '^Secur32', '^IPHLPAPI', '^USERENV', '^PSAPI', '^DBGHELP', '^WINHTTP',
    '^WTSAPI32', '^DNSAPI', '^dwmapi', '^uxtheme', '^NETAPI32', '^d3d11',
    '^dxgi', '^WINMM', '^WLDAP32', '^VERSION', '^MPR', '^PROPSYS',
    '^powrprof', '^HVSOCKET', '^RtlGenRandom', '^profapi', '^UCRTBASE',
    '^MSWSOCK', '^D3DCOMPILER', '^mfplat', '^mf\.dll', '^mfreadwrite',
    '^CLDAPI', '^SspiCli', '^CoreMessaging', '^CRYPTBASE'
)

function Test-System($name) {
    foreach ($pat in $systemPatterns) {
        if ($name -match $pat) { return $true }
    }
    return $false
}

function Get-Deps($dllPath) {
    $out = & $dumpbin /dependents $dllPath 2>&1
    $deps = @()
    foreach ($line in $out) {
        if ($line -match '^\s+(\S+\.dll)\s*$') {
            $deps += $matches[1]
        }
    }
    return $deps
}

$checked = @{}
$queue = [System.Collections.ArrayList]@()
$queue.Add('C:\obs-studio\build_x64\rundir\RelWithDebInfo\obs-plugins\64bit\obs-face-captions.dll') | Out-Null
$allMissing = @()

while ($queue.Count -gt 0) {
    $current = $queue[0]
    $queue.RemoveAt(0)
    $name = Split-Path $current -Leaf
    $key = $name.ToLower()
    if ($checked.ContainsKey($key)) { continue }
    $checked[$key] = $true

    if (-not (Test-Path $current)) { continue }

    Write-Host ('Checking: ' + $name) -ForegroundColor DarkGray
    $deps = Get-Deps $current
    foreach ($dep in $deps) {
        $depKey = $dep.ToLower()
        if ($checked.ContainsKey($depKey)) { continue }
        if (Test-System $dep) {
            $checked[$depKey] = $true
            continue
        }

        $found = $null
        foreach ($dir in $searchDirs) {
            $p = Join-Path $dir $dep
            if (Test-Path $p) { $found = $p; break }
        }
        if (-not $found) {
            Write-Host ('  MISSING: ' + $dep + ' (needed by ' + $name + ')') -ForegroundColor Red
            $allMissing += ($dep + ' (needed by ' + $name + ')')
            $checked[$depKey] = $true
        } else {
            $queue.Add($found) | Out-Null
        }
    }
}

Write-Host ''
if ($allMissing.Count -eq 0) {
    Write-Host 'All dependencies found.' -ForegroundColor Green
} else {
    Write-Host 'MISSING DEPENDENCIES:' -ForegroundColor Yellow
    foreach ($m in $allMissing) { Write-Host ('  ' + $m) }
}
