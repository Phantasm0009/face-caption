# Copy missing DLLs (libprotobuf.dll, abseil_dll.dll) from vcpkg to OBS folders
$vcpkg = 'C:\vcpkg\installed\x64-windows\bin'
$bin64 = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit'
$pluginDir = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\obs-plugins\64bit'

$missing = @('libprotobuf.dll', 'abseil_dll.dll')

foreach ($name in $missing) {
    $src = Join-Path $vcpkg $name
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $bin64 $name) -Force
        Copy-Item -Path $src -Destination (Join-Path $pluginDir $name) -Force
        Write-Host ('Copied ' + $name + ' from vcpkg to bin\64bit and plugin folder') -ForegroundColor Green
    } else {
        Write-Host ('NOT FOUND in vcpkg bin: ' + $src) -ForegroundColor Yellow
        # Search broader under C:\vcpkg
        $found = Get-ChildItem -Path 'C:\vcpkg' -Filter $name -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            Write-Host ('  Found at: ' + $found.FullName) -ForegroundColor Cyan
            Copy-Item -Path $found.FullName -Destination (Join-Path $bin64 $name) -Force
            Copy-Item -Path $found.FullName -Destination (Join-Path $pluginDir $name) -Force
            Write-Host ('  Copied ' + $name) -ForegroundColor Green
        } else {
            Write-Host ('  Not found anywhere under C:\vcpkg') -ForegroundColor Red
        }
    }
}

# Re-run the test load
Write-Host ''
Write-Host 'Testing plugin load...'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class DllTest {
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern IntPtr LoadLibrary(string lpFileName);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern bool SetDllDirectory(string lpPathName);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool FreeLibrary(IntPtr hModule);
}
'@

[DllTest]::SetDllDirectory($bin64) | Out-Null
$dll = Join-Path $pluginDir 'obs-face-captions.dll'
$h = [DllTest]::LoadLibrary($dll)
$e = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
if ($h -eq [IntPtr]::Zero) {
    Write-Host ('Plugin load FAILED: Win32 error ' + $e) -ForegroundColor Red
} else {
    Write-Host 'Plugin load SUCCESS!' -ForegroundColor Green
    [DllTest]::FreeLibrary($h) | Out-Null
}
