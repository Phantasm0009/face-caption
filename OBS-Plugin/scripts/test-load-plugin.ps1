# Diagnostic: try to load obs-face-captions.dll with LoadLibraryEx and report the Win32 error.
# Run from OBS-Plugin: .\scripts\test-load-plugin.ps1

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class DllDiag {
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern bool SetDllDirectory(string lpPathName);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool FreeLibrary(IntPtr hModule);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern IntPtr LoadLibrary(string lpFileName);

    // DONT_RESOLVE_DLL_REFERENCES = 0x1: load DLL but do not call DllMain / resolve imports
    // This lets us test if the file itself is valid without needing dependencies.
    public const uint DONT_RESOLVE = 0x00000001;
    public const uint ALTERED_SEARCH = 0x00000008;
    public const uint SEARCH_DLL_LOAD_DIR = 0x00000100;
    public const uint SEARCH_DEFAULT_DIRS = 0x00001000;
}
'@

$bin64 = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit'
$pluginDir = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\obs-plugins\64bit'
$dll = Join-Path $pluginDir 'obs-face-captions.dll'

Write-Host ('Plugin DLL: ' + $dll)
Write-Host ('Exists: ' + (Test-Path $dll))
Write-Host ''

# Test 1: DONT_RESOLVE_DLL_REFERENCES - just map the PE, ignore imports
Write-Host 'Test 1: LoadLibraryEx with DONT_RESOLVE_DLL_REFERENCES (no dependency resolution)'
$h1 = [DllDiag]::LoadLibraryEx($dll, [IntPtr]::Zero, [DllDiag]::DONT_RESOLVE)
$e1 = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
if ($h1 -eq [IntPtr]::Zero) {
    Write-Host ('  FAILED: Win32 error ' + $e1) -ForegroundColor Red
} else {
    Write-Host '  SUCCESS (PE file is valid)' -ForegroundColor Green
    [DllDiag]::FreeLibrary($h1) | Out-Null
}
Write-Host ''

# Test 2: Set DLL directory to bin\64bit, then load normally
Write-Host ('Test 2: SetDllDirectory to bin\64bit, then LoadLibrary')
[DllDiag]::SetDllDirectory($bin64) | Out-Null
$h2 = [DllDiag]::LoadLibrary($dll)
$e2 = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
if ($h2 -eq [IntPtr]::Zero) {
    Write-Host ('  FAILED: Win32 error ' + $e2) -ForegroundColor Red
} else {
    Write-Host '  SUCCESS' -ForegroundColor Green
    [DllDiag]::FreeLibrary($h2) | Out-Null
}
Write-Host ''

# Test 3: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
Write-Host 'Test 3: LoadLibraryEx with SEARCH_DLL_LOAD_DIR | SEARCH_DEFAULT_DIRS'
$flags3 = [DllDiag]::SEARCH_DLL_LOAD_DIR -bor [DllDiag]::SEARCH_DEFAULT_DIRS
$h3 = [DllDiag]::LoadLibraryEx($dll, [IntPtr]::Zero, $flags3)
$e3 = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
if ($h3 -eq [IntPtr]::Zero) {
    Write-Host ('  FAILED: Win32 error ' + $e3) -ForegroundColor Red
} else {
    Write-Host '  SUCCESS' -ForegroundColor Green
    [DllDiag]::FreeLibrary($h3) | Out-Null
}
Write-Host ''

# Test 4: Pre-load obs.dll from bin\64bit, then try loading the plugin
Write-Host 'Test 4: Pre-load obs.dll from bin\64bit, then LoadLibrary plugin'
$obsDll = Join-Path $bin64 'obs.dll'
if (Test-Path $obsDll) {
    $hObs = [DllDiag]::LoadLibrary($obsDll)
    $eObs = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    if ($hObs -eq [IntPtr]::Zero) {
        Write-Host ('  Pre-load obs.dll FAILED: Win32 error ' + $eObs) -ForegroundColor Red
    } else {
        Write-Host '  Pre-loaded obs.dll OK' -ForegroundColor Green
        $h4 = [DllDiag]::LoadLibrary($dll)
        $e4 = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        if ($h4 -eq [IntPtr]::Zero) {
            Write-Host ('  Plugin load FAILED: Win32 error ' + $e4) -ForegroundColor Red
        } else {
            Write-Host '  Plugin load SUCCESS' -ForegroundColor Green
            [DllDiag]::FreeLibrary($h4) | Out-Null
        }
        [DllDiag]::FreeLibrary($hObs) | Out-Null
    }
} else {
    Write-Host ('  obs.dll not found at ' + $obsDll) -ForegroundColor Yellow
}
