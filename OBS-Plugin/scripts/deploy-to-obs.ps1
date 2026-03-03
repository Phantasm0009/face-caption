# Deploy Face Captions plugin to OBS
# Run from OBS-Plugin: .\scripts\deploy-to-obs.ps1
# When running OBS from rundir, OBS loads plugins from rundir\RelWithDebInfo\obs-plugins\64bit.
# Windows resolves plugin dependencies from the APPLICATION directory first (bin\64bit), not the plugin folder.
# So we copy dependency DLLs into bin\64bit (next to obs64.exe) so the loader finds them (avoids error 126).

param(
    [string]$ObsPlugins = 'C:\obs-studio\build_x64\obs-plugins\64bit',
    [string]$VoskRoot = 'C:\vosk-api',
    [string]$VcpkgBin = 'C:\vcpkg\installed\x64-windows\bin'
)

$PluginRoot = Split-Path $PSScriptRoot -Parent
$BuildDir = Join-Path $PluginRoot "build"
$script:RundirPluginsAbs = $null
$DllSource = Join-Path $BuildDir "RelWithDebInfo\obs-face-captions.dll"
$ModelsSource = Join-Path $PluginRoot "models"

if (-not (Test-Path $DllSource)) {
    Write-Error ('Build not found: ' + $DllSource + '. Run cmake --build . --config RelWithDebInfo in build folder first.')
    exit 1
}

# Deploy to the given path (default: build_x64\obs-plugins\64bit)
$Cascade = Join-Path $PluginRoot "data\haarcascade_frontalface_default.xml"
$ObsBase = Split-Path (Split-Path $ObsPlugins -Parent) -Parent
$PluginDataDir = Join-Path $ObsBase "data\obs-plugins\obs-face-captions"

# Find Visual C++ Redist CRT folder (MSVCP140.dll, VCRUNTIME140.dll, etc.)
function Get-VCRedistDir {
    $vsroots = @(
        'C:\Program Files\Microsoft Visual Studio\2022\Community',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise'
    )
    foreach ($root in $vsroots) {
        $redist = Join-Path $root "VC\Redist\MSVC"
        if (-not (Test-Path $redist)) { continue }
        $verDirs = Get-ChildItem -Path $redist -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($verDir in $verDirs) {
            $x64 = Join-Path $verDir.FullName "x64"
            if (-not (Test-Path $x64)) { continue }
            $crt = Get-ChildItem -Path $x64 -Directory -Filter "Microsoft.VC*.CRT" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($crt -and (Test-Path (Join-Path $crt.FullName "vcruntime140.dll"))) { return $crt.FullName }
        }
    }
    return $null
}

# Copy dependency DLLs next to the plugin so LoadLibrary can find them (avoids error 126)
# Returns $true if libvosk.dll was copied, $false otherwise
function Copy-PluginDependencies {
    param([string]$DestDir)
    $voskCopied = $false
    if (-not (Test-Path $DestDir)) { return $voskCopied }
    # VC++ runtime (MSVCP140, VCRUNTIME140) - required by plugin; copy from VS Redist if vcpkg doesn't have them
    $vcRedist = Get-VCRedistDir
    if ($vcRedist) {
        Get-ChildItem -Path $vcRedist -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination $DestDir -Force
            Write-Host ('  Copied ' + $_.Name + ' (VC Redist) to ' + $DestDir)
        }
    }
    # Vosk: copy all DLLs from Vosk root and lib/ so libvosk.dll and its dependencies (e.g. OpenBLAS) are present
    foreach ($dir in @($VoskRoot, (Join-Path $VoskRoot "lib"))) {
        if (-not (Test-Path $dir)) { continue }
        Get-ChildItem -Path $dir -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination $DestDir -Force
            Write-Host ('  Copied ' + $_.Name + ' (Vosk) to ' + $DestDir)
            if ($_.Name -match "libvosk|^vosk\.dll") { $voskCopied = $true }
        }
    }
    # OpenCV and common vcpkg DLLs (plugin links against OpenCV)
    if (Test-Path $VcpkgBin) {
        $patterns = @("opencv_*.dll", "tiff.dll", "libpng16.dll", "zlib1.dll", "zlib.dll", "jpeg62.dll", "libwebp*.dll", "libsharpyuv.dll",
            "msvcp140*.dll", "vcruntime140*.dll", "concrt140.dll", "vccorlib140*.dll",
            "libprotobuf.dll", "abseil_dll.dll")
        foreach ($pat in $patterns) {
            Get-ChildItem -Path $VcpkgBin -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item -Path $_.FullName -Destination $DestDir -Force
                Write-Host ('  Copied ' + $_.Name + ' to ' + $DestDir)
            }
        }
    }
    return $voskCopied
}

$PluginsDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ObsPlugins)
$ObsBaseP = Split-Path (Split-Path $PluginsDir -Parent) -Parent
$DataDirP = Join-Path $ObsBaseP "data\obs-plugins\obs-face-captions"
New-Item -ItemType Directory -Force -Path $PluginsDir | Out-Null
Copy-Item -Path $DllSource -Destination (Join-Path $PluginsDir "obs-face-captions.dll") -Force
Write-Host ('Copied DLL to ' + $PluginsDir)
$voskOk1 = Copy-PluginDependencies -DestDir $PluginsDir
New-Item -ItemType Directory -Force -Path $DataDirP | Out-Null
if (Test-Path $ModelsSource) {
    $ModelsDest = Join-Path $DataDirP "models"
    New-Item -ItemType Directory -Force -Path $ModelsDest | Out-Null
    Copy-Item -Path (Join-Path $ModelsSource '*') -Destination $ModelsDest -Recurse -Force
    Write-Host ('Copied models to ' + $DataDirP + '\models')
}
if (Test-Path $Cascade) {
    Copy-Item -Path $Cascade -Destination $DataDirP -Force
    Write-Host ('Copied Haar cascade to ' + $DataDirP)
}

# When using build_x64 default, also deploy to rundir so running from rundir finds the plugin
$RundirPlugins = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\obs-plugins\64bit'
if ($ObsPlugins -eq 'C:\obs-studio\build_x64\obs-plugins\64bit' -and (Test-Path (Split-Path $RundirPlugins -Parent))) {
    $RundirPluginsAbs = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RundirPlugins)
    $RundirBase = Split-Path (Split-Path $RundirPluginsAbs -Parent) -Parent
    $RundirData = Join-Path $RundirBase "data\obs-plugins\obs-face-captions"
    New-Item -ItemType Directory -Force -Path $RundirPluginsAbs | Out-Null
    Copy-Item -Path $DllSource -Destination (Join-Path $RundirPluginsAbs "obs-face-captions.dll") -Force
    Write-Host ('Copied DLL to rundir: ' + $RundirPluginsAbs)
    $RundirBin64 = Join-Path $RundirBase "bin\64bit"
    # OBS loads plugins with DLL search path set to the plugin folder. Copy ALL DLLs from bin\64bit into the
    # plugin folder so obs.dll and its dependencies (Qt, etc.) are found when the plugin loads (fixes error 126).
    if (Test-Path $RundirBin64) {
        $binDlls = Get-ChildItem -Path $RundirBin64 -Filter '*.dll' -ErrorAction SilentlyContinue
        foreach ($d in $binDlls) {
            Copy-Item -Path $d.FullName -Destination (Join-Path $RundirPluginsAbs $d.Name) -Force
        }
        Write-Host ('Copied OBS runtime DLLs from bin\64bit to plugin folder (' + (@($binDlls).Count) + ' files).')
    }
    $voskOk2 = Copy-PluginDependencies -DestDir $RundirPluginsAbs
    $script:RundirPluginsAbs = $RundirPluginsAbs
    # Also copy deps next to obs64.exe for any loader path that uses exe directory
    if (Test-Path (Split-Path $RundirBin64 -Parent)) {
        New-Item -ItemType Directory -Force -Path $RundirBin64 | Out-Null
        Write-Host ('Copying plugin dependencies to OBS exe folder (so loader finds them): ' + $RundirBin64)
        Copy-PluginDependencies -DestDir $RundirBin64 | Out-Null
    }
    New-Item -ItemType Directory -Force -Path $RundirData | Out-Null
    if (Test-Path $ModelsSource) {
        $ModelsDest = Join-Path $RundirData "models"
        New-Item -ItemType Directory -Force -Path $ModelsDest | Out-Null
        Copy-Item -Path (Join-Path $ModelsSource '*') -Destination $ModelsDest -Recurse -Force
    }
    if (Test-Path $Cascade) { Copy-Item -Path $Cascade -Destination $RundirData -Force }
    $libvoskInBin = Test-Path (Join-Path $RundirBin64 "libvosk.dll")
    if (-not $libvoskInBin -and $voskOk2) {
        Write-Host 'Note: libvosk.dll was copied to plugins folder but not found in bin\64bit; copying again.' -ForegroundColor Yellow
        Copy-PluginDependencies -DestDir $RundirBin64 | Out-Null
    }
}

Write-Host 'Restart OBS. Leave Vosk Model Path empty to use best model.'
Write-Host 'Start OBS from the BUILD (so the plugin loads): from OBS-Plugin run: .\scripts\run-obs-from-build.ps1' -ForegroundColor Cyan
Write-Host 'Do not use the installed OBS or a Start Menu shortcut - that uses a different exe and the plugin will fail (error 126).' -ForegroundColor Cyan
if (-not $voskOk1) {
    Write-Host ''
    Write-Host ('WARNING: No Vosk DLLs found at ' + $VoskRoot + ' (or ' + $VoskRoot + '\lib).') -ForegroundColor Yellow
    Write-Host 'The Face Captions plugin will fail to load (error 126). Copy ALL .dll files from your Vosk package into the folder next to obs64.exe:' -ForegroundColor Yellow
    $exeDir = 'C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit'
    Write-Host ('  ' + $exeDir) -ForegroundColor Cyan
    Write-Host 'Get Vosk from: https://github.com/alphacep/vosk-api/releases. Or run with: -VoskRoot C:\path\to\vosk-api' -ForegroundColor Yellow
}
exit 0
