# Download OpenCV Haar cascade for face detection
# Run from OBS-Plugin: .\scripts\download_haarcascade.ps1

$Url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
$DataDir = Join-Path (Split-Path $PSScriptRoot -Parent) "data"
$OutPath = Join-Path $DataDir "haarcascade_frontalface_default.xml"

if (Test-Path $OutPath) {
    Write-Host "Haar cascade already exists at $OutPath"
    exit 0
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Write-Host "Downloading haarcascade_frontalface_default.xml..."
try {
    Invoke-WebRequest -Uri $Url -OutFile $OutPath -UseBasicParsing
} catch {
    Write-Error "Download failed: $_"
    exit 1
}
Write-Host "Saved to $OutPath"
