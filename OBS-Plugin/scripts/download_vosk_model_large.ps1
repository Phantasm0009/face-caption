# Download Vosk large English model. Run from OBS-Plugin: .\scripts\download_vosk_model_large.ps1
$ModelUrl = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
$ModelsDir = Join-Path (Split-Path $PSScriptRoot -Parent) "models"
$ModelDir = Join-Path $ModelsDir "vosk-model-en-us-0.22"
$ZipPath = Join-Path $env:TEMP "vosk-model-en-us-0.22.zip"
if (Test-Path (Join-Path $ModelDir "conf\model.conf")) { Write-Host "Exists: $ModelDir"; exit 0 }
Write-Host "Downloading large Vosk model (~1.8GB)..."
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Invoke-WebRequest -Uri $ModelUrl -OutFile $ZipPath -UseBasicParsing
Expand-Archive -Path $ZipPath -DestinationPath $ModelsDir -Force
Remove-Item $ZipPath -ErrorAction SilentlyContinue
Write-Host "Done: $ModelDir"
