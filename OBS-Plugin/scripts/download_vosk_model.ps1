# Download Vosk small English model. Run from OBS-Plugin: .\scripts\download_vosk_model.ps1
$ModelUrl = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
$Root = Split-Path $PSScriptRoot -Parent
$ModelsDir = Join-Path $Root "models"
$ZipPath = Join-Path $env:TEMP "vosk-model-small-en-us-0.15.zip"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Invoke-WebRequest -Uri $ModelUrl -OutFile $ZipPath -UseBasicParsing
Expand-Archive -Path $ZipPath -DestinationPath $ModelsDir -Force
Remove-Item $ZipPath -ErrorAction SilentlyContinue
Write-Host "Done. Model in $ModelsDir"
