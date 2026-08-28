[CmdletBinding()]
param(
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryPath = $PSScriptRoot
$backendPath = Join-Path $repositoryPath 'backend'
$frontendPath = Join-Path $repositoryPath 'frontend'
$backendPython = Join-Path $repositoryPath '.runtime\serving-venv\Scripts\python.exe'
$frontendCli = Join-Path $frontendPath 'node_modules\.bin\ng.cmd'
$modelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.onnx'
$fp16ModelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.fp16.onnx'
$quantizedModelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.int8.onnx'
$voicesPath = Join-Path $backendPath 'model-artifacts\voices-v1.0.bin'
$audio8Path = Join-Path $backendPath 'model-artifacts\audio8-tts-preview-0.6b-int4\runtime_manifest.json'
$speechT5Path = Join-Path $backendPath 'model-artifacts\speecht5-tts\pytorch_model.bin'
$speechT5VocoderPath = Join-Path $backendPath 'model-artifacts\speecht5-hifigan\pytorch_model.bin'
$speechT5SpeakerPath = Join-Path $backendPath 'model-artifacts\speecht5-speakers\cmu-slt.npy'

function Assert-RequiredFile {
    param(
        [Parameter(Mandatory)]
        [string]$LiteralPath,

        [Parameter(Mandatory)]
        [string]$SetupHint
    )

    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "Missing: $LiteralPath`n$SetupHint"
    }
}

Assert-RequiredFile -LiteralPath $backendPython -SetupHint @"
Create the backend environment:
  cd $repositoryPath
  uv venv --python 3.12 .runtime\serving-venv
  uv pip install --python .runtime\serving-venv\Scripts\python.exe -e "backend[dev,serving]"
  uv pip install --python .runtime\serving-venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
"@

Assert-RequiredFile -LiteralPath $frontendCli -SetupHint @"
Install frontend dependencies:
  cd $frontendPath
  npm ci
"@

Assert-RequiredFile -LiteralPath $modelPath -SetupHint @"
Download the local model artifacts:
  cd $backendPath
  $backendPython scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $fp16ModelPath -SetupHint @"
Download the local model artifacts:
  cd $backendPath
  $backendPython scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $quantizedModelPath -SetupHint @"
Download the local model artifacts:
  cd $backendPath
  $backendPython scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $voicesPath -SetupHint @"
Download the local voice artifacts:
  cd $backendPath
  $backendPython scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $audio8Path -SetupHint @"
Download the CPU-compatible Audio8 and SpeechT5 artifacts:
  cd $backendPath
  $backendPython scripts\download_cpu_models.py
"@

Assert-RequiredFile -LiteralPath $speechT5Path -SetupHint @"
Download the CPU-compatible Audio8 and SpeechT5 artifacts:
  cd $backendPath
  $backendPython scripts\download_cpu_models.py
"@

Assert-RequiredFile -LiteralPath $speechT5VocoderPath -SetupHint @"
Download the CPU-compatible Audio8 and SpeechT5 artifacts:
  cd $backendPath
  $backendPython scripts\download_cpu_models.py
"@

Assert-RequiredFile -LiteralPath $speechT5SpeakerPath -SetupHint @"
Download the CPU-compatible Audio8 and SpeechT5 artifacts:
  cd $backendPath
  $backendPython scripts\download_cpu_models.py
"@

Write-Host 'OpenVoice Lab prerequisites are ready.' -ForegroundColor Green
Write-Host 'Audio8 INT4 and SpeechT5 CPU synthesis are enabled.' -ForegroundColor Green

if ($CheckOnly) {
    Write-Host 'Check complete. No servers were started.'
    exit 0
}

$backendProcess = $null
$frontendProcess = $null

try {
    $backendProcess = Start-Process `
        -FilePath $backendPython `
        -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--reload', '--host', '127.0.0.1', '--port', '8000') `
        -WorkingDirectory $backendPath `
        -PassThru

    $frontendProcess = Start-Process `
        -FilePath $frontendCli `
        -ArgumentList @('serve', '--host', '127.0.0.1', '--port', '4200', '--proxy-config', 'proxy.conf.json') `
        -WorkingDirectory $frontendPath `
        -PassThru
} catch {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id
    }

    throw
}

Write-Host ''
Write-Host 'OpenVoice Lab is starting:' -ForegroundColor Green
Write-Host '  Web:      http://localhost:4200'
Write-Host '  API docs: http://localhost:8000/docs'
Write-Host '  Health:   http://localhost:8000/health'
Write-Host ''
Write-Host "Backend PID: $($backendProcess.Id)"
Write-Host "Frontend PID: $($frontendProcess.Id)"
Write-Host 'Use Ctrl+C in each server window to stop it.'
