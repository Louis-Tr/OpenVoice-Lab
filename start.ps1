[CmdletBinding()]
param(
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryPath = $PSScriptRoot
$backendPath = Join-Path $repositoryPath 'backend'
$frontendPath = Join-Path $repositoryPath 'frontend'
$baseBackendPython = Join-Path $backendPath '.venv\Scripts\python.exe'
$experimentBackendPython = Join-Path $repositoryPath '.runtime\stage12-venv\Scripts\python.exe'
$backendPython = if (Test-Path -LiteralPath $experimentBackendPython -PathType Leaf) {
    $experimentBackendPython
} else {
    $baseBackendPython
}
$frontendCli = Join-Path $frontendPath 'node_modules\.bin\ng.cmd'
$modelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.onnx'
$fp16ModelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.fp16.onnx'
$quantizedModelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.int8.onnx'
$voicesPath = Join-Path $backendPath 'model-artifacts\voices-v1.0.bin'

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
  cd $backendPath
  py -3 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
"@

Assert-RequiredFile -LiteralPath $frontendCli -SetupHint @"
Install frontend dependencies:
  cd $frontendPath
  npm ci
"@

Assert-RequiredFile -LiteralPath $modelPath -SetupHint @"
Download the local model artifacts:
  cd $backendPath
  .\.venv\Scripts\python.exe scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $fp16ModelPath -SetupHint @"
Download the local model artifacts:
  cd $backendPath
  .\.venv\Scripts\python.exe scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $quantizedModelPath -SetupHint @"
Download the local model artifacts:
  cd $backendPath
  .\.venv\Scripts\python.exe scripts\download_models.py
"@

Assert-RequiredFile -LiteralPath $voicesPath -SetupHint @"
Download the local voice artifacts:
  cd $backendPath
  .\.venv\Scripts\python.exe scripts\download_models.py
"@

Write-Host 'OpenVoice Lab prerequisites are ready.' -ForegroundColor Green
if ($backendPython -eq $experimentBackendPython) {
    Write-Host 'Stage 12 CPU comparison runtime is enabled.' -ForegroundColor Green
} else {
    Write-Host 'Stage 12 CPU dependencies are not enabled; the Experiment tab remains available.' -ForegroundColor Yellow
    Write-Host 'Follow the Stage 12 setup in README.md to enable local SpeechT5 comparisons.' -ForegroundColor Yellow
}

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
