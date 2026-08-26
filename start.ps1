[CmdletBinding()]
param(
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryPath = $PSScriptRoot
$backendPath = Join-Path $repositoryPath 'backend'
$frontendPath = Join-Path $repositoryPath 'frontend'
$backendPython = Join-Path $backendPath '.venv\Scripts\python.exe'
$frontendCli = Join-Path $frontendPath 'node_modules\.bin\ng.cmd'
$modelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.onnx'
$quantizedModelPath = Join-Path $backendPath 'model-artifacts\kokoro-v1.0.int8.onnx'
$voicesPath = Join-Path $backendPath 'model-artifacts\voices-v1.0.bin'
$npmCommand = Get-Command 'npm.cmd' -ErrorAction SilentlyContinue

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

if (-not $npmCommand) {
    throw 'npm.cmd is not available. Install Node.js and reopen PowerShell.'
}

Assert-RequiredFile -LiteralPath $modelPath -SetupHint @"
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
        -FilePath $npmCommand.Source `
        -ArgumentList @('start', '--', '--host=127.0.0.1', '--port=4200') `
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
