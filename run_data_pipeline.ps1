<#
.SYNOPSIS
    Runs the OpenVoice Lab medical TTS data-preparation pipeline.

.DESCRIPTION
    This is the root-level convenience launcher. It uses the repository's
    stage11-test Python environment and invokes all data-pipeline stages in
    order: inventory, validation, audio preparation, text/quality checks,
    duplicate detection, ASR, review, splitting, SpeechT5 preparation, and
    reporting.

    The launcher does not modify the raw dataset. Generated audio, manifests,
    reports, and review files are written below data-processing\.

.EXAMPLE
    .\run_data_pipeline.ps1 -Smoke
    Runs a bounded 64-record smoke test in data-processing\*\medical_tts\smoke.

.EXAMPLE
    .\run_data_pipeline.ps1 -Smoke -Limit 128 -RunName smoke-128 -AsrMode full
    Runs a 128-record smoke test and requests Whisper ASR.

.EXAMPLE
    .\run_data_pipeline.ps1 -SkipReview
    Runs the full dataset and accepts manual-review flags without reviewer
    decisions. Hard validation and audio-quality exclusions remain active.

.EXAMPLE
    .\run_data_pipeline.ps1 -ReviewServer
    Starts the review UI at http://127.0.0.1:8765/.

.EXAMPLE
    Get-Help .\run_data_pipeline.ps1 -Full
    Displays this usage guide.
#>
[CmdletBinding()]
param(
    [switch]$Smoke,
    [int]$Limit = 64,
    [string]$RunName = 'smoke',
    [ValidateSet('disabled', 'cache_only', 'full')]
    [string]$AsrMode,
    [switch]$SkipReview,
    [switch]$ReviewServer,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryPath = $PSScriptRoot
$pythonPath = Join-Path $repositoryPath '.runtime\stage11-test-venv\Scripts\python.exe'
$configPath = Join-Path $repositoryPath 'training\config\dataset.yaml'

function Assert-RequiredFile {
    param(
        [Parameter(Mandatory)] [string]$LiteralPath,
        [Parameter(Mandatory)] [string]$SetupHint
    )

    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "Missing: $LiteralPath`n$SetupHint"
    }
}

Assert-RequiredFile -LiteralPath $pythonPath -SetupHint @"
Create the training environment or use an existing compatible Python environment:
  py -3.11 -m venv .runtime\stage11-test-venv
  .runtime\stage11-test-venv\Scripts\python.exe -m pip install -r training\requirements.lock
"@
Assert-RequiredFile -LiteralPath $configPath -SetupHint 'The dataset configuration is missing.'

if ($Limit -le 0) {
    throw '-Limit must be positive.'
}

if ($ReviewServer) {
    Write-Host 'Starting the review server at http://127.0.0.1:8765/' -ForegroundColor Green
    & $pythonPath -m training.data_pipeline.review_server --config $configPath
    exit $LASTEXITCODE
}

if ($CheckOnly) {
    Write-Host 'Data-pipeline prerequisites are ready.' -ForegroundColor Green
    Write-Host "Python: $pythonPath"
    Write-Host "Config: $configPath"
    exit 0
}

$arguments = @(
    '-m', 'training.data_pipeline.run',
    '--config', $configPath
)

if ($Smoke) {
    $arguments += @('--limit', $Limit, '--run-name', $RunName, '--accept-unreviewed-for-smoke')
} elseif ($SkipReview) {
    $arguments += '--skip-review'
}

if ($AsrMode) {
    $arguments += @('--asr-mode', $AsrMode)
}

Write-Host 'Running the OpenVoice Lab data pipeline...' -ForegroundColor Green
if ($Smoke) {
    Write-Host "Mode: smoke ($Limit records, run name '$RunName')"
} else {
    Write-Host 'Mode: full dataset'
}

Push-Location $repositoryPath
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Data pipeline failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Data pipeline completed.' -ForegroundColor Green
Write-Host 'Reports:   data-processing\reports\medical_tts'
Write-Host 'Manifests: data-processing\manifests\medical_tts'
