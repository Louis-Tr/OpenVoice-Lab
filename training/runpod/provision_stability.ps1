param(
    [string]$RepositoryRoot = 'D:\OpenVoice-Lab',
    [int]$MaximumMinutes = 45
)

$ErrorActionPreference = 'Stop'
$apiKey = $null
foreach ($line in Get-Content -LiteralPath (Join-Path $RepositoryRoot '.env')) {
    if ($line -match '^RUNPOD_API=(.+)$') {
        $apiKey = $Matches[1].Trim('"', "'")
        break
    }
}
if (-not $apiKey) {
    throw 'RUNPOD_API is missing from .env'
}
$headers = @{ Authorization = "Bearer $apiKey" }
$existing = @(
    Invoke-RestMethod `
        -Uri 'https://rest.runpod.io/v1/pods?includeMachine=true' `
        -Headers $headers `
        -Method Get |
        Where-Object { $_.name -like 'ovl-s11-stability*' }
)
if ($existing.Count) {
    throw 'An OpenVoice stability Pod already exists; refusing to create another'
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "openvoice-stability-$stamp"
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$keyPath = Join-Path $tempRoot 'id_ed25519'
& ssh-keygen -q -t ed25519 -N '' -f $keyPath
if ($LASTEXITCODE -ne 0) {
    throw 'ssh-keygen failed'
}
$publicKey = (Get-Content -LiteralPath "$keyPath.pub" -Raw).Trim()
$passwordBytes = New-Object byte[] 24
[System.Security.Cryptography.RandomNumberGenerator]::Fill($passwordBytes)
$jupyterPassword = [Convert]::ToBase64String($passwordBytes).TrimEnd('=')

$body = @{
    name              = "ovl-s11-stability-$stamp"
    imageName         = 'runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04'
    gpuTypeIds        = @('NVIDIA GeForce RTX 4090')
    gpuCount          = 1
    computeType       = 'GPU'
    interruptible     = $false
    cloudType         = 'SECURE'
    containerDiskInGb = 50
    volumeInGb        = 30
    volumeMountPath   = '/workspace'
    ports             = @('22/tcp')
    supportPublicIp   = $true
    env               = @{
        SSH_PUBLIC_KEY = $publicKey
        PUBLIC_KEY     = $publicKey
    }
}

$pod = $null
try {
    $pod = Invoke-RestMethod `
        -Uri 'https://rest.runpod.io/v1/pods' `
        -Headers $headers `
        -ContentType 'application/json' `
        -Body ($body | ConvertTo-Json -Depth 6) `
        -Method Post
    $watchdog = Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            (Join-Path $RepositoryRoot 'training\runpod\watchdog.ps1'),
            '-PodId', $pod.id,
            '-MaximumMinutes', $MaximumMinutes,
            '-RepositoryRoot', $RepositoryRoot
        ) `
        -RedirectStandardOutput (Join-Path $tempRoot 'watchdog.log') `
        -RedirectStandardError (Join-Path $tempRoot 'watchdog-error.log') `
        -WindowStyle Hidden `
        -PassThru
    $state = [ordered]@{
        podId         = $pod.id
        name          = $pod.name
        desiredStatus = $pod.desiredStatus
        costPerHr     = $pod.costPerHr
        gpu           = $pod.gpu.displayName
        cloud         = 'SECURE'
        keyPath       = $keyPath
        tempRoot      = $tempRoot
        watchdogPid   = $watchdog.Id
        createdAt     = (Get-Date).ToUniversalTime().ToString('o')
    }
    $statePath = Join-Path $RepositoryRoot 'artifacts\stage11\stability\runpod-state.json'
    New-Item -ItemType Directory -Force -Path (Split-Path $statePath) | Out-Null
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    $state | ConvertTo-Json
} catch {
    if ($pod) {
        try {
            Invoke-RestMethod `
                -Uri "https://rest.runpod.io/v1/pods/$($pod.id)" `
                -Headers $headers `
                -Method Delete | Out-Null
        } catch {
            Write-Warning "Could not terminate failed stability Pod $($pod.id): $_"
        }
    }
    throw
}
