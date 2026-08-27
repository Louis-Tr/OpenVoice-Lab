param(
    [string]$RepositoryRoot = 'D:\OpenVoice-Lab'
)

$ErrorActionPreference = 'Stop'
$statePath = Join-Path $RepositoryRoot 'artifacts\stage11\runpod-state.json'
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
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
try {
    Invoke-RestMethod `
        -Uri "https://rest.runpod.io/v1/pods/$($state.podId)" `
        -Headers $headers `
        -Method Delete | Out-Null
} catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 404) {
        throw
    }
}
if ($state.watchdogPid) {
    Stop-Process -Id $state.watchdogPid -ErrorAction SilentlyContinue
}
[pscustomobject]@{
    podId = $state.podId
    terminated = $true
    watchdogStopped = $true
} | ConvertTo-Json

