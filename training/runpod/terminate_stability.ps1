param(
    [string]$RepositoryRoot = 'D:\OpenVoice-Lab'
)

$ErrorActionPreference = 'Stop'
$statePath = Join-Path $RepositoryRoot 'artifacts\stage11\stability\runpod-state.json'
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
foreach ($path in @($state.keyPath, "$($state.keyPath).pub")) {
    if ($path -and (Test-Path -LiteralPath $path)) {
        Remove-Item -LiteralPath $path -Force
    }
}
[pscustomobject]@{
    podId = $state.podId
    terminated = $true
    watchdogStopped = $true
    temporaryKeysRemoved = $true
} | ConvertTo-Json
