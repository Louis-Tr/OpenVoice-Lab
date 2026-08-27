param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$PodId,

    [ValidateRange(10, 180)]
    [int]$MaximumMinutes = 75,

    [string]$RepositoryRoot = 'D:\OpenVoice-Lab'
)

$ErrorActionPreference = 'Stop'
$envPath = Join-Path $RepositoryRoot '.env'
$apiKey = $null
foreach ($line in Get-Content -LiteralPath $envPath) {
    if ($line -match '^RUNPOD_API=(.+)$') {
        $apiKey = $Matches[1].Trim('"', "'")
        break
    }
}
if (-not $apiKey) {
    throw 'RUNPOD_API is missing from .env'
}

Start-Sleep -Seconds ($MaximumMinutes * 60)
$headers = @{ Authorization = "Bearer $apiKey" }
try {
    Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" -Headers $headers -Method Delete | Out-Null
} catch {
    # A 404 means the primary process already terminated the Pod. Any other
    # error is written to the watchdog log by the caller's redirected streams.
    if ($_.Exception.Response.StatusCode.value__ -ne 404) {
        throw
    }
}

