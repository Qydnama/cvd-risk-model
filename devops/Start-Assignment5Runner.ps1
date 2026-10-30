param([switch]$RegisterOnly)
$ErrorActionPreference = 'Stop'
$runnerRoot = Join-Path $env:USERPROFILE '.cache/cvd-assignment5/runner'
$archive = Join-Path (Split-Path -Parent $runnerRoot) 'actions-runner-win-x64-2.338.0.zip'
New-Item -ItemType Directory -Force -Path $runnerRoot | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $runnerRoot 'config.cmd'))) {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/actions/runner/releases/download/v2.338.0/actions-runner-win-x64-2.338.0.zip' -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f48e0750a21812bca5f82de5f7f5aeae71abee647fab5a582f1742d07eba455f') {
        throw 'Runner archive checksum mismatch'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $runnerRoot
}
if (-not (Test-Path -LiteralPath (Join-Path $runnerRoot '.runner'))) {
    $registration = gh api --method POST repos/Qydnama/cvd-risk-model/actions/runners/registration-token | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $registration.token) { throw 'GitHub runner registration failed' }
    Push-Location $runnerRoot
    try {
        & ./config.cmd --url 'https://github.com/Qydnama/cvd-risk-model' --token $registration.token --name 'cvd-assignment5-local' --labels 'cvd-assignment5' --work '_work' --unattended --ephemeral
        if ($LASTEXITCODE -ne 0) { throw 'Runner configuration failed' }
    } finally { Pop-Location; $registration = $null }
}
if (-not $RegisterOnly) {
    $process = Start-Process -FilePath (Join-Path $runnerRoot 'run.cmd') -WorkingDirectory $runnerRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runnerRoot 'runner-output.log') -RedirectStandardError (Join-Path $runnerRoot 'runner-error.log') -PassThru
    Write-Output ('Started one-job runner, process ' + $process.Id)
    Write-Output 'The runner unregisters automatically after one deployment job.'
}
