param(
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ArchiveSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{40}$')][string]$SourceRevision,
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'CvdResearchDemo')
)
$ErrorActionPreference = 'Stop'
if ($Image -notmatch '^sha256:[a-f0-9]{64}$') {
    throw 'Deployment requires an immutable Docker image ID.'
}
$Destination = [IO.Path]::GetFullPath($Destination)
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ArchiveSha256) {
    throw 'Archive checksum mismatch; the current deployment is unchanged.'
}
& docker load --input $Archive
if ($LASTEXITCODE -ne 0) { throw 'Image import failed; the current deployment is unchanged.' }
& docker image inspect $Image | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'The expected image ID is missing from the archive.' }
$demoOutput = & docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs '/tmp:rw,noexec,nosuid,size=16m' $Image
if ($LASTEXITCODE -ne 0) { throw 'Container verification failed; the current deployment is unchanged.' }
$result = ($demoOutput -join "`n") | ConvertFrom-Json
if ($result.status -ne 'PASS' -or $result.source_revision -ne $SourceRevision -or
    $result.model_id -ne 'minimal_v1' -or $result.verified_binary_artifacts -ne 4 -or
    $result.clinical_deployment_allowed -ne $false -or $result.predictions.Count -ne 3) {
    throw 'Research demo contract failed; the current deployment is unchanged.'
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$releaseDir = Join-Path (Join-Path $Destination 'releases') $SourceRevision
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $releaseDir 'verification.json') -Encoding UTF8
$manifest = [ordered]@{
    status = 'DEPLOYED'; application = 'CVD research batch demo'; model_id = 'minimal_v1'
    image_reference = $Image; source_revision = $SourceRevision
    archive_sha256 = $ArchiveSha256
    deployed_utc = [DateTime]::UtcNow.ToString('o'); research_use_only = $true
    verification = (Join-Path $releaseDir 'verification.json')
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $releaseDir 'deployment.json') -Encoding UTF8
$launcher = @'
$ErrorActionPreference = 'Stop'
$deployment = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'current.json') | ConvertFrom-Json
& docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs '/tmp:rw,noexec,nosuid,size=16m' $deployment.image_reference
if ($LASTEXITCODE -ne 0) { throw 'The deployed research demonstration failed.' }
'@
$launcher | Set-Content -LiteralPath (Join-Path $Destination 'Run-CvdDemo.ps1') -Encoding UTF8
$current = Join-Path $Destination 'current.json'
if (Test-Path -LiteralPath $current) {
    Copy-Item -LiteralPath $current -Destination (Join-Path $Destination 'previous.json') -Force
}
$next = Join-Path $Destination 'current.next.json'
$manifest | ConvertTo-Json | Set-Content -LiteralPath $next -Encoding UTF8
Move-Item -LiteralPath $next -Destination $current -Force
Write-Output 'CD deployment PASS'
Write-Output ('Image: ' + $Image)
Write-Output ('Source commit: ' + $SourceRevision)
Write-Output ('Installed launcher: ' + (Join-Path $Destination 'Run-CvdDemo.ps1'))
$result | ConvertTo-Json -Depth 8
