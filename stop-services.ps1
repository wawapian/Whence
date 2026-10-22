$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$recordPath = Join-Path $projectRoot 'backend\.runtime\started-processes.json'
if (-not (Test-Path -LiteralPath $recordPath)) { Write-Host 'No owned processes recorded.'; exit 0 }
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
foreach ($role in @('workspace', 'embedding', 'reranker')) {
    $ownedId = $record.$role
    if (-not $ownedId) { continue }
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$ownedId)"
    if (-not $processInfo) { continue }
    $commandText = [string]$processInfo.CommandLine
    $expectedProgram = if ($role -eq 'workspace') { 'uvicorn' } else { 'llama-server' }
    if ($commandText.IndexOf($projectRoot, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or $commandText -notlike "*$expectedProgram*") {
        Write-Warning "Skipped PID $ownedId; it does not match this project."
        continue
    }
    & "$env:SystemRoot\System32\taskkill.exe" /PID ([int]$ownedId) /T /F
}
