$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv' }
}
& '.venv/Scripts/python.exe' -m pip install -e '.[dev,video]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Host 'Set OPENAI_API_KEY in .env, then run this script again.'
    exit 0
}
& '.venv/Scripts/browser-agent.exe' @args
exit $LASTEXITCODE

