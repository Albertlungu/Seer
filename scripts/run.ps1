$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $dir ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$ok = (Test-Path $python) -and ((& $python -c "import sys" 2>$null); $LASTEXITCODE -eq 0)
if (-not $ok) {
    Write-Host "Setting up virtual environment..."
    python -m venv $venv
    & (Join-Path $venv "Scripts\pip") install -r (Join-Path $dir "requirements-prod.txt") --quiet
}
& $python (Join-Path $dir "main.py")
