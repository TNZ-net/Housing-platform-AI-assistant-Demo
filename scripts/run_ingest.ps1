Set-Location (Join-Path $PSScriptRoot "..")
& .\.venv\Scripts\python.exe -m hp.ingest
exit $LASTEXITCODE