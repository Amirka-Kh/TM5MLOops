# Exit immediately if a command exits with a non-zero status
$ErrorActionPreference = "Stop"

# Activate the virtual environment
& ..\venv\Scripts\Activate.ps1

# Take a data sample
Write-Output "Sampling data..."
python ..\src\sample_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Output "Failed to sample data."
    exit 1
}

# Validate the data sample
Write-Output "Validating data..."
python ..\src\validate_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Output "Data validation failed. Data will not be versioned."
    exit 1
}
Write-Output "Data validation passed."

# Version the data sample
Write-Output "Versioning data..."
git commit -m "Add sampled data $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") and version with DVC"
if ($LASTEXITCODE -ne 0) {
    Write-Output "Git commit failed."
    exit 1
}

git tag -a "v1.1" -m "add data version v1.1"
if ($LASTEXITCODE -ne 0) {
    Write-Output "Git tag failed."
    exit 1
}

dvc push
if ($LASTEXITCODE -ne 0) {
    Write-Output "DVC push failed."
    exit 1
}

git push
if ($LASTEXITCODE -ne 0) {
    Write-Output "Git push failed."
    exit 1
}
