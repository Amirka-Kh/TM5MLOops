#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment
source ../venv/bin/activate

# Take a data sample and validate it
echo "Sampling data and validating data.."

if python src/data.py; then
    echo "Data validation passed."

    # Version the data sample
    echo "Versioning data..."
    git commit -m "Add sampled data $EPOCHSECONDS and version with DVC"
    dvc push
    git push
else
    echo "Data validation failed. Data will not be versioned."
    exit 1
fi