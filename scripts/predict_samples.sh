#!/bin/bash

# Versions to iterate over
VERSIONS=("1.19" "1.18" "1.7")

# Random states to iterate over (example: incremental from 0 to 9)
# Uncomment the following line to use a range
# RANDOM_STATES=$(seq 0 9)

# Or use a predefined list of random states
RANDOM_STATES=("14" "42")

# Loop over each version
for VERSION in "${VERSIONS[@]}"; do
  # Loop over each random state
  for RANDOM_STATE in "${RANDOM_STATES[@]}"; do
    # Run the mlflow command with the current version and random state
    echo "Running mlflow with version=${VERSION} and random_state=${RANDOM_STATE}..."
    mlflow run . --env-manager local -e predict -P version=${VERSION} -P random_state=${RANDOM_STATE}
  done
done
