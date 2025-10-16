#!/bin/bash

# Helper script to submit experiments to SLURM
# Usage: ./submit_experiment.sh <config_name> [env_file]
#
# Example:
#   ./submit_experiment.sh learning_rate_high
#   ./submit_experiment.sh learning_rate_medium .env.production

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_name> [env_file]"
    echo ""
    echo "Available configs:"
    ls configs/*.json | xargs -n 1 basename | sed 's/.json$//'
    exit 1
fi

CONFIG_NAME=$1
ENV_FILE=${2:-.env}

CONFIG_FILE="configs/${CONFIG_NAME}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo ""
    echo "Available configs:"
    ls configs/*.json | xargs -n 1 basename | sed 's/.json$//'
    exit 1
fi

echo "Submitting job with config: $CONFIG_FILE"
echo "Environment file: $ENV_FILE"

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export variables for the SLURM script
export CONFIG_FILE
export CONFIG_NAME
export ENV_FILE
export PROJECT_DIR

# Submit the job with dynamic job name
sbatch --export=ALL --job-name="${CONFIG_NAME}" run_experiment.sh

echo "Job submitted!"
echo "Monitor with: squeue -u \$USER"
echo "View logs in: logs/"
