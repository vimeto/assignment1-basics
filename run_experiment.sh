#!/bin/bash
#SBATCH --job-name=cs336_train
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config file: $CONFIG_FILE"
echo "Environment file: $ENV_FILE"

# Load modules
module purge
module load pytorch

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify uv is available
uv --version

# Use the PROJECT_DIR passed from submit script, or try to detect it
if [ -z "$PROJECT_DIR" ]; then
    echo "Warning: PROJECT_DIR not set, attempting to detect..."
    PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

echo "Project directory: $PROJECT_DIR"

# Change to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Set default values if not provided
CONFIG_FILE=${CONFIG_FILE:-"configs/001.json"}
ENV_FILE=${ENV_FILE:-".env"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Looking in: $(pwd)"
    echo "Directory contents:"
    ls -la
    exit 1
fi

echo "Using config: $CONFIG_FILE"

# Load environment variables from .env file if it exists
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from: $ENV_FILE"
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
else
    echo "Warning: Environment file $ENV_FILE not found. Continuing without it."
fi

# Run the training script with uv
echo "Starting training..."
uv run python cs336_basics/scripts/train_transformer.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "logs/training_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
