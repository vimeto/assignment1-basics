#!/bin/bash
#SBATCH --job-name=cs336_train
#SBATCH --account=project_2013932
#SBATCH --partition=gpu
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

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p /scratch/project_2013932/vtoivone/checkpoints

# Set default values if not provided
CONFIG_FILE=${CONFIG_FILE:-"configs/001.json"}
ENV_FILE=${ENV_FILE:-".env"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
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
