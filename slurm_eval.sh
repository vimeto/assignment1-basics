#!/bin/bash
#SBATCH --job-name=eval_debug
#SBATCH --account=project_2013932
#SBATCH --time=01:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Usage: sbatch slurm_eval.sh <checkpoint_path> <config_json>
# Example: sbatch slurm_eval.sh /scratch/project_2013932/vtoivone/checkpoints/learning_rate_medium/step_00004500.pt configs/eval_learning_rate_medium.json

set -e  # Exit on error

echo "========================================"
echo "SLURM Evaluation Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Parse arguments
CONFIG_JSON=${1:-""}

if [ -z "$CONFIG_JSON" ]; then
    echo "ERROR: Config JSON not provided"
    echo "Usage: sbatch slurm_eval.sh <eval_config.json>"
    echo "Example: sbatch slurm_eval.sh configs/eval_learning_rate_medium.json"
    exit 1
fi

echo "Config: $CONFIG_JSON"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module purge
module load pytorch/2.4

# Activate virtual environment
source .venv/bin/activate

# Print Python environment info
echo "========================================"
echo "Environment Info"
echo "========================================"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo ""

# Enable CUDA debugging for exact error location
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "========================================"
echo "CUDA Debug Mode Enabled"
echo "========================================"
echo "CUDA_LAUNCH_BLOCKING=1"
echo "TORCH_USE_CUDA_DSA=1"
echo ""

# Parse config JSON to extract parameters
echo "========================================"
echo "Parsing Configuration"
echo "========================================"

VOCAB_SIZE=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['vocab_size'])")
CONTEXT_LENGTH=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['context_length'])")
D_MODEL=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['d_model'])")
NUM_LAYERS=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['num_layers'])")
NUM_HEADS=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['num_heads'])")
D_FF=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['d_ff'])")
ROPE_THETA=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['model']['rope_theta'])")

CHECKPOINT_PATH=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['checkpoint']['path'])")
VAL_PATH=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['data']['val_path'])")
BATCH_SIZE=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['training']['batch_size'])")
EVAL_BATCHES=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['training']['eval_batches'])")
PRECISION=$(python -c "import json; print(json.load(open('$CONFIG_JSON'))['training']['precision'])")

echo "Model config:"
echo "  vocab_size: $VOCAB_SIZE"
echo "  context_length: $CONTEXT_LENGTH"
echo "  d_model: $D_MODEL"
echo "  num_layers: $NUM_LAYERS"
echo "  num_heads: $NUM_HEADS"
echo "  d_ff: $D_FF"
echo "  rope_theta: $ROPE_THETA"
echo ""
echo "Checkpoint:"
echo "  path: $CHECKPOINT_PATH"
echo ""
echo "Data config:"
echo "  val_path: $VAL_PATH"
echo ""
echo "Eval config:"
echo "  batch_size: $BATCH_SIZE"
echo "  eval_batches: $EVAL_BATCHES"
echo "  precision: $PRECISION"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if validation data exists
if [ ! -f "$VAL_PATH" ]; then
    echo "ERROR: Validation data not found: $VAL_PATH"
    exit 1
fi

# Run evaluation
echo "========================================"
echo "Running Evaluation"
echo "========================================"
echo "Command: python cs336_basics/scripts/eval_only.py \\"
echo "  --checkpoint $CHECKPOINT_PATH \\"
echo "  --val-path $VAL_PATH \\"
echo "  --vocab-size $VOCAB_SIZE \\"
echo "  --context-length $CONTEXT_LENGTH \\"
echo "  --d-model $D_MODEL \\"
echo "  --num-layers $NUM_LAYERS \\"
echo "  --num-heads $NUM_HEADS \\"
echo "  --d-ff $D_FF \\"
echo "  --rope-theta $ROPE_THETA \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --eval-batches $EVAL_BATCHES \\"
echo "  --precision $PRECISION \\"
echo "  --device cuda \\"
echo "  --debug"
echo ""

python cs336_basics/scripts/eval_only.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --val-path "$VAL_PATH" \
    --vocab-size "$VOCAB_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --d-model "$D_MODEL" \
    --num-layers "$NUM_LAYERS" \
    --num-heads "$NUM_HEADS" \
    --d-ff "$D_FF" \
    --rope-theta "$ROPE_THETA" \
    --batch-size "$BATCH_SIZE" \
    --eval-batches "$EVAL_BATCHES" \
    --precision "$PRECISION" \
    --device cuda \
    --debug

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job Complete"
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check the error output above for the exact line that failed"
    echo "2. Look for 'NaN', 'Inf', or 'device' related messages"
    echo "3. Check the full log file: logs/eval_${SLURM_JOB_ID}.err"
    echo "4. Try with a smaller --eval-batches value (e.g., 5)"
    echo "5. Try with --precision fp32 if using mixed precision"
fi

exit $EXIT_CODE
