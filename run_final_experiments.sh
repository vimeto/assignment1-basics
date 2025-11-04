#!/usr/bin/env bash
set -euo pipefail

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Array of config files
CONFIGS=(
  "configs/owt_speedrun_lrpush.json"
  "configs/owt_speedrun_x0mix.json"
  "configs/owt_speedrun_batchscale.json"
  "configs/owt_speedrun_decayplus.json"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Run each config with its own run name. Timeout after 2 hours (7200 seconds).
for idx in "${!CONFIGS[@]}"; do
  run_name="owt-final-$((idx + 1))"
  config="${CONFIGS[idx]}"
  log_file="logs/${run_name}_$(date +%Y%m%d_%H%M%S).log"
  echo "Launching $run_name with $config (logging to $log_file)"
  timeout 7200 uv run python cs336_basics/scripts/train.py \
    --config "$config" \
    --wandb-run-name "$run_name" 2>&1 | tee "$log_file"
done
