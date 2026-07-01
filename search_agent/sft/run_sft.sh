#!/bin/bash
# Usage: NUM_GPUS=2 bash run_sft.sh
#
# llamafactory-cli train auto-detects GPUs via NPROC_PER_NODE env var.
# NUM_GPUS sets NPROC_PER_NODE; if unset, all visible GPUs are used.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Triton autotune cache must be on local (non-NFS) storage to avoid SIGBUS
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-$$}

# Disable NCCL P2P for GPUs that don't support it (e.g. A6000, L40S)
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_MODEL" == *"A6000"* || "$GPU_MODEL" == *"L40S"* ]]; then
    echo "Detected $GPU_MODEL, disabling NCCL P2P"
    export NCCL_P2P_DISABLE=1
fi

# Load WANDB_API_KEY (KEYS_ENV_PATH is set in .bashrc)
export WANDB_API_KEY=$(grep -E "^WANDB_API_KEY=" "${KEYS_ENV_PATH}" | cut -d'=' -f2- | tr -d '"'"'"' ')
export WANDB_PROJECT="${WANDB_PROJECT:-medmix-sft}"

# Use SLURM job ID if available, otherwise timestamp
RUN_ID="${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}"
BASE_OUTPUT_DIR="/path/to/SFT/qwen3-1.7b"

# Log file
mkdir -p "$SCRIPT_DIR/log"
LOG_FILE="$SCRIPT_DIR/log/training_${RUN_ID}.log"
echo "Run ID: ${RUN_ID}"
echo "Logging to: $LOG_FILE"

# llamafactory-cli internally calls torchrun using NPROC_PER_NODE
NPROC_PER_NODE="${NUM_GPUS:-}" \
llamafactory-cli train "$SCRIPT_DIR/train_sft.yaml" \
    output_dir="${BASE_OUTPUT_DIR}/output_${RUN_ID}" \
    logging_dir="${BASE_OUTPUT_DIR}/output_${RUN_ID}/logs" \
    run_name="qwen3_1.7b_medmix_sft_${RUN_ID}" \
    2>&1 | tee "$LOG_FILE"
