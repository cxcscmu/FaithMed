#!/bin/bash
#SBATCH --job-name=medrm_agent
#SBATCH --output=/path/to/log/slurm_log/%x_%j.out
#SBATCH --error=/path/to/log/slurm_log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=1-00:00:00

set -euo pipefail
set -x

# --- GPU detection ---
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ "$GPU_MODEL" == *"A6000"* || "$GPU_MODEL" == *"L40S"* ]]; then
    echo "Detected $GPU_MODEL, disabling NCCL P2P"
    export NCCL_P2P_DISABLE=1
else
    echo "Detected $GPU_MODEL, keeping NCCL P2P enabled"
fi

# --- Ray tmp setup and pre-clean ---
# Keep Ray logs/spilled objects off /tmp. /tmp can be small or shared on compute nodes,
# while /data has enough capacity for long rollout jobs.
JOB_TMP_ID="${SLURM_JOB_ID:-manual_$$}"
TMP_ROOT="/path/to/tmp"
RAY_TMP_ROOT="$TMP_ROOT/ray"
export TMPDIR="$TMP_ROOT/$JOB_TMP_ID"
export RAY_TMPDIR="$RAY_TMP_ROOT/$JOB_TMP_ID"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

echo "[pre-clean] host=$(hostname)"
echo "[pre-clean] TMPDIR=$TMPDIR"
echo "[pre-clean] RAY_TMPDIR=$RAY_TMPDIR"
echo "[pre-clean] df -h /tmp $TMP_ROOT (before)"
df -h /tmp "$TMP_ROOT" || true

# Clean legacy Ray sessions from /tmp, left by older jobs that did not set RAY_TMPDIR.
if [ -d /tmp/ray ]; then
    echo "[pre-clean] legacy /tmp/ray size (before):"
    du -sh /tmp/ray 2>/dev/null || true

    echo "[pre-clean] removing legacy $USER /tmp/ray sessions older than 1 day..."
    find /tmp/ray -maxdepth 1 -type d -name "session_*" -user "$USER" -mtime +1 -print -exec rm -rf {} \; 2>/dev/null || true

    echo "[pre-clean] legacy /tmp/ray size (after):"
    du -sh /tmp/ray 2>/dev/null || true
fi

# Clean old per-job Ray tmp dirs on /data. Keep the current job directory.
if [ -d "$RAY_TMP_ROOT" ]; then
    echo "[pre-clean] $RAY_TMP_ROOT size (before):"
    du -sh "$RAY_TMP_ROOT" 2>/dev/null || true

    echo "[pre-clean] removing $USER Ray tmp job dirs on /data older than 1 day..."
    find "$RAY_TMP_ROOT" -mindepth 1 -maxdepth 1 -type d -user "$USER" -mtime +1 -print -exec rm -rf {} \; 2>/dev/null || true

    echo "[pre-clean] $RAY_TMP_ROOT size (after):"
    du -sh "$RAY_TMP_ROOT" 2>/dev/null || true
fi

echo "[pre-clean] df -h /tmp $TMP_ROOT (after)"
df -h /tmp "$TMP_ROOT" || true
# --- end Ray tmp setup and pre-clean ---

time=$(date +%Y%m%d_%H%M%S)

REPO_ROOT="${SLURM_SUBMIT_DIR:-/path/to/FaithMed}"
cd "$REPO_ROOT/verl-agent"
source ~/miniconda3/bin/activate verl-agent


VERL_OUTPUT_FILE="/path/to/log/verl_${SLURM_JOB_ID}.out"

echo "=== Starting FaithMed Agentic RL training: job=${SLURM_JOB_ID} at ${time} ==="

stdbuf -oL -eL ./examples/faithmed_trainer/run_medrm.sh "$@" \
    > "$VERL_OUTPUT_FILE" 2>&1 &

wait
