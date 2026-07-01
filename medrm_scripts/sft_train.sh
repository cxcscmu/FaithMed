#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --output=/path/to/FaithMed/sft_train.out
#SBATCH --error=/path/to/FaithMed/sft_train.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

mkdir -p /path/to/FaithMed

echo "======================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "======================================"

cd /path/to/FaithMed
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent

# Triton autotune cache must be on local (non-NFS) storage
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

NUM_GPUS=4 bash search_agent/sft/run_sft.sh

echo "======================================"
echo "Job finished: $(date)"
echo "======================================"
