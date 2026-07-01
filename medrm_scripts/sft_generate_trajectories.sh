#!/bin/bash
#SBATCH --job-name=sft_traj
#SBATCH --output=/path/to/sft_traj.out
#SBATCH --error=/path/to/sft_traj.err
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

mkdir -p /path/to/FaithMed

echo "======================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "======================================"

cd /path/to/FaithMed
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent

python -m search_agent.sft.generate_trajectories \
    --model bedrock \
    --search_engine medcorp \
    --sample_limit 1000 500 1000 500 \
    --workers 8

echo "======================================"
echo "Job finished: $(date)"
echo "======================================"
