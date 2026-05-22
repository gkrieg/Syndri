#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --mem=64Gb
#SBATCH --job-name=admix_lr
#SBATCH --output=/home/jwinchel/Syndri/logs/admixture_lr_%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/admixture_lr_%A_%a.err
#SBATCH -t 0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

set -euo pipefail

# Edit this list and the SBATCH --array range together if you add/remove rates.
LEARNING_RATES=(
  1e-4
  3e-4
  1e-3
  3e-3
  1e-2
)

LR="${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}"
LR_TAG="${LR//./p}"
LR_TAG="${LR_TAG//-/_}"

cd /home/jwinchel/Syndri/Heimdall
mkdir -p logs admixture_prediction_results/lr_sweep

module load cuda-12.1

source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

echo "Running admixture_prediction learning-rate sweep"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "LR=${LR}"

python train.py \
  +experiments=admixture_prediction \
  optimizer.args.lr="${LR}" \
  run_name="admixture_prediction_lr_${LR_TAG}" \
  work_dir="admixture_prediction_results/lr_sweep/lr_${LR_TAG}"
