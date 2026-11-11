#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/shared-syndri-qc-%x-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/shared-syndri-qc-%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
exec /home/jwinchel/miniconda3/envs/heimdall/bin/python -u "$SHARED_QC_SCRIPT" "$SHARED_QC_MANIFEST" "$SLURM_ARRAY_TASK_ID"
