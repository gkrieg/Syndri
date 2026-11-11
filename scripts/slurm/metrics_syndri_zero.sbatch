#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-zero-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/syndri-zero-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
/home/jwinchel/miniconda3/envs/heimdall/bin/python -u "$ZERO_METRICS_WORKER" "$ZERO_METRICS_MANIFEST" "$SLURM_ARRAY_TASK_ID"
