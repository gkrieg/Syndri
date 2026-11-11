#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/all-type-cosine-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/all-type-cosine-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
/home/jwinchel/miniconda3/envs/heimdall/bin/python -u "$COSINE_RUN/worker.py" "$COSINE_RUN/manifest.json" "$SLURM_ARRAY_TASK_ID"
