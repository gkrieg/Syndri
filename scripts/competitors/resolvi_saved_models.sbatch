#!/usr/bin/env bash
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/resolvi-saved-%x-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/resolvi-saved-%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
/home/jwinchel/miniconda3/envs/rctd/bin/python "$RUN_WORKER" "$RUN_MANIFEST" "$SLURM_ARRAY_TASK_ID" "$RUN_STAGE"
