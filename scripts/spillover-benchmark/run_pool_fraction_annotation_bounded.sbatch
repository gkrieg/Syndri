#!/usr/bin/env bash
# Reuse completed factorization; exact global KNN in chunks, serial bridge tests.
#SBATCH --job-name=poolfraction-celladmix-bounded-annotation
#SBATCH --partition=cpu
#SBATCH --nodelist=oven-1-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-celladmix-bounded-annotation-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-celladmix-bounded-annotation-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CELLADMIX_ANNOTATION_BACKEND=bounded
cd /home/jwinchel/Syndri
exec /home/jwinchel/miniconda3/envs/heimdall/bin/python -u \
  scripts/spillover-benchmark/run_pool_fraction_transcript_methods.py \
  --stage celladmix --task-id "${SLURM_ARRAY_TASK_ID:?}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
