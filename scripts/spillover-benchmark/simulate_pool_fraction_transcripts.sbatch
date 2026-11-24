#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-transcripts
#SBATCH --partition=cpu
#SBATCH --nodelist=oven-1-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-transcripts-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-transcripts-%A_%a.err
set -euo pipefail
# Sampling uses one process per allocated CPU; each process uses one BLAS thread.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
cd /home/jwinchel/Syndri
exec /home/jwinchel/miniconda3/envs/heimdall/bin/python -u \
  scripts/spillover-benchmark/run_pool_fraction_transcripts.py \
  --stage "${POOL_TRANSCRIPT_STAGE:?set POOL_TRANSCRIPT_STAGE to pilot or full}" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
