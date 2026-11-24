#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-transcript-methods
#SBATCH --partition=cpu
#SBATCH --nodelist=oven-1-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-transcript-methods-%x-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-transcript-methods-%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"
: "${POOL_TRANSCRIPT_METHOD_STAGE:?set prepare, celladmix, or ovrlpy}"
case "$POOL_TRANSCRIPT_METHOD_STAGE" in
  ovrlpy) interpreter=/home/jwinchel/miniconda3/envs/ovrlpy/bin/python ;;
  prepare|celladmix) interpreter=/home/jwinchel/miniconda3/envs/heimdall/bin/python ;;
  *) echo 'Unknown stage' >&2; exit 1 ;;
esac
cd /home/jwinchel/Syndri
exec "$interpreter" -u scripts/spillover-benchmark/run_pool_fraction_transcript_methods.py \
  --stage "$POOL_TRANSCRIPT_METHOD_STAGE" --task-id "${SLURM_ARRAY_TASK_ID:?}" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
