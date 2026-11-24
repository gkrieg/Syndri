#!/usr/bin/env bash
#SBATCH --job-name=syndrisim-pool-figures
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/syndrisim-pool-figures-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/syndrisim-pool-figures-%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd /home/jwinchel/Syndri
exec /home/jwinchel/miniconda3/envs/heimdall/bin/python -u \
  scripts/spillover-benchmark/render_pool_fraction_notebook.py
