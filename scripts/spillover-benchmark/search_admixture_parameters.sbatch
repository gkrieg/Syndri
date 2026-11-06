#!/usr/bin/env bash
#SBATCH --job-name=spillover-param-search
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-param-search-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-param-search-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

python -u /home/jwinchel/Syndri/scripts/spillover-benchmark/search_admixture_parameters.py \
  --max-query-per-cell-type 1000 \
  --seeds 0 1 2
