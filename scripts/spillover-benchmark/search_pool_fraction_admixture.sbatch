#!/usr/bin/env bash
#SBATCH --job-name=spillover-pool-fraction
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-pool-fraction-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-pool-fraction-%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"

REPO_DIR=/home/jwinchel/Syndri
DATA_ROOT="${SYNDRI_DATA_ROOT:-/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx}"
OUTPUT_ROOT="${SYNDRI_POOL_OUTPUT:-$REPO_DIR/outputs/SyndriSIM_pool_fraction}"
WEIGHT_ROOT="${SYNDRI_DONOR_WEIGHT_ROOT:-$REPO_DIR/outputs/SyndriSIM_donor_weighted_exposed_20261120}"
PYTHON_BIN="${SYNDRI_PYTHON:-/home/jwinchel/miniconda3/envs/heimdall/bin/python}"
cd "$REPO_DIR"

"$PYTHON_BIN" -u scripts/spillover-benchmark/search_pool_fraction_admixture.py \
  --data-root "$DATA_ROOT" --donor-weight-root "$WEIGHT_ROOT" --output-dir "$OUTPUT_ROOT" \
  --calibration-slices Lung5_Rep1 Lung5_Rep2 Lung5_Rep3 \
  --lambda-max-values .1 .25 .5 .75 1. --seeds 0 1 2 \
  --max-query-per-cell-type 500 --min-evaluation-cells 50 \
  --query-seed 91827 --seed 20261120 --batch-size 256 \
  --pilot-slice Lung12 --pilot-cells-per-type 100 --run-all
