#!/usr/bin/env bash
#SBATCH --job-name=spillover-donor-weights
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-donor-weights-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-donor-weights-%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"

REPO_DIR=/home/jwinchel/Syndri
DATA_ROOT="${SYNDRI_DATA_ROOT:-/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx}"
OUTPUT_ROOT="${SYNDRI_WEIGHTED_OUTPUT:-$REPO_DIR/outputs/SyndriSIM_donor_weighted_exposed_20261120}"
PYTHON_BIN="${SYNDRI_PYTHON:-/home/jwinchel/miniconda3/envs/heimdall/bin/python}"
cd "$REPO_DIR"

"$PYTHON_BIN" -u scripts/spillover-benchmark/prepare_weighted_test_admixture.py \
  --training "$DATA_ROOT/spillover-benchmark/train_val.h5ad" \
  --test-dir "$DATA_ROOT/spillover-benchmark/test_admixture/slices" \
  --reference "$DATA_ROOT/spillover-benchmark/test_admixture/train_val_sc_reference_unique.h5ad" \
  --spatial-dir "$DATA_ROOT/results/final" \
  --markers "$DATA_ROOT/results/markers/pydeseq2/all_cell_type_vs_rest_deseq2.csv" \
  --output-dir "$OUTPUT_ROOT" \
  --calibration-slices Lung5_Rep1 Lung5_Rep2 Lung5_Rep3 \
  --pilot-slice Lung12 --pilot-cells-per-type 100 \
  --min-calibration-cells 50 --min-calibration-slices 2 --shrinkage .25 --seed 20261120 \
  --run-all
