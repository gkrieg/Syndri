#!/usr/bin/env bash
#SBATCH --job-name=spillover-calibration
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-calibration-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-calibration-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

REPO_DIR=/home/jwinchel/Syndri
NOTEBOOK="$REPO_DIR/data/nsclc_cosmx/CalibrationExperiment.ipynb"
OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark

cd "$REPO_DIR"
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

mkdir -p "$OUTPUT_DIR"

echo "Running $NOTEBOOK on $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Memory: 64G"
echo "Output directory: $OUTPUT_DIR"

python -m nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output CalibrationExperiment.executed \
  --output-dir "$OUTPUT_DIR" \
  "$NOTEBOOK"
