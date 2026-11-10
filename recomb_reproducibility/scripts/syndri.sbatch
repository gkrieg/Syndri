#!/bin/bash
#SBATCH --job-name=syndri-recomb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=syndri-recomb-%j.log
# Submit from the repository root after activating your environment.
# Add your site's partition and GPU request to sbatch when training/inferencing.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
test -f run.py
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export NUMBA_NUM_THREADS=8 WANDB_MODE=disabled
TASK=${TASK:-reproduce}
DATASET=${DATASET:-NC}
PARAMS=${PARAMS:-configs/reproduction.yaml}
case "$TASK" in
  train|infer) python run.py --params "$PARAMS" "$TASK" --dataset "$DATASET" ;;
  evaluate|preprocess) python run.py --params "$PARAMS" "$TASK" --dataset "${DATASET:-all}" ;;
  reproduce) python run.py --params "$PARAMS" reproduce ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 2 ;;
esac

