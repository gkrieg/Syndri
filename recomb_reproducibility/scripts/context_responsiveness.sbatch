#!/bin/bash
#SBATCH --job-name=syndri-context
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=syndri-context-%j.log

set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_DIR"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export WANDB_MODE=disabled
# Activate the package environment before sbatch. Add partition/GPU flags on submission.
# STAGE=all regenerates contexts, runs the bundled BX checkpoint, analyzes, and plots.
# STAGE=plot (default) only renders bundled numerical results on CPU.
python scripts/context_responsiveness.py --params "${PARAMS:-configs/reproduction.yaml}" --stage "${STAGE:-plot}"
