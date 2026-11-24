#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-method-panels
#SBATCH --partition=cpu
#SBATCH --nodelist=oven-1-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-method-panels-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-method-panels-%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
cd /home/jwinchel/Syndri
# Keep manual and dependent refreshes from writing the notebook concurrently.
exec flock outputs/SyndriSIM_pool_fraction/manifests/method_plot_render.lock \
  /home/jwinchel/miniconda3/envs/heimdall/bin/python -u \
  scripts/spillover-benchmark/render_pool_fraction_notebook.py \
  --notebook figures/Figure3_SyndriSIM_PoolFraction.ipynb \
  --cell-ids configuration method-results-data syndri-roc method-pr method-auprc \
    removal-plot-function removal-ranking-ap known-added-removal native-count-removal best-f1
