#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-syndri-train
#SBATCH --partition=gpu-large
#SBATCH --nodelist=oven-1-3
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-syndri-train-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-syndri-train-%j.err
set -euo pipefail
module load cuda-12.1
export PATH=/home/jwinchel/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin
unset VIRTUAL_ENV _OLD_VIRTUAL_PATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_MODE=online
cd /home/jwinchel/Syndri/Heimdall
exec /home/jwinchel/miniconda3/envs/heimdall/bin/python -u \
  /home/jwinchel/Syndri/scripts/spillover-benchmark/train_pool_fraction_syndri.py
