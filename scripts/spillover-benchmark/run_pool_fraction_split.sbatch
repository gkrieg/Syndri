#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-split
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-split-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-split-%A_%a.err
set -euo pipefail
export PATH=/home/jwinchel/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin
unset VIRTUAL_ENV _OLD_VIRTUAL_PATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
exec bash /home/jwinchel/Syndri/scripts/slurm/run_split_h5ad_layers.sbatch
