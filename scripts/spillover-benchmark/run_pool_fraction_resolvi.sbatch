#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-resolvi
#SBATCH --partition=gpu-large
#SBATCH --nodelist=oven-1-3
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-resolvi-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-resolvi-%A_%a.err
set -euo pipefail
exec bash /home/jwinchel/Syndri/scripts/spillover-benchmark/run_resolvi_test.sbatch
