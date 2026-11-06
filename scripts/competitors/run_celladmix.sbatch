#!/usr/bin/env bash
#SBATCH --job-name=competitor-celladmix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/competitor-celladmix-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/competitor-celladmix-%A_%a.err
set -euo pipefail
final=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMPETITOR_CELLADMIX_MANIFEST")
[[ -f "$final" ]] || { echo 'Missing final AnnData' >&2; exit 1; }
dataset=$(basename "$(dirname "$(dirname "$(dirname "$final")")")")
name=$(basename "$final" .h5ad)
out="/work/magroup/skrieger/Syndri/data/datasets/$dataset/results/competitors/celladmix/factorization/$name"
mkdir -p "$out"
[[ ! -e "$out/assignments.csv" && ! -e "$out/factor_summary.csv" && ! -e "$out/factor_loadings.csv" ]] || {
  echo 'Factorization outputs already exist; refusing to overwrite' >&2; exit 1;
}
scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/celladmix.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
/home/jwinchel/miniconda3/envs/rctd/bin/python \
  /home/jwinchel/Syndri/scripts/competitors/stage_celladmix.py "$final" "$scratch/transcripts.csv"
cd "$scratch"
/home/jwinchel/miniconda3/envs/r_env/bin/Rscript \
  /home/jwinchel/Syndri/scripts/competitors/run_celladmix.R \
  "$scratch/transcripts.csv" "$out/assignments.csv" "$out/factor_summary.csv" "$out/factor_loadings.csv"
