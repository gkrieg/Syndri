#!/usr/bin/env bash
#SBATCH --job-name=celladmix-factor23-xenium-32g4c
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/celladmix-factor23-xenium-32g4c-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/celladmix-factor23-xenium-32g4c-%A_%a.err
set -euo pipefail
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMPETITOR_CELLADMIX_MANIFEST")
IFS=$'\t' read -r final ntypes k <<< "$line"
test -f "$final"
test "$k" -eq 23
dataset=$(basename "$(dirname "$(dirname "$(dirname "$final")")")")
name=$(basename "$final" .h5ad)
out="/work/magroup/skrieger/Syndri/data/datasets/$dataset/results/competitors/celladmix/factorization_k1p5ct/$name"
mkdir -p "$out"
test ! -e "$out/assignments.csv"
test ! -e "$out/factor_summary.csv"
test ! -e "$out/factor_loadings.csv"
scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/celladmix-k1p5ct.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
/home/jwinchel/miniconda3/envs/rctd/bin/python /home/jwinchel/Syndri/scripts/competitors/stage_celladmix.py "$final" "$scratch/transcripts.csv"
export CELLADMIX_K="$k" CELLADMIX_CORES=4
cd "$scratch"
/home/jwinchel/miniconda3/envs/r_env/bin/Rscript -e 'run_knn_nmf <- function(df,k,h,nmol.dsamp,n.cores) cellAdmix::run_knn_nmf(df,k=as.integer(Sys.getenv("CELLADMIX_K")),h=h,nmol.dsamp=nmol.dsamp,n.cores=n.cores); source("/home/jwinchel/Syndri/scripts/competitors/run_celladmix.R")' "$scratch/transcripts.csv" "$out/assignments.csv" "$out/factor_summary.csv" "$out/factor_loadings.csv"
echo "completed $name with $k factors ($ntypes cell types), 4 workers"
