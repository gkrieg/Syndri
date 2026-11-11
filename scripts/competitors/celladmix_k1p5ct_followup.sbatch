#!/usr/bin/env bash
#SBATCH --job-name=celladmix-followup
#SBATCH --partition=gpu-small,gpu-large,cpu
#SBATCH --exclude=oven-0-13,oven-0-17
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/celladmix-followup-%x-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/celladmix-followup-%x-%A_%a.err
set -euo pipefail
: "${CELLADMIX_FOLLOWUP_MANIFEST:?}"
: "${CELLADMIX_STAGE:?}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CELLADMIX_FOLLOWUP_MANIFEST")
IFS=$'\t' read -r final ntypes k <<< "$line"
test -f "$final"
dataset=$(basename "$(dirname "$(dirname "$(dirname "$final")")")")
case "$dataset" in
  breast_xenium)
    label=harmony_cell_type; cx=x_centroid; cy=y_centroid; radius=15
    reference=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad
    reference_label=cell_type ;;
  nsclc_cosmx)
    label=cell_type_harmonized; cx=CenterX_global_px; cy=CenterY_global_px; radius=50
    reference=/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/interim/single-cell/sc_integrated.h5ad
    reference_label=cell_type_harmonized ;;
  *) echo "Unsupported dataset: $dataset" >&2; exit 1 ;;
esac
name=$(basename "$final" .h5ad)
base=/work/magroup/skrieger/Syndri/data/datasets/$dataset/results
fit=$base/competitors/celladmix/factorization_k1p5ct/$name
annotation=$base/competitors/celladmix/annotation_k1p5ct/$name
corrected=$base/competitors/celladmix_k1p5ct/$name.h5ad
metrics_root=$base/metrics/competitors/celladmix_k1p5ct
case "$CELLADMIX_STAGE" in
  annotation)
    if [[ -s "$annotation/factor_map.csv" && -s "$annotation/bridge_raw.rds" && -s "$annotation/bridge_res.rds" ]]; then
      echo "Existing annotation: $annotation"; exit 0
    fi
    test -s "$fit/assignments.csv"
    scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/celladmix-bridge.XXXXXXXX")
    trap 'rm -rf "$scratch"' EXIT
    /home/jwinchel/miniconda3/envs/rctd/bin/python \
      /home/jwinchel/Syndri/scripts/competitors/stage_celladmix.py "$final" "$scratch/transcripts.csv"
    /home/jwinchel/miniconda3/envs/r_env/bin/Rscript \
      /home/jwinchel/Syndri/scripts/competitors/annotate_celladmix_k1p5ct.R \
      "$scratch/transcripts.csv" "$fit/assignments.csv" "$k" "$annotation" "${SLURM_CPUS_PER_TASK:-8}"
    ;;
  correction)
    if [[ -s "$corrected" ]]; then echo "Existing corrected AnnData: $corrected"; exit 0; fi
    test -s "$annotation/factor_map.csv"
    /home/jwinchel/miniconda3/envs/rctd/bin/python \
      /home/jwinchel/Syndri/scripts/competitors/export_celladmix.py \
      "$final" "$fit/assignments.csv" "$annotation/factor_map.csv" \
      --method-name celladmix_k1p5ct
    ;;
  metrics)
    test -s "$corrected"
    metrics=$metrics_root/non_tcell/$name
    mkdir -p "$metrics"
    non_t=$metrics/non_tcell_metrics.csv
    if [[ ! -e "$non_t" ]]; then
      /home/jwinchel/miniconda3/envs/heimdall/bin/python \
        /home/jwinchel/Syndri/eval/metrics_eval.py \
        --input-files "$corrected" --output-dir "$metrics" \
        --cell-type-output non_tcell_metrics.csv \
        --config-output non_tcell_metrics_config.json \
        --label-key "$label" --exclude-labels '' \
        --additional-counts-layer celladmix_corrected_counts \
        --additional-counts-case-label celladmix_k1p5ct \
        --only-additional-counts --n-pcs 50 --library-size-target-sum 10000 \
        --leiden-resolution 1 --leiden-optimize-resolution false \
        --leiden-n-neighbors 15 --sample-frac 1.0 --sample-random-state 0 \
        --n-jobs "${SLURM_CPUS_PER_TASK:-8}" --random-state 0 --skip-tcell-contamination \
        --skip-tcell-neighborhood-metrics --skip-tcell-cosine-similarity \
        --skip-cell-type-profiles
    fi
    /home/jwinchel/miniconda3/envs/heimdall/bin/python - "$non_t" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
cols = ['silhouette_score', 'calinski_harabasz', 'davies_bouldin',
        'leiden_nmi_vs_label', 'leiden_adjusted_rand_vs_label',
        'median_genes_per_cell', 'median_transcripts_per_cell']
if len(df) != 1 or df['error'].fillna('').astype(str).str.len().any() or df[cols].isna().any().any():
    raise SystemExit(f'Incomplete non-T-cell metrics: {sys.argv[1]}')
PY
    tdir=$metrics_root/tcell/$name
    if [[ -e "$tdir/tcell_contamination_metrics.csv" ]]; then
      echo "Existing T-cell metrics: $tdir"; exit 0
    fi
    mkdir -p "$tdir"
    /home/jwinchel/miniconda3/envs/heimdall/bin/python \
      /home/jwinchel/Syndri/eval/metrics_eval.py \
      --input-files "$corrected" --output-dir "$tdir" \
      --config-output tcell_metrics_config.json \
      --skip-cell-type-separation --label-key "$label" --tcell-cell-type-key "$label" \
      --tcell-sample-key _metrics_tissue --coord-key _tcell_metrics_coordinates \
      --coord-x "$cx" --coord-y "$cy" --tcell-radius-um "$radius" \
      --exclude-labels '' --filter-tcell-invalid-cells \
      --admix-probs-layer syndri_admix_probs_X \
      --additional-counts-layer celladmix_corrected_counts \
      --additional-counts-case-label celladmix_k1p5ct --only-additional-counts \
      --cell-type-profile-output tissue_cell_type_profiles.csv \
      --purified-condition-label celladmix_k1p5ct --uncorrected-condition-label raw_X \
      --reference-data "$reference" --reference-cell-type-key "$reference_label" \
      --reference-counts-layer '' \
      --tcell-cosine-cell-type-pattern '(?i)(T CD[48]|Treg|CD4-positive|cytotoxic T|T-regulatory)' \
      --tcell-top-n 20 --tcell-min-cells-per-label 100 \
      --tcell-downsample-if-n-cells 50000 --tcell-max-cells-per-label 25000 \
      --tcell-exact-zscore auto --sample-frac 1.0 --sample-random-state 0 \
      --random-state 0 --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
    /home/jwinchel/miniconda3/envs/heimdall/bin/python - "$tdir" <<'PY'
from pathlib import Path
import pandas as pd, sys
root = Path(sys.argv[1])
for name in ['tcell_contamination_metrics.csv', 'tcell_cosine_similarity.csv',
             'tcell_lr_top_genes.csv', 'tissue_cell_type_profiles.csv']:
    path = root / name
    if not path.exists(): raise SystemExit(f'Missing {path}')
    try: table = pd.read_csv(path)
    except pd.errors.EmptyDataError: continue
    if 'error' in table and table['error'].fillna('').astype(str).str.len().any():
        raise SystemExit(f'Errors in {path}')
PY
    ;;
  *) echo "Unknown stage: $CELLADMIX_STAGE" >&2; exit 1 ;;
esac
