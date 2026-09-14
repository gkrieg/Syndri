#!/usr/bin/env bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=96Gb
#SBATCH --job-name=tcell-final-metrics
#SBATCH --output=/home/jwinchel/Syndri/logs/tcell-final-metrics-%A_%a.txt
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=8
set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

: "${INPUT_LIST_FILE:?Set INPUT_LIST_FILE}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${LABEL_KEY:?Set LABEL_KEY}"
: "${REFERENCE_DATA:?Set REFERENCE_DATA}"
: "${COORD_X:?Set COORD_X}"
: "${COORD_Y:?Set COORD_Y}"
: "${NEIGHBORHOOD_RADIUS:?Set radius in native coordinate units}"
mapfile -t files < "$INPUT_LIST_FILE"
task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task >= ${#files[@]} )); then
  echo "Array index $task is outside the input manifest." >&2
  exit 1
fi
input=${files[$task]}
sample=$(basename "$input" .h5ad)
output="$OUTPUT_ROOT/$sample"
mkdir -p "$output"
echo "Neighborhood radius: $NEIGHBORHOOD_RADIUS ${COORD_UNITS:-coordinate units}"

# A fresh coordinate key makes the evaluator use the selected obs columns.
# Each input is a single slice, so use its filename as the neighborhood group
# rather than inherited reference/batch labels. Both fields are in-memory only.
python /home/jwinchel/Syndri/eval/metrics_eval.py \
  --input-files "$input" \
  --output-dir "$output" \
  --config-output tcell_metrics_config.json \
  --tcell-output tcell_contamination_metrics.csv \
  --tcell-effect-output tcell_contamination_effects.csv \
  --tcell-lr-gene-output tcell_lr_top_genes.csv \
  --tcell-cosine-output tcell_cosine_similarity.csv \
  --cell-type-profile-output tissue_cell_type_profiles.csv \
  --skip-cell-type-separation \
  --label-key "$LABEL_KEY" \
  --tcell-cell-type-key "$LABEL_KEY" \
  --tcell-sample-key _metrics_tissue \
  --coord-key _tcell_metrics_coordinates \
  --coord-x "$COORD_X" \
  --coord-y "$COORD_Y" \
  --tcell-radius-um "$NEIGHBORHOOD_RADIUS" \
  --exclude-labels "" \
  --filter-tcell-invalid-cells \
  --admix-probs-layer syndri_admix_probs_X \
  --additional-counts-layer split_purified_counts_X \
  --purified-condition-label SPLIT \
  --uncorrected-condition-label raw_X \
  --target-removed-fractions "${TARGET_REMOVED_FRACTIONS:-0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5}" \
  --cleaning-mode probability_scale \
  --cell-type-specific-removal \
  --removal-group-key "$LABEL_KEY" \
  --probability-transform power \
  --probability-power 2 \
  --reference-data "$REFERENCE_DATA" \
  --reference-cell-type-key cell_type \
  --reference-counts-layer "" \
  --tcell-cosine-cell-type-pattern '(?i)(T CD[48]|Treg|CD4-positive|cytotoxic T|T-regulatory)' \
  --tcell-top-n 20 \
  --tcell-min-cells-per-label 100 \
  --tcell-downsample-if-n-cells 50000 \
  --tcell-max-cells-per-label 25000 \
  --tcell-exact-zscore auto \
  --sample-frac "${SAMPLE_FRAC:-1.0}" \
  --sample-random-state 0 \
  --random-state 0 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"

# Neighborhood pairs can legitimately be absent when too few proximal/distal
# cells qualify. Report recorded computation errors, not absent eligible pairs.
python - "$output" "$NEIGHBORHOOD_RADIUS" "${COORD_UNITS:-coordinate units}" "${PHYSICAL_RADIUS_UM:-}" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

output = Path(sys.argv[1])
config_path = output / "tcell_metrics_config.json"
config = json.loads(config_path.read_text())
config.update(neighborhood_radius=float(sys.argv[2]), coordinate_units=sys.argv[3],
              physical_radius_um=float(sys.argv[4]) if sys.argv[4] else None)
config_path.write_text(json.dumps(config, indent=2) + "\n")
failures = []
for name in ["tcell_contamination_metrics.csv", "tcell_cosine_similarity.csv",
             "tcell_lr_top_genes.csv", "tissue_cell_type_profiles.csv"]:
    try:
        table = pd.read_csv(output / name)
    except pd.errors.EmptyDataError:
        print(f"No eligible results: {name}")
        continue
    if "error" in table:
        errors = table["error"].fillna("").astype(str)
        failures.extend(f"{name}: {error}" for error in errors[errors != ""].unique())
    print(f"{name}: {len(table)} rows")
if failures:
    raise SystemExit("\n".join(failures))
PY
