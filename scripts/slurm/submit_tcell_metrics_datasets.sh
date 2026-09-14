#!/usr/bin/env bash
# Run with bash; one Slurm array per dataset and one task per ready slice.
set -euo pipefail

datasets=(breast_xenium nsclc_cosmx)
export TARGET_REMOVED_FRACTIONS=0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5
export SAMPLE_FRAC=1.0
concurrent_tasks_per_dataset=2
base=/work/magroup/skrieger/Syndri/data/datasets
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python
script=/home/jwinchel/Syndri/scripts/slurm/metrics_eval_tcell_final.sbatch

declare -A references radii
references[breast_xenium]="$base/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad"
references[nsclc_cosmx]="$base/nsclc_cosmx/interim/single-cell/sc_integrated.h5ad"
export PHYSICAL_RADIUS_UM=15
# Optional calibration. When blank, reuse breast's numeric radius in pixels.
nsclc_micrometers_per_pixel=""
radii[breast_xenium]="$PHYSICAL_RADIUS_UM"
radii[nsclc_cosmx]=50
if [[ " ${datasets[*]} " == *" nsclc_cosmx "* && -n "$nsclc_micrometers_per_pixel" ]]; then
  radii[nsclc_cosmx]=$("$metadata_python" - "$PHYSICAL_RADIUS_UM" "$nsclc_micrometers_per_pixel" <<'PY'
import math
import sys

try:
    scale = float(sys.argv[2])
except ValueError:
    raise SystemExit("Set nsclc_micrometers_per_pixel at the top of this script before submission.")
if not math.isfinite(scale) or scale <= 0:
    raise SystemExit("nsclc_micrometers_per_pixel must be positive and finite.")
print(float(sys.argv[1]) / scale)
PY
)
fi

mkdir -p /home/jwinchel/Syndri/logs
submit_dataset() (
  dataset=$1
  export REFERENCE_DATA="${references[$dataset]}"
  export NEIGHBORHOOD_RADIUS="${radii[$dataset]}"
  export OUTPUT_ROOT="$base/$dataset/results/metrics/tcell"
  case "$dataset" in
    breast_xenium)
      export LABEL_KEY=harmony_cell_type COORD_X=x_centroid COORD_Y=y_centroid COORD_UNITS=micrometers ;;
    nsclc_cosmx)
      export LABEL_KEY=cell_type COORD_X=CenterX_global_px COORD_Y=CenterY_global_px COORD_UNITS=pixels
      if [[ -z "$nsclc_micrometers_per_pixel" ]]; then
        export PHYSICAL_RADIUS_UM=""
      fi
      ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
  esac
  [[ -f "$REFERENCE_DATA" ]] || { echo "Missing reference: $REFERENCE_DATA" >&2; exit 1; }

  ready=$("$metadata_python" - "$base/$dataset/results/final" "$LABEL_KEY" "$COORD_X" "$COORD_Y" <<'PY'
import sys
from pathlib import Path
import h5py

directory, label, x, y = sys.argv[1:]
files = sorted(Path(directory).glob("*.h5ad"))
if not files:
    print(f"No final H5AD files in {directory}", file=sys.stderr)
for path in files:
    with h5py.File(path, "r") as handle:
        missing = [key for key in ["syndri_admix_probs_X", "split_purified_counts_X"]
                   if key not in handle["layers"]]
        if missing:
            print(f"Skipping {path.name}: missing {', '.join(missing)}", file=sys.stderr)
            continue
        for key in [label, x, y]:
            if key not in handle["obs"]:
                raise ValueError(f"{path} lacks obs[{key!r}]")
    print(path)
PY
)
  if [[ -z "$ready" ]]; then
    echo "$dataset: no slices ready for T-cell metrics."
    exit 0
  fi
  mapfile -t files <<< "$ready"
  mkdir -p "$OUTPUT_ROOT"
  INPUT_LIST_FILE=$(mktemp "$OUTPUT_ROOT/inputs.XXXXXX.txt")
  export INPUT_LIST_FILE
  printf '%s\n' "${files[@]}" > "$INPUT_LIST_FILE"
  echo "$dataset: submitting ${#files[@]} slices; radius $NEIGHBORHOOD_RADIUS $COORD_UNITS."
  sbatch --job-name="tcell-$dataset" \
    --array="0-$((${#files[@]} - 1))%${concurrent_tasks_per_dataset}" \
    --export=ALL "$script"
)

for dataset in "${datasets[@]}"; do
  submit_dataset "$dataset"
done
