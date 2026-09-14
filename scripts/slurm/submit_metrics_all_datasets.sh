#!/usr/bin/env bash
# Run with bash after Syndri inference. No source H5AD files are modified.
set -euo pipefail

# datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)
datasets=(nsclc_cosmx)
export TARGET_REMOVED_FRACTIONS=0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5
export SAMPLE_FRAC=1.0
export EXCLUDE_LABELS="nan"
concurrent_tasks_per_dataset=2
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python
base=/work/magroup/skrieger/Syndri/data/datasets
script=/home/jwinchel/Syndri/scripts/slurm/metrics_eval_final.sbatch
mkdir -p /home/jwinchel/Syndri/logs

submit_dataset() (
  dataset=$1
  export OUTPUT_ROOT="$base/$dataset/results/metrics/non_tcell"
  case "$dataset" in
    agingbrain_merfish) export LABEL_KEY=cell_type ;;
    brain_merfish) export LABEL_KEY=class ;;
    nsclc_cosmx) export LABEL_KEY=cell_type ;;
    breast_xenium) export LABEL_KEY=harmony_cell_type ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
  esac

  # Only schedule files with both completed prediction layers. Rerun this
  # wrapper after remaining inference jobs finish to include those slices.
  ready=$("$metadata_python" - "$base/$dataset/results/final" "$LABEL_KEY" <<'PY'
import sys
from pathlib import Path
import h5py

directory, label = sys.argv[1:]
files = sorted(Path(directory).glob("*.h5ad"))
if not files:
    print(f"No final H5AD files in {directory}", file=sys.stderr)
for path in files:
    with h5py.File(path, "r") as handle:
        missing = [key for key in ("syndri_admix_probs_X", "split_purified_counts_X")
                   if key not in handle["layers"]]
        if missing:
            print(f"Skipping {path.name}: missing {', '.join(missing)}", file=sys.stderr)
            continue
        if label not in handle["obs"]:
            raise ValueError(f"{path} lacks obs[{label!r}]")
    print(path)
PY
)
  if [[ -z "$ready" ]]; then
    echo "$dataset: no slices ready for metrics."
    exit 0
  fi
  mapfile -t files <<< "$ready"
  mkdir -p "$OUTPUT_ROOT"
  INPUT_LIST_FILE=$(mktemp "$OUTPUT_ROOT/inputs.XXXXXX.txt")
  export INPUT_LIST_FILE
  printf '%s\n' "${files[@]}" > "$INPUT_LIST_FILE"
  echo "$dataset: submitting ${#files[@]} slices, 10 cases per slice."
  sbatch --job-name="metrics-$dataset" \
    --array="0-$((${#files[@]} - 1))%${concurrent_tasks_per_dataset}" \
    --export=ALL "$script"
)

for dataset in "${datasets[@]}"; do
  submit_dataset "$dataset"
done
