#!/usr/bin/env bash
set -euo pipefail

# Edit the settings below, then run: bash submit_syndri_final.sh
# Each checkpoint directory must contain config.txt and model-<milestone>.pt.
# Each reference should contain the selected model's training cell_type_dict.
datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)

declare -A checkpoint_dirs milestones reference_files

checkpoint_dirs[agingbrain_merfish]="/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/10x_agingbrain_donors5to8/"
milestones[agingbrain_merfish]="80"
reference_files[agingbrain_merfish]="/compute/oven-0-13/jwinchel/scRNA-seq/adata/10x_agingbrain_donors5to8.h5ad"

checkpoint_dirs[brain_merfish]="/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/10x_wholemousebrain_Zhuang-ABCA-4.003/"
milestones[brain_merfish]="130"
reference_files[brain_merfish]="/compute/oven-0-13/jwinchel/scRNA-seq/adata/10x_wholemousebrain_Zhuang-ABCA-4.003.h5ad"

checkpoint_dirs[nsclc_cosmx]="/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/10x_nsclc-cosmx_Lung9-Rep1/"
milestones[nsclc_cosmx]="108"
reference_files[nsclc_cosmx]="/compute/oven-0-13/jwinchel/scRNA-seq/adata/10x_nsclc-cosmx_Lung9-Rep1.h5ad"

checkpoint_dirs[breast_xenium]="/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_matchedcelltypes_optimized_unexpressedgenes_withquerycelltype/"
milestones[breast_xenium]="50"
reference_files[breast_xenium]="/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad"

concurrent_tasks_per_dataset=3
# Keep completed inference results by default; set to 1 to recompute them.
rerun_completed=0
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python
declare -A batch_sizes=(
  [agingbrain_merfish]=16 [brain_merfish]=16
  [nsclc_cosmx]=16 [breast_xenium]=16
)

base=/work/magroup/skrieger/Syndri/data/datasets

# Subshells keep each dataset's exported settings separate.
run_dataset() (
dataset=$1
phase=$2
export CHECKPOINT_DIR="${checkpoint_dirs[$dataset]}"
export MILESTONE="${milestones[$dataset]}"
export REFERENCE_DATA="${reference_files[$dataset]}"
export BATCH_SIZE="${batch_sizes[$dataset]}"

if [[ -z "$CHECKPOINT_DIR" || -z "$MILESTONE" || -z "$REFERENCE_DATA" ]]; then
  echo "Fill in checkpoint_dirs, milestones, and reference_files for $dataset at the top of this script." >&2
  exit 1
fi
if [[ ! "$MILESTONE" =~ ^[0-9]+$ ]]; then
  echo "Milestone for $dataset must be a nonnegative integer: $MILESTONE" >&2
  exit 1
fi

source_dir="$base/$dataset/results/split/merged_h5ad"
final_dir="$base/$dataset/results/final"

case "$dataset" in
  agingbrain_merfish)
    pattern='st_integrated_MsBrainAgingSpatialDonor_[5-8]_[0-2].h5ad'
    expected=12
    export CELL_TYPE_COL=cell_type
    export COORD_COLS=center_x,center_y
    ;;
  brain_merfish)
    pattern='st_integrated_slice-Zhuang-ABCA-4.00[0-9].h5ad'
    expected=3
    export CELL_TYPE_COL=class
    export COORD_COLS=x,y
    ;;
  nsclc_cosmx)
    pattern='*annotated.h5ad'
    expected=8
    export CELL_TYPE_COL=cell_type
    export COORD_COLS=CenterX_global_px,CenterY_global_px
    ;;
  breast_xenium)
    pattern='*rctd_annotated.h5ad'
    expected=19
    export CELL_TYPE_COL=harmony_cell_type
    export COORD_COLS=x_centroid,y_centroid
    ;;
  *)
    echo "Unknown dataset: $dataset" >&2
    exit 1
    ;;
esac

[[ -f "$CHECKPOINT_DIR/config.txt" ]] || {
  echo "Missing $CHECKPOINT_DIR/config.txt" >&2
  exit 1
}
[[ -f "$CHECKPOINT_DIR/model-${MILESTONE}.pt" ]] || {
  echo "Missing checkpoint model-${MILESTONE}.pt" >&2
  exit 1
}
[[ -f "$REFERENCE_DATA" ]] || {
  echo "Missing reference: $REFERENCE_DATA" >&2
  exit 1
}

shopt -s nullglob
files=("$source_dir"/$pattern)

if (( ${#files[@]} != expected )); then
  echo "Expected $expected SPLIT outputs; found ${#files[@]} in $source_dir" >&2
  exit 1
fi

# Validate reference genes against the actual inference inputs. Reuse existing
# final files after failures, and skip completed predictions unless requested.
pending=$("$metadata_python" - "$REFERENCE_DATA" "$final_dir" "$rerun_completed" "${files[@]}" <<'PY'
import sys
from pathlib import Path
import h5py

def genes(handle):
    var = handle["var"]
    index = var[var.attrs["_index"]]
    if isinstance(index, h5py.Group) and "values" in index:
        if index["mask"][()].any():
            raise ValueError("Gene index contains missing values.")
        return index["values"].asstr()[()].tolist()
    if isinstance(index, h5py.Group):
        codes = index["codes"][()]
        if (codes < 0).any():
            raise ValueError("Gene index contains missing values.")
        return index["categories"].asstr()[()][codes].tolist()
    return index.asstr()[()].tolist()

with h5py.File(sys.argv[1], "r") as handle:
    reference_genes = genes(handle)
if not reference_genes or len(set(reference_genes)) != len(reference_genes):
    raise ValueError("Reference gene names must be nonempty and unique.")
if any(not gene or "\n" in gene or "\r" in gene for gene in reference_genes):
    raise ValueError("Reference gene names must be nonempty single-line strings.")

final_dir = Path(sys.argv[2])
for source in map(Path, sys.argv[4:]):
    target = final_dir / source.name
    actual = target if target.exists() else source
    with h5py.File(actual, "r") as handle:
        if "syndri_admix_probs_X" in handle["layers"] and sys.argv[3] != "1":
            continue
        if "split_purified_counts_X" not in handle["layers"]:
            raise ValueError(f"Missing SPLIT layer in {actual}")
        input_genes = genes(handle)
        if len(set(input_genes)) != len(input_genes):
            raise ValueError(f"Duplicate gene names in {actual}")
        missing = set(reference_genes) - set(input_genes)
        if missing:
            raise ValueError(f"{actual} lacks {len(missing)} reference genes: {sorted(missing)[:20]}")
    print(source)
PY
)

if [[ -z "$pending" ]]; then
  echo "$dataset: all predictions already exist; skipping."
  exit 0
fi
mapfile -t files <<< "$pending"

if [[ "$phase" == "validate" ]]; then
  echo "Validated $dataset: ${#files[@]} files, milestone $MILESTONE"
  exit 0
fi

mkdir -p "$final_dir" /home/jwinchel/Syndri/logs

# Unique manifests keep later submissions from changing queued jobs' inputs.
ST_DATA_LIST_FILE=$(mktemp "$final_dir/syndri_inputs.XXXXXX.txt")
export ST_DATA_LIST_FILE
export GENE_LIST="${ST_DATA_LIST_FILE%.txt}.genes.txt"

"$metadata_python" - "$REFERENCE_DATA" "$GENE_LIST" <<'PY'
import sys
from pathlib import Path
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    var = handle["var"]
    index = var[var.attrs["_index"]]
    if isinstance(index, h5py.Group) and "values" in index:
        if index["mask"][()].any():
            raise ValueError("Gene index contains missing values.")
        genes = index["values"].asstr()[()].tolist()
    elif isinstance(index, h5py.Group):
        codes = index["codes"][()]
        if (codes < 0).any():
            raise ValueError("Gene index contains missing values.")
        genes = index["categories"].asstr()[()][codes].tolist()
    else:
        genes = index.asstr()[()].tolist()
Path(sys.argv[2]).write_text("\n".join(genes) + "\n", encoding="utf-8")
print(f"Using {len(genes)} reference genes in reference order.")
PY

for file in "${files[@]}"; do
  target="$final_dir/$(basename "$file")"
  if [[ ! -e "$target" ]]; then
    cp -- "$file" "$target"
  fi
  printf '%s\n' "$target" >> "$ST_DATA_LIST_FILE"
done

# Predict admixture probabilities from the original spatial counts.
export INPUT_LAYERS=X
export PRED_LAYERS=syndri_admix_probs_X
export CONFIG_SOURCE=checkpoint
export APPLY_ADMIXTURE=0
export PRESERVE_SPLIT_COLUMN=0
export SPLIT_COL=split
export SPLIT_VALUE=val
export TISSUE_COL=""
export FAST_INFERENCE=1
export DRY_RUN=0
export OVERWRITE_LAYERS="$rerun_completed"
# The query-cell-type tokenizer has a separate setting from neighbor labels.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:+$EXTRA_OVERRIDES }fc.args.query_cell_type_col=$CELL_TYPE_COL"

sbatch \
  --job-name="syndri-$dataset" \
  --array="0-$((${#files[@]} - 1))%${concurrent_tasks_per_dataset}" \
  --export=ALL \
  /home/jwinchel/Syndri/scripts/slurm/admixture_inference_layers.sbatch
)

if [[ ! "$concurrent_tasks_per_dataset" =~ ^[1-9][0-9]*$ ]]; then
  echo "concurrent_tasks_per_dataset must be a positive integer." >&2
  exit 1
fi

# Validate all datasets before copying files or submitting any jobs.
for dataset in "${datasets[@]}"; do
  run_dataset "$dataset" validate
done

for dataset in "${datasets[@]}"; do
  run_dataset "$dataset" submit
done
