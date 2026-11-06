#!/usr/bin/env bash
set -euo pipefail

repo=/home/jwinchel/Syndri
source_dir=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/spatial
results_root=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/breast_xenium_10neighbor_token_sweep
reference=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python
inference_python=/home/jwinchel/miniconda3/envs/heimdall/bin/python
concurrency=${CONCURRENT_TASKS:-2}
checkpoint_root="$repo/Heimdall/admixture_prediction_results/breast_xenium_10neighbor_token_sweep"
inference_worker="$repo/scripts/slurm/admixture_inference_layers.sbatch"
models=(
  breast_xenium_k10_visible1
  breast_xenium_k10_visible2
  breast_xenium_k10_visible4
  breast_xenium_k10_visible6
  breast_xenium_k10_visible8
  breast_xenium_k10_visible10
)

[[ -x "$metadata_python" ]] || { echo "Missing metadata Python: $metadata_python" >&2; exit 1; }
[[ -x "$inference_python" ]] || { echo "Missing inference Python: $inference_python" >&2; exit 1; }
[[ -f "$reference" ]] || { echo "Missing reference: $reference" >&2; exit 1; }
[[ -f "$inference_worker" ]] || { echo "Missing inference worker: $inference_worker" >&2; exit 1; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "CONCURRENT_TASKS must be a positive integer" >&2; exit 1; }

shopt -s nullglob
inputs=("$source_dir"/GSM*rctd_annotated.h5ad)
shopt -u nullglob
(( ${#inputs[@]} == 19 )) || {
  echo "Expected 19 spatial inputs in $source_dir; found ${#inputs[@]}" >&2
  exit 1
}

# Validate every requested checkpoint before staging any large input files.
for model in "${models[@]}"; do
  checkpoint="$checkpoint_root/$model"
  [[ -f "$checkpoint/config.txt" ]] || { echo "Missing $checkpoint/config.txt" >&2; exit 1; }
  [[ -f "$checkpoint/model-26.pt" ]] || { echo "Missing $checkpoint/model-26.pt" >&2; exit 1; }
  configured_tokens=$(awk '/num_neighbor_cell_type_tokens:/ {print $2; exit}' "$checkpoint/config.txt")
  expected_tokens=${model##*visible}
  [[ "$configured_tokens" == "$expected_tokens" ]] || {
    echo "$model config reports $configured_tokens visible neighbor tokens; expected $expected_tokens" >&2
    exit 1
  }
done

mkdir -p "$results_root" "$repo/logs"

for model in "${models[@]}"; do
  checkpoint="$checkpoint_root/$model"
  output_dir="$results_root/$model"
  mkdir -p "$output_dir"
  manifest=$(mktemp "$output_dir/.inference_inputs.XXXXXX.txt")
  gene_list="$output_dir/reference_genes.txt"

  "$metadata_python" - "$reference" "$gene_list" "$output_dir" "${inputs[@]}" <<'PY'
import os
import shutil
import sys
from pathlib import Path
import h5py

reference, gene_file, output_dir = map(Path, sys.argv[1:4])
sources = [Path(p) for p in sys.argv[4:]]

def read_genes(handle):
    var = handle["var"]
    index = var[var.attrs["_index"]]
    if isinstance(index, h5py.Group) and "values" in index:
        if index["mask"][()].any():
            raise ValueError("Gene index contains missing values")
        return index["values"].asstr()[()].tolist()
    if isinstance(index, h5py.Group):
        codes = index["codes"][()]
        if (codes < 0).any():
            raise ValueError("Gene index contains missing values")
        return index["categories"].asstr()[()][codes].tolist()
    return index.asstr()[()].tolist()

with h5py.File(reference, "r") as handle:
    genes = read_genes(handle)
if not genes or len(genes) != len(set(genes)):
    raise ValueError("Reference gene names must be nonempty and unique")
if any(not gene or "\n" in gene or "\r" in gene for gene in genes):
    raise ValueError("Reference gene names must be nonempty single-line strings")

for source in sources:
    target = output_dir / source.name
    if target.exists() and target.stat().st_size != source.stat().st_size:
        raise ValueError(f"Existing staged file differs in size from source: {target}")
    if not target.exists():
        temporary = target.with_name(f".{target.name}.staging-{os.getpid()}")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    with h5py.File(target, "r") as handle:
        target_genes = read_genes(handle)
        if len(target_genes) != len(set(target_genes)):
            raise ValueError(f"Duplicate gene names in {target}")
        missing = set(genes) - set(target_genes)
        if missing:
            raise ValueError(f"{target} lacks {len(missing)} reference genes: {sorted(missing)[:20]}")

gene_file.write_text("\n".join(genes) + "\n", encoding="utf-8")
print(f"Validated {len(sources)} inputs against {len(genes)} reference genes in {output_dir}")
PY

  for input in "${inputs[@]}"; do
    printf '%s\n' "$output_dir/$(basename "$input")" >> "$manifest"
  done

  sbatch \
    --job-name="syn-vis${model##*visible}" \
    --cpus-per-task=4 \
    --array="0-18%${concurrency}" \
    --export="ALL,ST_DATA_LIST_FILE=$manifest,CHECKPOINT_DIR=$checkpoint,MILESTONE=26,REFERENCE_DATA=$reference,GENE_LIST=$gene_list,BATCH_SIZE=16,CELL_TYPE_COL=harmony_cell_type,COORD_X=x_centroid,COORD_Y=y_centroid,INPUT_LAYERS=X,PRED_LAYERS=syndri_admix_probs_X,CONFIG_SOURCE=checkpoint,APPLY_ADMIXTURE=0,PRESERVE_SPLIT_COLUMN=0,SPLIT_COL=split,SPLIT_VALUE=val,FAST_INFERENCE=1,DRY_RUN=0,OVERWRITE_LAYERS=0,EXTRA_OVERRIDES=fc.args.query_cell_type_col=harmony_cell_type,PYTHON=$inference_python" \
    "$inference_worker"
done
