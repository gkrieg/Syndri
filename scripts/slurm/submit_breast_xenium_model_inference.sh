#!/usr/bin/env bash
set -euo pipefail

# Stage each Xenium slice into its own model-specific result directory, then
# submit one inference array (one slice per task) for each requested checkpoint.

repo=/home/jwinchel/Syndri
source_dir=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/spatial
results_root=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results
reference=/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python
inference_python=/home/jwinchel/miniconda3/envs/heimdall/bin/python
concurrency=${CONCURRENT_TASKS:-2}

declare -A checkpoints milestones
checkpoints[breast_xenium_k10_separators]="$repo/Heimdall/admixture_prediction_results/breast_xenium_k10_separators"
milestones[breast_xenium_k10_separators]=40
checkpoints[chromium_breast_xenium_full_retrain]="$repo/Heimdall/admixture_prediction_results/chromium_breast_xenium_full_retrain"
milestones[chromium_breast_xenium_full_retrain]=28

[[ -x "$metadata_python" ]] || { echo "Missing metadata Python: $metadata_python" >&2; exit 1; }
[[ -x "$inference_python" ]] || { echo "Missing inference Python: $inference_python" >&2; exit 1; }
[[ -f "$reference" ]] || { echo "Missing reference: $reference" >&2; exit 1; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "CONCURRENT_TASKS must be a positive integer" >&2; exit 1; }

shopt -s nullglob
inputs=("$source_dir"/GSM*rctd_annotated.h5ad)
shopt -u nullglob
(( ${#inputs[@]} == 19 )) || {
  echo "Expected 19 spatial inputs in $source_dir; found ${#inputs[@]}" >&2
  exit 1
}

mkdir -p "$results_root" "$repo/logs"

# Read H5AD gene names without loading the expression matrix. Validate that all
# spatial inputs contain every gene used by the training reference.
for model in "${!checkpoints[@]}"; do
  checkpoint=${checkpoints[$model]}
  milestone=${milestones[$model]}
  [[ -f "$checkpoint/config.txt" ]] || { echo "Missing $checkpoint/config.txt" >&2; exit 1; }
  [[ -f "$checkpoint/model-${milestone}.pt" ]] || { echo "Missing $checkpoint/model-${milestone}.pt" >&2; exit 1; }
done

for model in breast_xenium_k10_separators chromium_breast_xenium_full_retrain; do
  outdir="$results_root/$model"
  mkdir -p "$outdir"
  manifest=$(mktemp "$outdir/.inference_inputs.XXXXXX.txt")
  genes_file="$outdir/reference_genes.txt"

  "$metadata_python" - "$reference" "$genes_file" "$outdir" "${inputs[@]}" <<'PY'
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
if not genes or len(set(genes)) != len(genes):
    raise ValueError("Reference gene names must be nonempty and unique")
if any(not g or "\n" in g or "\r" in g for g in genes):
    raise ValueError("Reference gene names must be nonempty, single-line strings")

for source in sources:
    target = output_dir / source.name
    if target.exists() and target.stat().st_size != source.stat().st_size:
        raise ValueError(f"Existing staged file differs in size from source: {target}")
    if not target.exists():
        # Stage to a temporary name and rename only after the copy completes.
        import shutil
        temporary = target.with_name(f".{target.name}.staging-{__import__('os').getpid()}")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    with h5py.File(target, "r") as handle:
        target_genes = read_genes(handle)
        if len(set(target_genes)) != len(target_genes):
            raise ValueError(f"Duplicate gene names in {target}")
        missing = set(genes) - set(target_genes)
        if missing:
            raise ValueError(f"{target} lacks {len(missing)} reference genes: {sorted(missing)[:20]}")

gene_file.write_text("\n".join(genes) + "\n", encoding="utf-8")
print(f"Validated {len(sources)} inputs against {len(genes)} reference genes in {output_dir}")
PY

  for input in "${inputs[@]}"; do
    printf '%s\n' "$outdir/$(basename "$input")" >> "$manifest"
  done

  checkpoint=${checkpoints[$model]}
  milestone=${milestones[$model]}
  extra_overrides="fc.args.query_cell_type_col=harmony_cell_type"

  sbatch \
    --job-name="syndri-${model}" \
    --array="0-18%${concurrency}" \
    --export="ALL,ST_DATA_LIST_FILE=$manifest,CHECKPOINT_DIR=$checkpoint,MILESTONE=$milestone,REFERENCE_DATA=$reference,GENE_LIST=$genes_file,BATCH_SIZE=16,CELL_TYPE_COL=harmony_cell_type,COORD_X=x_centroid,COORD_Y=y_centroid,INPUT_LAYERS=X,PRED_LAYERS=syndri_admix_probs_X,CONFIG_SOURCE=checkpoint,APPLY_ADMIXTURE=0,PRESERVE_SPLIT_COLUMN=0,SPLIT_COL=split,SPLIT_VALUE=val,FAST_INFERENCE=1,DRY_RUN=0,OVERWRITE_LAYERS=0,EXTRA_OVERRIDES=$extra_overrides,PYTHON=$inference_python" \
    "$repo/scripts/slurm/admixture_inference_layers.sbatch"
done
