#!/usr/bin/env bash
# Inference and dependent metrics for the regenerated aging-brain MERFISH slices.
set -euo pipefail

repo=/home/jwinchel/Syndri
base=/work/magroup/skrieger/Syndri/data/datasets/agingbrain_merfish
source_dir=$base/results/split/regenerated_rctd_20261010/merged_h5ad
stage_dir=$base/results/model_runs/10x_agingbrain_donors5to8_m87_regenerated_rctd_20261010
metrics_dir=$base/results/metrics/model_runs/10x_agingbrain_donors5to8_m87_regenerated_rctd_20261010/non_tcell
checkpoint=$repo/Heimdall/admixture_prediction_results/10x_agingbrain_donors5to8
reference=/compute/oven-0-13/jwinchel/scRNA-seq/adata/10x_agingbrain_donors5to8.h5ad
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python

[[ -f "$checkpoint/config.txt" && -f "$checkpoint/model-87.pt" ]] || {
  echo "Missing milestone 87 checkpoint or config in $checkpoint" >&2
  exit 1
}
[[ -f "$reference" ]] || { echo "Missing reference $reference" >&2; exit 1; }
[[ ! -e "$stage_dir" && ! -e "$metrics_dir" ]] || {
  echo "Refusing to reuse existing inference or metric results directory" >&2
  exit 1
}

shopt -s nullglob
sources=("$source_dir"/st_integrated_MsBrainAgingSpatialDonor_[5-8]_[0-2].h5ad)
shopt -u nullglob
(( ${#sources[@]} == 12 )) || {
  echo "Expected 12 regenerated SPLIT inputs, found ${#sources[@]}" >&2
  exit 1
}

# Validate before creating outputs, then copy each source to the isolated run.
"$metadata_python" - "$reference" "$stage_dir" "${sources[@]}" <<'PY'
import os
import shutil
import sys
from pathlib import Path
import h5py

reference = Path(sys.argv[1])
stage = Path(sys.argv[2])
sources = [Path(p) for p in sys.argv[3:]]

def genes(handle):
    var = handle["var"]
    index = var[var.attrs["_index"]]
    if isinstance(index, h5py.Group):
        if "values" in index:
            return index["values"].asstr()[()].tolist()
        return index["categories"].asstr()[()][index["codes"][()]].tolist()
    return index.asstr()[()].tolist()

with h5py.File(reference) as handle:
    reference_genes = genes(handle)
    model_cell_types = set(handle["uns"]["cell_type_dict"])
assert reference_genes and len(reference_genes) == len(set(reference_genes))
for source in sources:
    with h5py.File(source) as handle:
        missing = set(reference_genes) - set(genes(handle))
        if missing:
            raise ValueError(f"{source}: {len(missing)} model genes missing")
        if "split_purified_counts_X" not in handle["layers"]:
            raise ValueError(f"{source}: missing SPLIT counts")
        labels = handle["obs"]["cell_type"]
        observed = set(labels["categories"].asstr()[()]) if isinstance(labels, h5py.Group) else set(labels.asstr()[()])
        unknown = observed - model_cell_types
        if unknown:
            raise ValueError(f"{source}: cell types absent from model reference: {sorted(unknown)}")

stage.mkdir(parents=True)
manifest = stage / "inference_inputs.txt"
for source in sources:
    target = stage / source.name
    temporary = stage / f".{source.name}.staging-{os.getpid()}"
    shutil.copy2(source, temporary)
    temporary.replace(target)
manifest.write_text("".join(f"{stage / source.name}\n" for source in sources))
(stage / "reference_genes.txt").write_text("\n".join(reference_genes) + "\n")
print(f"Staged {len(sources)} slices with {len(reference_genes)} model genes in {stage}")
PY

mkdir -p "$metrics_dir" "$repo/logs"
inference_job=$(sbatch --parsable \
  --job-name=syndri-agingbrain-m87 \
  --array=0-11%3 \
  --export="ALL,ST_DATA_LIST_FILE=$stage_dir/inference_inputs.txt,CHECKPOINT_DIR=$checkpoint,MILESTONE=87,REFERENCE_DATA=$reference,GENE_LIST=$stage_dir/reference_genes.txt,BATCH_SIZE=16,CELL_TYPE_COL=cell_type,COORD_X=center_x,COORD_Y=center_y,INPUT_LAYERS=X,PRED_LAYERS=syndri_admix_probs_X,CONFIG_SOURCE=checkpoint,APPLY_ADMIXTURE=0,PRESERVE_SPLIT_COLUMN=0,SPLIT_COL=split,SPLIT_VALUE=val,FAST_INFERENCE=1,DRY_RUN=0,OVERWRITE_LAYERS=0,EXTRA_OVERRIDES=fc.args.query_cell_type_col=cell_type" \
  "$repo/scripts/slurm/admixture_inference_layers.sbatch")
echo "Inference job: $inference_job"

metrics_job=$(sbatch --parsable \
  --job-name=metrics-agingbrain-m87 \
  --array=0-11%2 \
  --dependency="afterok:$inference_job" \
  --export="ALL,INPUT_LIST_FILE=$stage_dir/inference_inputs.txt,OUTPUT_ROOT=$metrics_dir,LABEL_KEY=cell_type,EXCLUDE_LABELS=nan,SYNDRI_REFERENCE_COSINE=0" \
  "$repo/scripts/slurm/metrics_eval_final.sbatch")
echo "Dependent metrics job: $metrics_job"
echo "Inference results: $stage_dir"
echo "Metric results: $metrics_dir"
