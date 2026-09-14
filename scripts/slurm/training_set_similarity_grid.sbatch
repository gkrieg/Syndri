#!/bin/bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=32Gb
#SBATCH --job-name syndri-training-grid
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-training-grid-%A_%a.txt
#SBATCH -t 1-23:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/jwinchel/Syndri
mkdir -p /home/jwinchel/Syndri/logs

source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

SCRIPT_PATH=${SCRIPT_PATH:-/home/jwinchel/Syndri/scripts/slurm/training_set_similarity_grid.sbatch}
RUN_MODE=${RUN_MODE:-controller}

SPATIAL_PATH=${SPATIAL_PATH:-}
SINGLE_CELL_PATH=${SINGLE_CELL_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-/compute/oven-0-13/jwinchel/spatial_adata/training_set_similarity_grid}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-training_set_similarity}
MANIFEST=${MANIFEST:-$OUTPUT_DIR/grid_manifest.tsv}
MAX_ARRAY_JOBS=${MAX_ARRAY_JOBS:-20}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Comma, semicolon, or whitespace separated value lists.
K_NEIGHBORS_VALUES=${K_NEIGHBORS_VALUES:-15}
MIN_SAMPLED_TRANSCRIPT_THRESHOLD_VALUES=${MIN_SAMPLED_TRANSCRIPT_THRESHOLD_VALUES:-0}
MIN_SOURCE_TRANSCRIPT_COUNT_VALUES=${MIN_SOURCE_TRANSCRIPT_COUNT_VALUES:-5}
USE_SAME_CELL_TYPE_NEIGHBORS_VALUES=${USE_SAME_CELL_TYPE_NEIGHBORS_VALUES:-false}
ONLY_SAMPLE_UNEXPRESSED_GENES_VALUES=${ONLY_SAMPLE_UNEXPRESSED_GENES_VALUES:-true}
MASK_OUT_RANDOM_GENES_VALUES=${MASK_OUT_RANDOM_GENES_VALUES:-false}
MASK_OUT_FRAC_VALUES=${MASK_OUT_FRAC_VALUES:-0.0}
TAU_RANGE_VALUES=${TAU_RANGE_VALUES:-0.4,0.7}
ADMIXTURE_SCALE_VALUES=${ADMIXTURE_SCALE_VALUES:-1.0}

# Other evaluator settings copied into every manifest row.
SPATIAL_LAYER=${SPATIAL_LAYER:-}
SINGLE_CELL_RAW_LAYER=${SINGLE_CELL_RAW_LAYER:-}
SPATIAL_CELL_TYPE_KEY=${SPATIAL_CELL_TYPE_KEY:-harmony_cell_type}
SINGLE_CELL_CELL_TYPE_KEY=${SINGLE_CELL_CELL_TYPE_KEY:-cell_type}
COORD_X=${COORD_X:-x_centroid}
COORD_Y=${COORD_Y:-y_centroid}
TISSUE_KEY=${TISSUE_KEY:-orig.ident}
NEIGHBOR_CELL_TYPE_KEY=${NEIGHBOR_CELL_TYPE_KEY:-}
NEIGHBOR_DICT_KEY=${NEIGHBOR_DICT_KEY:-neighbor_dict}
NEIGHBOR_ID_COL=${NEIGHBOR_ID_COL:-}
BUILD_NEIGHBOR_DICT=${BUILD_NEIGHBOR_DICT:-true}
GROUP_MODE=${GROUP_MODE:-both}
MIN_CELLS_PER_GROUP=${MIN_CELLS_PER_GROUP:-20}
MAX_CELLS_PER_DOMAIN=${MAX_CELLS_PER_DOMAIN:-0}
CLASSIFIER=${CLASSIFIER:-true}
CLASSIFIER_BY_GROUP=${CLASSIFIER_BY_GROUP:-false}
CLASSIFIER_MAX_CELLS_PER_DOMAIN=${CLASSIFIER_MAX_CELLS_PER_DOMAIN:-5000}
CLASSIFIER_MAX_FEATURES=${CLASSIFIER_MAX_FEATURES:-2000}
TOP_GENE_COUNTS=${TOP_GENE_COUNTS:-50,100,500}
RANDOM_STATE=${RANDOM_STATE:-0}

export \
  SPATIAL_PATH \
  SINGLE_CELL_PATH \
  OUTPUT_DIR \
  OUTPUT_PREFIX \
  MANIFEST \
  K_NEIGHBORS_VALUES \
  MIN_SAMPLED_TRANSCRIPT_THRESHOLD_VALUES \
  MIN_SOURCE_TRANSCRIPT_COUNT_VALUES \
  USE_SAME_CELL_TYPE_NEIGHBORS_VALUES \
  ONLY_SAMPLE_UNEXPRESSED_GENES_VALUES \
  MASK_OUT_RANDOM_GENES_VALUES \
  MASK_OUT_FRAC_VALUES \
  TAU_RANGE_VALUES \
  ADMIXTURE_SCALE_VALUES \
  SPATIAL_LAYER \
  SINGLE_CELL_RAW_LAYER \
  SPATIAL_CELL_TYPE_KEY \
  SINGLE_CELL_CELL_TYPE_KEY \
  COORD_X \
  COORD_Y \
  TISSUE_KEY \
  NEIGHBOR_CELL_TYPE_KEY \
  NEIGHBOR_DICT_KEY \
  NEIGHBOR_ID_COL \
  BUILD_NEIGHBOR_DICT \
  GROUP_MODE \
  MIN_CELLS_PER_GROUP \
  MAX_CELLS_PER_DOMAIN \
  CLASSIFIER \
  CLASSIFIER_BY_GROUP \
  CLASSIFIER_MAX_CELLS_PER_DOMAIN \
  CLASSIFIER_MAX_FEATURES \
  TOP_GENE_COUNTS \
  RANDOM_STATE

make_manifest() {
  if [[ -z "$SPATIAL_PATH" || -z "$SINGLE_CELL_PATH" ]]; then
    echo "SPATIAL_PATH and SINGLE_CELL_PATH are required." >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
  python - <<'PY'
from itertools import product
from pathlib import Path
import csv
import os
import re


def values(name):
    raw = os.environ[name].strip()
    if not raw:
        return [""]
    if name == "TAU_RANGE_VALUES":
        parts = re.split(r"[;|]", raw)
    else:
        parts = re.split(r"[,;\s]+", raw)
    return [part.strip() for part in parts if part.strip()]


output_dir = Path(os.environ["OUTPUT_DIR"])
manifest = Path(os.environ["MANIFEST"])
manifest.parent.mkdir(parents=True, exist_ok=True)

grid_keys = [
    "k_neighbors",
    "min_sampled_transcript_threshold",
    "min_source_transcript_count",
    "use_same_cell_type_neighbors",
    "only_sample_unexpressed_genes",
    "mask_out_random_genes",
    "mask_out_frac",
    "tau_range",
    "admixture_scale",
]
grid_values = {
    "k_neighbors": values("K_NEIGHBORS_VALUES"),
    "min_sampled_transcript_threshold": values("MIN_SAMPLED_TRANSCRIPT_THRESHOLD_VALUES"),
    "min_source_transcript_count": values("MIN_SOURCE_TRANSCRIPT_COUNT_VALUES"),
    "use_same_cell_type_neighbors": values("USE_SAME_CELL_TYPE_NEIGHBORS_VALUES"),
    "only_sample_unexpressed_genes": values("ONLY_SAMPLE_UNEXPRESSED_GENES_VALUES"),
    "mask_out_random_genes": values("MASK_OUT_RANDOM_GENES_VALUES"),
    "mask_out_frac": values("MASK_OUT_FRAC_VALUES"),
    "tau_range": values("TAU_RANGE_VALUES"),
    "admixture_scale": values("ADMIXTURE_SCALE_VALUES"),
}
fixed = {
    "spatial_path": os.environ["SPATIAL_PATH"],
    "single_cell_path": os.environ["SINGLE_CELL_PATH"],
    "sample_admixture_from_neighbors": "true",
    "spatial_layer": os.environ.get("SPATIAL_LAYER", ""),
    "single_cell_raw_layer": os.environ.get("SINGLE_CELL_RAW_LAYER", ""),
    "spatial_cell_type_key": os.environ["SPATIAL_CELL_TYPE_KEY"],
    "single_cell_cell_type_key": os.environ["SINGLE_CELL_CELL_TYPE_KEY"],
    "coord_x": os.environ["COORD_X"],
    "coord_y": os.environ["COORD_Y"],
    "tissue_key": os.environ["TISSUE_KEY"],
    "neighbor_cell_type_key": os.environ.get("NEIGHBOR_CELL_TYPE_KEY", ""),
    "neighbor_dict_key": os.environ["NEIGHBOR_DICT_KEY"],
    "neighbor_id_col": os.environ.get("NEIGHBOR_ID_COL", ""),
    "build_neighbor_dict": os.environ["BUILD_NEIGHBOR_DICT"],
    "group_mode": os.environ["GROUP_MODE"],
    "min_cells_per_group": os.environ["MIN_CELLS_PER_GROUP"],
    "max_cells_per_domain": os.environ["MAX_CELLS_PER_DOMAIN"],
    "classifier": os.environ["CLASSIFIER"],
    "classifier_by_group": os.environ["CLASSIFIER_BY_GROUP"],
    "classifier_max_cells_per_domain": os.environ["CLASSIFIER_MAX_CELLS_PER_DOMAIN"],
    "classifier_max_features": os.environ["CLASSIFIER_MAX_FEATURES"],
    "top_gene_counts": os.environ["TOP_GENE_COUNTS"],
    "random_state": os.environ["RANDOM_STATE"],
}

fieldnames = ["config_id", *fixed.keys(), *grid_keys]
rows = []
for idx, combo in enumerate(product(*(grid_values[key] for key in grid_keys))):
    row = {"config_id": f"cfg_{idx:05d}", **fixed, **dict(zip(grid_keys, combo))}
    row["admixture_scales"] = row.pop("admixture_scale")
    rows.append(row)

fieldnames = ["config_id", *fixed.keys(), "admixture_scales", *[key for key in grid_keys if key != "admixture_scale"]]
with manifest.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

(output_dir / "grid_size.txt").write_text(f"{len(rows)}\n")
print(f"Wrote {len(rows)} grid rows to {manifest}")
PY
}

submit_grid() {
  local n_rows
  n_rows=$(cat "$OUTPUT_DIR/grid_size.txt")
  if (( n_rows < 1 )); then
    echo "Generated grid has no rows." >&2
    exit 1
  fi

  local array_spec="0-$((n_rows - 1))%$MAX_ARRAY_JOBS"
  echo "Submitting worker array: $array_spec"
  worker_job_id=$(
    sbatch --parsable \
      --array="$array_spec" \
      --export=ALL,RUN_MODE=worker,SCRIPT_PATH="$SCRIPT_PATH",MANIFEST="$MANIFEST",OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_PREFIX="$OUTPUT_PREFIX" \
      "$SCRIPT_PATH"
  )
  echo "Submitted worker job: $worker_job_id"

  collate_job_id=$(
    sbatch --parsable \
      --dependency=afterok:"$worker_job_id" \
      --export=ALL,RUN_MODE=collate,SCRIPT_PATH="$SCRIPT_PATH",MANIFEST="$MANIFEST",OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_PREFIX="$OUTPUT_PREFIX" \
      "$SCRIPT_PATH"
  )
  echo "Submitted collate job: $collate_job_id"
}

run_worker() {
  local task_id=${SLURM_ARRAY_TASK_ID:-0}
  echo "Running grid task $task_id"
  echo "  manifest: $MANIFEST"
  echo "  output: $OUTPUT_DIR/task_${task_id}"

  # EXTRA_ARGS is intentionally word-split so simple additional flags can be supplied via sbatch --export.
  # shellcheck disable=SC2086
  python /home/jwinchel/Syndri/eval/training_set_similarity_metrics.py \
    --manifest "$MANIFEST" \
    --manifest-index "$task_id" \
    --manifest-delimiter tab \
    --output-dir "$OUTPUT_DIR/task_${task_id}" \
    --output-prefix "$OUTPUT_PREFIX" \
    $EXTRA_ARGS
}

run_collate() {
  echo "Collating results under $OUTPUT_DIR"
  python - <<'PY'
from pathlib import Path
import os
import pandas as pd

base = Path(os.environ["OUTPUT_DIR"])
prefix = os.environ["OUTPUT_PREFIX"]
manifest = Path(os.environ["MANIFEST"])

outputs = {
    "summary": f"{prefix}_summary.csv",
    "cell_metrics": f"{prefix}_cell_metrics.csv",
    "gene_correlations": f"{prefix}_gene_correlations.csv",
    "rank_agreement": f"{prefix}_rank_agreement.csv",
    "classifier": f"{prefix}_classifier.csv",
    "per_gene": f"{prefix}_per_gene.csv",
}

manifest_df = pd.read_csv(manifest, sep="\t")
for label, filename in outputs.items():
    frames = []
    for task_dir in sorted(base.glob("task_*"), key=lambda p: int(p.name.split("_")[-1])):
        path = task_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        task_id = int(task_dir.name.split("_")[-1])
        df = pd.read_csv(path)
        df.insert(0, "task_id", task_id)
        df.insert(1, "task_dir", str(task_dir))
        frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out_path = base / f"all_{filename}"
        out.to_csv(out_path, index=False)
        print(f"Wrote {len(out)} rows: {out_path}")
    else:
        print(f"No rows found for {label}")

manifest_copy = base / "submitted_grid_manifest.tsv"
manifest_df.to_csv(manifest_copy, sep="\t", index=False)
print(f"Copied manifest: {manifest_copy}")
PY
}

case "$RUN_MODE" in
  controller)
    make_manifest
    submit_grid
    ;;
  worker)
    run_worker
    ;;
  collate)
    run_collate
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE. Use controller, worker, or collate." >&2
    exit 1
    ;;
esac

# SPATIAL_PATH=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/GSM9509174_breast_breast_B1_B1_3_rctd_annotated_dropout_1pct.h5ad \
# SINGLE_CELL_PATH=/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad \
# OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/training_set_parameter_optimization/ \
# K_NEIGHBORS_VALUES="5,10,15,30" \
# MIN_SAMPLED_TRANSCRIPT_THRESHOLD_VALUES="10" \
# MIN_SOURCE_TRANSCRIPT_COUNT_VALUES="10" \
# USE_SAME_CELL_TYPE_NEIGHBORS_VALUES="false,true" \
# ONLY_SAMPLE_UNEXPRESSED_GENES_VALUES="true,false" \
# MASK_OUT_RANDOM_GENES_VALUES="false" \
# MASK_OUT_FRAC_VALUES="0.0" \
# TAU_RANGE_VALUES="0.0,0.2;0.2,0.4;0.4,0.7;0.7,0.85;0.85,1.0" \
# MAX_ARRAY_JOBS=5 \
# sbatch /home/jwinchel/Syndri/scripts/slurm/training_set_similarity_grid.sbatch