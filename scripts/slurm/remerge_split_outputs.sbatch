#!/usr/bin/env bash
#SBATCH -p cpu,gpu-small
#SBATCH --mem=32Gb
#SBATCH --job-name=remerge-split
#SBATCH --output=/home/jwinchel/Syndri/logs/remerge-split-%A_%a.txt
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=2
set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${PY_CONDA_ENV:-rctd}"

: "${INPUT_LIST_FILE:?Set INPUT_LIST_FILE to a frozen list of final h5ad files}"
: "${SPLIT_OUTPUT_ROOT:?Set SPLIT_OUTPUT_ROOT to the dataset split_output directory}"

mapfile -t files < "$INPUT_LIST_FILE"
task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task >= ${#files[@]} )); then
  echo "Array index $task is outside ${#files[@]} inputs." >&2
  exit 1
fi

input=${files[$task]}
sample=$(basename "$input" .h5ad)
task_split_root="$SPLIT_OUTPUT_ROOT/$sample"

echo "Remerging existing SPLIT export"
echo "  input/output: $input"
echo "  SPLIT output: $task_split_root"

python /home/jwinchel/Syndri/data/split_h5ad_io.py merge \
  --h5ad "$input" \
  --split-output-root "$task_split_root" \
  --layers "${INPUT_LAYERS:-X}" \
  --result-layer-prefix "${RESULT_LAYER_PREFIX:-split_purified_counts}" \
  --fill-missing "${FILL_MISSING:-original}" \
  --overwrite

# Fail the task if a regression ever produces an integer SPLIT layer again.
python - "$input" "${RESULT_LAYER_PREFIX:-split_purified_counts}_X" <<'PY'
import sys
import anndata as ad
import numpy as np

path, layer = sys.argv[1:]
adata = ad.read_h5ad(path, backed="r")
try:
    dtype = adata.layers[layer].dtype
finally:
    adata.file.close()
if not np.issubdtype(dtype, np.floating):
    raise SystemExit(f"{path}: {layer} has non-floating dtype {dtype}")
print(f"Verified {layer} dtype: {dtype}")
PY
