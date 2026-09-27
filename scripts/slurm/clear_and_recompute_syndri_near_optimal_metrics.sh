#!/usr/bin/env bash
# Remove prior syndri_reference_cosine results and submit the current
# cell_type_smallest_near_optimal_k case for every configured metric family.
set -euo pipefail

repo=/home/jwinchel/Syndri
base=/work/magroup/skrieger/Syndri/data/datasets
python_bin=/home/jwinchel/miniconda3/envs/heimdall/bin/python
datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)

"$python_bin" - "$base" <<'PY'
import shutil
import sys
from pathlib import Path

import pandas as pd

base = Path(sys.argv[1])
label = "syndri_reference_cosine"
changed_files = 0
removed_rows = 0
removed_directories = 0

for dataset in ["agingbrain_merfish", "brain_merfish", "nsclc_cosmx", "breast_xenium"]:
    metrics = base / dataset / "results" / "metrics"
    for family in ["tcell", "non_tcell"]:
        root = metrics / family
        if not root.exists():
            continue
        for path in root.glob("*/*.csv"):
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, UnicodeDecodeError):
                continue
            remove = pd.Series(False, index=frame.index)
            for column in [c for c in ["case", "condition", "setting"] if c in frame]:
                remove |= frame[column].astype(str).eq(label)
            if remove.any():
                removed_rows += int(remove.sum())
                frame.loc[~remove].to_csv(path, index=False)
                changed_files += 1
        for name in ["syndri_thresholds", "syndri_cosine_curves"]:
            for directory in root.glob(f"*/{name}"):
                shutil.rmtree(directory)
                removed_directories += 1

print(
    f"Cleared {removed_rows:,} prior {label} rows from {changed_files:,} CSVs "
    f"and removed {removed_directories:,} diagnostic directories."
)
PY

export SPLIT_ONLY=0
export SYNDRI_REFERENCE_COSINE=1
export SYNDRI_COSINE_ONLY=1
export SYNDRI_ONE_SE_MULTIPLIER=${SYNDRI_ONE_SE_MULTIPLIER:-1}

echo "Submitting non-T-cell adaptive metrics for all four datasets."
bash "$repo/scripts/slurm/submit_metrics_all_datasets.sh" "${datasets[@]}"

echo "Submitting T-cell adaptive metrics for Breast Xenium and NSCLC CosMx."
bash "$repo/scripts/slurm/submit_tcell_metrics_datasets.sh" breast_xenium nsclc_cosmx

echo "Submitting p_admix metrics, including the adaptive setting, for all four datasets."
export ADMIXTURE_BACKEND=python
export OVERWRITE=1
bash "$repo/scripts/slurm/submit_score_admixture_all_datasets.sh" "${datasets[@]}"
