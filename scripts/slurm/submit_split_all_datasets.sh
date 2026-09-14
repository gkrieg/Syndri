#!/usr/bin/env bash
set -euo pipefail

base=/work/magroup/skrieger/Syndri/data/datasets
script=/home/jwinchel/Syndri/scripts/slurm/run_split_h5ad_layers.sbatch

mkdir -p /home/jwinchel/Syndri/logs

submit_split() (
    dataset=$1

    export INPUT_DIR="$base/$dataset/$2"
    export INPUT_GLOB="$3"
    export REFERENCE_CELL_TYPE_COL="$4"
    export REFERENCE_H5AD="${5:-}"

    export INPUT_LIST_FILE=""
    export INPUT_FILES=""
    export INPUT_LAYERS=X
    export REFERENCE_LAYER=X

    export OUTPUT_ROOT="$base/$dataset/results/split"
    export INTERMEDIATE_ROOT="$OUTPUT_ROOT/intermediate"
    export SPLIT_OUTPUT_ROOT="$OUTPUT_ROOT/split_output"
    export MERGED_OUTPUT_DIR="$OUTPUT_ROOT/merged_h5ad"
    export MERGE_INPLACE=0
    export RESULT_LAYER_PREFIX=split_purified_counts

    shopt -s nullglob
    files=("$INPUT_DIR"/$INPUT_GLOB)

    if (( ${#files[@]} == 0 )); then
        echo "No matching files for $dataset" >&2
        exit 1
    fi

    echo "Submitting ${#files[@]} files for $dataset"

    sbatch \
        --job-name="split-$dataset" \
        --array="0-$((${#files[@]} - 1))%2" \
        --export=ALL \
        "$script"
)

# submit_split agingbrain_merfish results/spatial \
#     'st_integrated_MsBrainAgingSpatialDonor_[5-8]_[0-2].h5ad' \
#     cell_type

submit_split brain_merfish results/spatial \
    'st_integrated_slice-Zhuang-ABCA-4.00[0-9].h5ad' \
    class

# submit_split nsclc_cosmx interim/spatial \
#     '*annotated.h5ad' \
#     cell_type

# submit_split breast_xenium interim/spatial \
#     '*rctd_annotated.h5ad' \
#     cell_type \
#     "$base/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad"