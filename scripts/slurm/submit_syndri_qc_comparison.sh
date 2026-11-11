#!/usr/bin/env bash
# Submit paired unfiltered and per-case QC metric runs without touching prior results.
set -euo pipefail

repo=/home/jwinchel/Syndri
base=/work/magroup/skrieger/Syndri/data/datasets
batch=$repo/scripts/slurm/metrics_syndri_qc_comparison.sbatch
manifest_dir=$(mktemp -d "$repo/logs/syndri-qc-comparison-20261014.XXXXXX")

for dataset in agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium; do
  case "$dataset" in
    agingbrain_merfish)
      input_dir=$base/$dataset/results/model_runs/10x_agingbrain_donors5to8_m87_regenerated_rctd_20261010
      label=cell_type
      expected=12 ;;
    brain_merfish)
      input_dir=$base/$dataset/results/final
      label=class
      expected=3 ;;
    nsclc_cosmx)
      input_dir=$base/$dataset/results/final
      label=cell_type
      expected=8 ;;
    breast_xenium)
      input_dir=$base/$dataset/results/final
      label=harmony_cell_type
      expected=19 ;;
  esac
  output_base=$base/$dataset/results/metrics/syndri_qc_comparison_20261014
  if [[ -e "$output_base" ]]; then
    echo "Refusing to reuse existing result directory: $output_base" >&2
    exit 1
  fi

  manifest=$manifest_dir/$dataset.txt
  /home/jwinchel/miniconda3/envs/rctd/bin/python - "$input_dir" "$label" "$expected" "$manifest" <<'PY'
import h5py
from pathlib import Path
import sys

directory, label, expected, manifest = sys.argv[1:]
files = sorted(Path(directory).glob('*.h5ad'))
if len(files) != int(expected):
    raise SystemExit(f'{directory}: expected {expected} slices, found {len(files)}')
for path in files:
    with h5py.File(path) as handle:
        missing = {'syndri_admix_probs_X', 'split_purified_counts_X'} - set(handle['layers'])
        if missing or label not in handle['obs']:
            raise SystemExit(f'{path}: missing layers {sorted(missing)} or label {label}')
Path(manifest).write_text(''.join(f'{path}\n' for path in files))
print(f'{directory}: validated {len(files)} slices')
PY

  for setting in unfiltered qc10counts5genes; do
    if [[ "$setting" == unfiltered ]]; then
      min_counts=0
      min_genes=0
    else
      min_counts=10
      min_genes=5
    fi
    output_root=$output_base/$setting/non_tcell
    job=$(sbatch --parsable --job-name="syndri-qc-$dataset-$setting" \
      --array="0-$((expected - 1))%2" --cpus-per-task=4 --mem=32G \
      --export="ALL,INPUT_LIST_FILE=$manifest,OUTPUT_ROOT=$output_root,LABEL_KEY=$label,SEPARATION_MIN_COUNTS=$min_counts,SEPARATION_MIN_GENES=$min_genes" \
      "$batch")
    echo "$dataset $setting: job $job, $expected slices, output $output_root"
  done
done
echo "Input manifests: $manifest_dir"
