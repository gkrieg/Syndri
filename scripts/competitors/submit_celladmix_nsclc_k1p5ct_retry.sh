#!/usr/bin/env bash
# Retry incomplete NSCLC annotations, then export corrected matrices and metrics.
set -euo pipefail

repo=/home/jwinchel/Syndri
source_manifest=$repo/logs/celladmix-factorization-k1p5ct-20261005-123021.tsv
manifest=$repo/logs/celladmix-nsclc-k1p5ct-20261013.tsv
retry_manifest=$repo/logs/celladmix-nsclc-k1p5ct-annotation-retry-20261013.tsv
batch=$repo/scripts/competitors/celladmix_k1p5ct_followup.sbatch

[[ -s "$source_manifest" && -f "$batch" ]] || {
  echo "Missing factorization manifest or follow-up batch script" >&2
  exit 1
}
[[ ! -e "$manifest" && ! -e "$retry_manifest" ]] || {
  echo "Refusing to overwrite existing NSCLC manifests" >&2
  exit 1
}

/home/jwinchel/miniconda3/envs/rctd/bin/python - "$source_manifest" "$manifest" "$retry_manifest" <<'PY'
from pathlib import Path
import sys
import pandas as pd

source, manifest, retry = map(Path, sys.argv[1:])
rows = [line for line in source.read_text().splitlines()
        if "/nsclc_cosmx/" in line.split("\t", 1)[0]]
if len(rows) != 8:
    raise ValueError(f"Expected eight NSCLC factorization rows, found {len(rows)}")

pending = []
for line in rows:
    final, ntypes, factors = line.split("\t")
    if (ntypes, factors) != ("15", "23"):
        raise ValueError(f"Unexpected factorization parameters: {line}")
    name = Path(final).stem
    root = Path(final).parents[1] / "competitors" / "celladmix"
    fit = root / "factorization_k1p5ct" / name
    if not all((fit / file).stat().st_size > 0 for file in
               ("assignments.csv", "factor_loadings.csv", "factor_summary.csv")):
        raise ValueError(f"Incomplete factorization: {fit}")
    annotation = root / "annotation_k1p5ct" / name
    required = [annotation / file for file in
                ("factor_map.csv", "bridge_raw.rds", "bridge_res.rds")]
    if all(path.is_file() and path.stat().st_size > 0 for path in required):
        if pd.read_csv(required[0]).empty:
            raise ValueError(f"No factor mappings in {required[0]}")
    else:
        pending.append(line)

manifest.write_text("\n".join(rows) + "\n")
retry.write_text("\n".join(pending) + ("\n" if pending else ""))
print(f"Validated {len(rows)} factorizations; {len(pending)} annotations need retry")
for line in pending:
    print("Retry:", Path(line.split("\t", 1)[0]).stem)
PY

annotation_dependency=()
retry_count=$(wc -l < "$retry_manifest")
if (( retry_count > 0 )); then
  annotation_job=$(sbatch --parsable --job-name=celladmix-nsclc-annotation-retry \
    --array="0-$((retry_count - 1))%1" --cpus-per-task=2 --mem=128G \
    --export="ALL,CELLADMIX_FOLLOWUP_MANIFEST=$retry_manifest,CELLADMIX_STAGE=annotation" \
    "$batch")
  annotation_dependency=(--dependency="afterok:$annotation_job")
  echo "Annotation retry job: $annotation_job (2 workers, 128 GB)"
fi

correction_job=$(sbatch --parsable --job-name=celladmix-nsclc-correction \
  --array=0-7%1 --cpus-per-task=4 --mem=96G \
  "${annotation_dependency[@]}" \
  --export="ALL,CELLADMIX_FOLLOWUP_MANIFEST=$manifest,CELLADMIX_STAGE=correction" \
  "$batch")
echo "Correction job: $correction_job"

metrics_job=$(sbatch --parsable --job-name=celladmix-nsclc-metrics \
  --array=0-7%1 --cpus-per-task=4 --mem=96G \
  --dependency="afterok:$correction_job" \
  --export="ALL,CELLADMIX_FOLLOWUP_MANIFEST=$manifest,CELLADMIX_STAGE=metrics" \
  "$batch")
echo "Metrics job: $metrics_job"
