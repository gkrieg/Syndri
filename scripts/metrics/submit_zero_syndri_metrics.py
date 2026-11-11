"""Submit prepared hard-zero metric manifests with four CPUs per task."""
import json
from pathlib import Path
import subprocess
import sys
run=Path(sys.argv[1]).resolve()
record=run/'jobs.json'
if record.exists():
    raise FileExistsError(record)
jobs={}
for family in ['non_tcell','tcell']:
    manifest=run/f'{family}.json'
    count=len(json.loads(manifest.read_text()))
    command=['sbatch','--parsable',f'--job-name=syndri-zero-{family}',
             f'--array=0-{count-1}%2','--cpus-per-task=4','--mem=48G',
             f'--export=ALL,ZERO_METRICS_WORKER={run}/zero_syndri_metrics.py,ZERO_METRICS_MANIFEST={manifest}',
             str(run/'metrics_syndri_zero.sbatch')]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    jobs[family]={'job_id':job,'tasks':count,'cpus':4,'memory':'48G'}
    record.write_text(json.dumps(jobs,indent=2)+'\n')
    print(family,job,count,flush=True)
