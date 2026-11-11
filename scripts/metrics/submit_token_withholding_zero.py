"""Submit an isolated 4-CPU hard-zero replay and dependent curve rendering."""
from pathlib import Path
import json,subprocess,sys
run=Path(sys.argv[1]).resolve();record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
jobs={}
for family in ['non_tcell','tcell']:
    manifest=run/f'{family}.json';count=len(json.loads(manifest.read_text()))
    command=['sbatch','--parsable',f'--job-name=token-zero-{family}',f'--array=0-{count-1}%2',
        '--cpus-per-task=4','--mem=48G',
        f'--export=ALL,ZERO_METRICS_WORKER={run}/zero_syndri_metrics.py,ZERO_METRICS_MANIFEST={manifest}',str(run/'metrics_syndri_zero.sbatch')]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    jobs[family]=dict(job_id=job,tasks=count,cpus=4,memory='48G',max_concurrent=2)
    record.write_text(json.dumps(jobs,indent=2)+'\n');print(family,job,count,flush=True)
dep='afterany:'+':'.join(jobs[k]['job_id'] for k in ['non_tcell','tcell'])
command=['sbatch','--parsable','--job-name=token-zero-plots','--cpus-per-task=4','--mem=12G',f'--dependency={dep}',f'--export=ALL,TOKEN_ZERO_RUN={run}',str(run/'finalize_token_zero.sbatch')]
job=subprocess.check_output(command,text=True).strip().split(';')[0]
jobs['plots']=dict(job_id=job,tasks=1,cpus=4,memory='12G',dependency=dep)
record.write_text(json.dumps(jobs,indent=2)+'\n');print('plots',job,flush=True)
