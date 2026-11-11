"""Submit prepared immutable manifests, with CosMx training first."""
import json,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1])
record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
jobs={}
def submit(name,manifest,stage,dependency=None):
    rows=json.loads((run/manifest).read_text())
    args=['sbatch','--parsable','--job-name='+name,'--array=0-'+str(len(rows)-1)+'%2',
          '--cpus-per-task=4','--mem=96G']
    if stage=='train':args+=['--partition=gpu-large,gpu-small','--gres=gpu:1']
    else:args+=['--partition=cpu,gpu-small,gpu-large']
    if dependency:args+=['--dependency='+dependency]
    args+=['--export=ALL,RUN_MANIFEST='+str(run/manifest)+',RUN_STAGE='+stage+',RUN_WORKER='+str(run/'scripts/resolvi_saved_worker.py'),str(run/'scripts/resolvi_saved_models.sbatch')]
    job=subprocess.check_output(args,text=True).strip().split(';')[0]
    jobs[name]=dict(job_id=job,n_slices=len(rows),dependency=dependency,stage=stage)
    record.write_text(json.dumps(jobs,indent=2))
    print(name,job,flush=True)
    return job
first=submit('resolvi-refit-cosmx','cosmx.json','train')
submit('resolvi-metrics-cosmx','cosmx.json','metrics','aftercorr:'+first)
rest=submit('resolvi-refit-remaining','remaining.json','train','afterany:'+first)
submit('resolvi-metrics-remaining','remaining.json','metrics','aftercorr:'+rest)
