"""Submit prepared shared-cohort metric manifests with four CPUs per task."""
import json,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1])
record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
jobs={}
for group,dependency in [('cosmx',None),('remaining',None)]:
    for family in ['non_tcell','tcell']:
        manifest=run/f'{family}_{group}.json'
        count=len(json.loads(manifest.read_text()))
        if not count:continue
        args=['sbatch','--parsable',f'--job-name=shared-qc-{group}-{family}',
              f'--array=0-{count-1}%2','--cpus-per-task=4',
              '--export=ALL,SHARED_QC_SCRIPT='+str(run/'scripts/shared_syndri_qc.py')+
              ',SHARED_QC_MANIFEST='+str(manifest),str(run/'scripts/metrics_shared_syndri_qc.sbatch')]
        job=subprocess.check_output(args,text=True).strip().split(';')[0]
        jobs[group+'_'+family]=dict(job_id=job,n_tasks=count,cpus=4,dependency=dependency)
        record.write_text(json.dumps(jobs,indent=2))
        print(group,family,job,count,flush=True)
