from pathlib import Path
import json,subprocess,sys
run=Path(sys.argv[1]);record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
n=len(json.loads((run/'manifest.json').read_text()));jobs={}
command=['sbatch','--parsable','--cpus-per-task=4',f'--array=0-{n-1}%2',f'--export=ALL,COSINE_RUN={run}',
         '/home/jwinchel/Syndri/scripts/slurm/all_cell_type_reference_cosine.sbatch']
job=subprocess.check_output(command,text=True).strip().split(';')[0]
jobs['metrics']=dict(job_id=job,n_tasks=n,cpus=4,memory='32G');record.write_text(json.dumps(jobs,indent=2)+'\n');print('metrics',job,n,flush=True)
final=subprocess.check_output(['sbatch','--parsable','--cpus-per-task=4','--dependency=afterok:'+job,
 f'--export=ALL,COSINE_RUN={run}','/home/jwinchel/Syndri/scripts/slurm/finalize_all_type_reference_cosine.sbatch'],text=True).strip().split(';')[0]
jobs['finalize']=dict(job_id=final,dependency='afterok:'+job);record.write_text(json.dumps(jobs,indent=2)+'\n');print('finalize',final,flush=True)
