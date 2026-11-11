"""Pilot first, then four additional slices and automatic aggregation."""
import json,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]);record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
jobs={}
def submit(key,mode,array=None,dependency=None):
 cmd=['sbatch','--parsable','--cpus-per-task=4',f'--job-name=response-{key}',f'--export=ALL,PANEL_RUN={run},PANEL_MODE={mode}']
 if array:cmd.append('--array='+array)
 if dependency:cmd.append('--dependency='+dependency)
 if mode=='infer':cmd+=['--partition=gpu-large,gpu-small','--gres=gpu:1','--mem=32G']
 if mode=='summarize':cmd+=['--mem=16G']
 cmd.append('/home/jwinchel/Syndri/scripts/slurm/responsiveness_all_types.sbatch')
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 jobs[key]=dict(job_id=job,mode=mode,array=array,dependency=dependency)
 record.write_text(json.dumps(jobs,indent=2)+'\n');print(key,job,flush=True);return job
p=submit('pilot-prepare','prepare','0')
i=submit('pilot-infer','infer','0',f'afterok:{p}')
rp=submit('remaining-prepare','prepare','1-4%2',f'afterok:{i}')
ri=submit('remaining-infer','infer','1-4%2',f'aftercorr:{rp}')
submit('summaries','summarize',dependency=f'afterok:{i}:{ri}')
