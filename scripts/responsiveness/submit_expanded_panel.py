"""Submit the 150-query experiment without repeating unchanged population analysis."""
import json,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]);record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
jobs={}
def submit(mode,array=None,dependency=None):
 command=['sbatch','--parsable','--cpus-per-task=4',f'--job-name=response-150-{mode}',f'--export=ALL,PANEL_RUN={run},PANEL_MODE={mode}']
 if array:command.append('--array='+array)
 if dependency:command.append('--dependency='+dependency)
 if mode=='infer':command+=['--partition=gpu-large,gpu-small','--gres=gpu:1']
 if mode=='finalize':command+=['--mem=32G']
 command.append('/home/jwinchel/Syndri/scripts/slurm/responsiveness_selected.sbatch')
 job=subprocess.check_output(command,text=True).strip().split(';')[0]
 jobs[mode]=dict(job_id=job,array=array,dependency=dependency,cpus=4,memory='32G')
 record.write_text(json.dumps(jobs,indent=2)+'\n');print(mode,job,flush=True);return job
p=submit('prepare','0-7%2');i=submit('infer','0-7%2','aftercorr:'+p)
submit('finalize',dependency='afterok:'+i)
