import json,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]);record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
d=json.loads((run/'design.json').read_text());indices=','.join(map(str,d['new_prediction_indices']));jobs={}
def submit(mode,array=None,dependency=None):
 cmd=['sbatch','--parsable','--cpus-per-task=4',f'--job-name=response-selected-{mode}',f'--export=ALL,PANEL_RUN={run},PANEL_MODE={mode}']
 if array:cmd.append('--array='+array)
 if dependency:cmd.append('--dependency='+dependency)
 if mode=='infer':cmd+=['--partition=gpu-large,gpu-small','--gres=gpu:1']
 if mode=='finalize':cmd+=['--mem=16G']
 cmd.append('/home/jwinchel/Syndri/scripts/slurm/responsiveness_selected.sbatch')
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 jobs[mode]=dict(job_id=job,array=array,dependency=dependency);record.write_text(json.dumps(jobs,indent=2)+'\n');print(mode,job,flush=True);return job
p=submit('prepare',indices+'%2');i=submit('infer',indices+'%2','aftercorr:'+p)
v=submit('variation',f'0-{len(d["slices"])-1}%2')
submit('finalize',dependency=f'afterok:{i}:{v}')
