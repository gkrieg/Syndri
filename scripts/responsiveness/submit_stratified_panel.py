"""Submit stratified preparation, inference, and dependent analysis with 4 CPUs."""
from pathlib import Path
import json,subprocess,sys
run=Path(sys.argv[1]);design=json.loads((run/'design.json').read_text());record=run/'jobs.json'
if record.exists():raise FileExistsError(record)
jobs={}
missing=[i for i,e in enumerate(design['slices']) if not Path(e['contexts']).with_suffix('.complete.json').exists()]
def submit(mode,array=None,dependency=None):
    cmd=['sbatch','--parsable','--cpus-per-task=4','--mem=32G',f'--job-name=response-strat300-{mode}',f'--export=ALL,PANEL_RUN={run},PANEL_MODE={mode}']
    if array:cmd+=['--array='+array]
    if dependency:cmd+=['--dependency='+dependency]
    if mode=='infer':cmd+=['--partition=gpu-large,gpu-small','--gres=gpu:1']
    cmd.append('/home/jwinchel/Syndri/scripts/slurm/responsiveness_stratified.sbatch')
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    jobs[mode]=dict(job_id=job,array=array,dependency=dependency,cpus=4,memory='32G')
    record.write_text(json.dumps(jobs,indent=2)+'\n');print(mode,job,flush=True);return job
prepared=submit('prepare',','.join(map(str,missing))+'%2') if missing else None
inference=submit('infer',f'0-{len(design["slices"])-1}%2',f'afterok:{prepared}' if prepared else None)
submit('finalize',dependency=f'afterok:{inference}')
