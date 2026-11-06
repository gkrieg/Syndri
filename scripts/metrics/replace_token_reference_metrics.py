"""Replace canceled T-cell metrics with all-type cosine/variance tasks and rendering."""
from pathlib import Path
import json,subprocess,sys,hashlib
r=Path(sys.argv[1]).resolve();path=r/'jobs.json';old=json.loads(path.read_text())
if 'reference' in old:raise FileExistsError('Reference array already submitted')
archive=r/'jobs_initial_tcell.json'
if not archive.exists():archive.write_text(json.dumps(old,indent=2)+'\n')
d=json.loads((r/'design.json').read_text());notebook=Path(d['notebook']);d['notebook_launch_sha256']=hashlib.sha256(notebook.read_bytes()).hexdigest();(r/'design.json').write_text(json.dumps(d,indent=2)+'\n')
count=len(json.loads((r/'reference.json').read_text()))
job=subprocess.check_output(['sbatch','--parsable','--job-name=token-zero-reference',f'--array=0-{count-1}%4','--cpus-per-task=4','--mem=32G',f'--export=ALL,TOKEN_ZERO_RUN={r}',str(r/'token_reference_metrics.sbatch')],text=True).strip().split(';')[0]
jobs=dict(non_tcell=dict(old['non_tcell'],max_concurrent=4),reference=dict(job_id=job,tasks=count,cpus=4,memory='32G',max_concurrent=4),superseded_tcell=dict(old['tcell'],status='canceled at user request; use reference metrics'),superseded_plots=dict(old['plots'],status='canceled; replaced by new dependency'))
path.write_text(json.dumps(jobs,indent=2)+'\n');print('reference',job,count,flush=True)
dep='afterany:'+jobs['non_tcell']['job_id']+':'+job
final=subprocess.check_output(['sbatch','--parsable','--job-name=token-zero-plots','--cpus-per-task=4','--mem=12G',f'--dependency={dep}',f'--export=ALL,TOKEN_ZERO_RUN={r}',str(r/'finalize_token_zero.sbatch')],text=True).strip().split(';')[0]
jobs['plots']=dict(job_id=final,tasks=1,cpus=4,memory='12G',dependency=dep);path.write_text(json.dumps(jobs,indent=2)+'\n');print('plots',final,flush=True)
