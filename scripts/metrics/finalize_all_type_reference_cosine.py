"""Audit completed all-type cosine runs and render the updated notebook."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import numpy as np,pandas as pd
run=Path(sys.argv[1]);entries=json.loads((run/'manifest.json').read_text());frames=[];coverage=[];checks=[]
for entry in entries:
 root=Path(entry['output'])
 if not (root/'complete.json').is_file():raise FileNotFoundError(root)
 table=pd.read_csv(root/'reference_cosine_similarity.csv');frames.append(table)
 methods=pd.read_csv(root/'method_coverage.csv');methods['dataset']=entry['dataset'];methods['slice']=Path(entry['input']).stem;coverage.append(methods)
 if entry['match_cancer_tcell_eligibility']:
  previous=Path(entry['source_config']).parent/'tcell_cosine_similarity.csv'
  if previous.is_file():
   old=pd.read_csv(previous)
   old=old.loc[old.condition.isin(['raw_X','SPLIT']) & old['error'].fillna('').eq('')]
   new=table.loc[table.method.isin(['Raw','SPLIT']) & table.status.eq('ok')]
   paired=old.merge(new,on=['condition','spatial_cell_type','reference_cell_type'],suffixes=('_previous','_all_types'),validate='one_to_one')
   delta=np.abs(paired.cosine_similarity_previous-paired.cosine_similarity_all_types)
   if len(paired)!=len(old):raise ValueError(f'Missing prior T-cell comparison rows: {previous}')
   if len(delta) and not np.allclose(paired.cosine_similarity_previous,paired.cosine_similarity_all_types,rtol=1e-7,atol=1e-8):
    raise ValueError(f'Existing T-cell metric changed: {previous}, max delta {delta.max()}')
   checks.append(dict(dataset=entry['dataset'],slice=Path(entry['input']).stem,n_tcell_rows=len(paired),max_absolute_difference=float(delta.max()) if len(delta) else None))
pd.concat(frames,ignore_index=True).to_csv(run/'all_reference_cosine.csv.gz',index=False,compression='gzip')
pd.concat(coverage,ignore_index=True).to_csv(run/'method_coverage.csv',index=False)
pd.DataFrame(checks).to_csv(run/'prior_tcell_reproduction.csv',index=False)
subprocess.run([sys.executable,str(run/'execute_metrics_notebook.py'),str(run/'metrics_notebook.ipynb')],check=True)
metadata=json.loads((run/'notebook_metadata.json').read_text());path=Path(metadata['path'])
if hashlib.sha256(path.read_bytes()).hexdigest()==metadata['sha256']:
 shutil.copy2(run/'metrics_notebook.ipynb',path)
else:print('Notebook changed while jobs ran; executed copy remains in',run/'metrics_notebook.ipynb',flush=True)
(run/'complete.json').write_text(json.dumps(dict(n_slices=len(entries),n_previous_tcell_checks=len(checks)),indent=2)+'\n')
print('All-cell-type reference cosine complete:',run,flush=True)
