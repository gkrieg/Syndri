"""Generate responses, ranking summaries, natural-context variation, and notebook."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pandas as pd
from all_types_panel import summarize
from rank_responsiveness import summarize_ranks

run=Path(sys.argv[1]);design=json.loads((run/'design.json').read_text())
summarize(run,response_scale='logit')
summarize_ranks(run)
subprocess.run([sys.executable,str(run/'context_baselines.py'),str(run)],check=True)
summary=[];distribution=[]
for entry in design['slices']:
 root=run/'context_variability'/entry['slice']
 if not (root/'complete.json').exists():raise FileNotFoundError(root)
 summary.append(pd.read_csv(root/'summary.csv'));distribution.append(pd.read_csv(root/'distribution.csv'))
s=pd.concat(summary,ignore_index=True);d=pd.concat(distribution,ignore_index=True)
s.to_csv(run/'context_variability/slice_context_variability.csv',index=False)
d.to_csv(run/'context_variability/slice_distributions.csv',index=False)
d.groupby(['cell_type','definition','malignant_neighbor_proportion']).fraction.mean().rename('mean_slice_fraction').reset_index().to_csv(run/'context_variability/slice_balanced_distribution.csv',index=False)
pooled=[]
for (cell_type,definition),frame in d.groupby(['cell_type','definition']):
 hist=frame.groupby('malignant_neighbor_proportion').n_cells.sum();n=int(hist.sum());x=hist.index.to_numpy();w=hist.to_numpy()/n if n else np.zeros(len(hist))
 mean=float(w@x);sd=float(np.sqrt(w@((x-mean)**2)))
 sub=s[s.cell_type.eq(cell_type)&s.definition.eq(definition)]
 pooled.append(dict(cell_type=cell_type,definition=definition,n_cells=n,n_slices=int(sub.n_cells.gt(0).sum()),
                    pooled_mean=mean,pooled_sd=sd,pooled_fraction_zero=float(w[x==0].sum()),
                    median_slice_sd=sub.sd.median(),median_slice_iqr=sub.iqr.median(),
                    n_slices_with_context_variation=int(sub.sufficient_family_variation.sum())))
pd.DataFrame(pooled).to_csv(run/'context_variability/pooled_context_variability.csv',index=False)
(run/'context_variability/complete.json').write_text(json.dumps(dict(n_slices=len(design['slices']),definitions=['model_context','physical_context']),indent=2)+'\n')
subprocess.run([sys.executable,str(run/'execute_panel_notebook.py'),str(run/'panel_notebook.ipynb')],check=True)
# Publish the completed notebook only after every plotting cell succeeds.
import shutil
import hashlib
notebook=Path('/home/jwinchel/Syndri/context_analysis/ResponsivenessAllCellTypes.ipynb')
expected=design.get('notebook_source_sha256')
if expected and hashlib.sha256(notebook.read_bytes()).hexdigest()!=expected:
 print('Notebook changed during computation; completed outputs remain in',run/'panel_notebook.ipynb',flush=True)
else:
 shutil.copy2(run/'panel_notebook.ipynb',notebook)
(run/'complete.json').write_text(json.dumps(dict(n_slices=len(design['slices']),notebook='ResponsivenessAllCellTypes.ipynb'),indent=2)+'\n')
print('Completed selected-slice panel:',run,flush=True)
