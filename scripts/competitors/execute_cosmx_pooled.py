"""Execute pooled panels using existing slice cohorts, without refitting any model."""
from pathlib import Path
import ast,json,os,re
import nbformat
from nbclient import NotebookClient
p=Path('/home/jwinchel/Syndri/figures/Figure4_CosMx_FibroblastCleaning.ipynb')
snapshot=p.read_bytes();n=nbformat.reads(snapshot.decode(),4)
streams='\n'.join(o.text for o in n.cells[4].get('outputs',[]) if o.output_type=='stream')
r=Path(re.search(r'Results: (.+)',streams).group(1).strip())
setup=f"RUN_DIR=Path({str(r)!r})\nsettings=json.loads((RUN_DIR/'settings.json').read_text())\npaths=sorted((ROOT/'final').glob('*.h5ad'))\nif SLICE_SUBSET is not None:paths=[p for p in paths if p.stem in SLICE_SUBSET]\nassert settings['methods']==METHODS, 'Existing displayed cohorts use different methods; run the per-slice setup first'"
helpers=n.cells[9].source.split('\nfor path in paths:')[0]
t=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('%matplotlib inline\n'+n.cells[1].source),
    nbformat.v4.new_code_cell(n.cells[2].source),nbformat.v4.new_code_cell(setup),
    nbformat.v4.new_code_cell(n.cells[7].source),nbformat.v4.new_code_cell(helpers),
    nbformat.v4.new_code_cell(n.cells[-1].source)])
out=r/('pooled_execution_'+os.environ.get('SLURM_JOB_ID',str(os.getpid())));out.mkdir(exist_ok=False)
try:NotebookClient(t,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(p.parent)}}).execute()
except BaseException:
    nbformat.write(t,out/'failed.ipynb');raise
nbformat.write(t,out/'executed.ipynb')
if p.read_bytes()==snapshot:
    n.cells[-1].outputs=t.cells[-1].outputs;n.cells[-1].execution_count=None;nbformat.write(n,p)
    print('Saved pooled plots in',p,flush=True)
else:print('Notebook changed during run; outputs saved in',out,flush=True)
for o in t.cells[-1].outputs:
    if o.output_type=='stream':print(o.text,flush=True)
