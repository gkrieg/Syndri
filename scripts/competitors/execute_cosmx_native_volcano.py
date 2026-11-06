"""Execute only the native volcano section using the notebook's displayed cohort."""
import ast,json,os,re
from pathlib import Path
import nbformat
from nbclient import NotebookClient
source=Path('/home/jwinchel/Syndri/figures/Figure4_CosMx_FibroblastCleaning.ipynb')
snapshot=source.read_bytes();n=nbformat.reads(snapshot.decode(),4)
streams='\n'.join(o.text for o in n.cells[4].get('outputs',[]) if o.output_type=='stream')
match=re.search(r'Results: (.+)',streams)
if not match:raise ValueError('Cannot identify the output directory used by existing plots')
run_dir=Path(match.group(1).strip())
plot_defs=ast.parse(n.cells[7].source)
helpers='\n\n'.join(ast.get_source_segment(n.cells[7].source,x) for x in plot_defs.body if isinstance(x,ast.FunctionDef))
setup=f"RUN_DIR = Path({str(run_dir)!r})\nprint('Using existing plot cohort directory:', RUN_DIR)"
t=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('%matplotlib inline\n'+n.cells[1].source),
    nbformat.v4.new_code_cell(n.cells[2].source),nbformat.v4.new_code_cell(setup),
    nbformat.v4.new_code_cell(helpers),nbformat.v4.new_code_cell(n.cells[11].source)])
execution=run_dir/('native_execution_'+os.environ.get('SLURM_JOB_ID',str(os.getpid())))
execution.mkdir(exist_ok=False)
client=NotebookClient(t,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(source.parent)}})
try:client.execute()
except BaseException:
    nbformat.write(t,execution/'failed.ipynb');raise
nbformat.write(t,execution/'executed.ipynb')
if source.read_bytes()==snapshot:
    n.cells[11].outputs=t.cells[-1].outputs;n.cells[11].execution_count=None
    nbformat.write(n,source)
    print('Saved native volcano output into',source,flush=True)
else:print('Notebook edited during run; results saved in',execution,flush=True)
for o in t.cells[-1].outputs:
    if o.output_type=='stream':print(o.text,flush=True)
