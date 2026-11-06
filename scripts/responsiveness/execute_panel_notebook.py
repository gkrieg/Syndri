import json,io,base64,contextlib,traceback,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);n=json.loads(p.read_text());ns={'__name__':'__main__'}
for i,c in enumerate(n['cells']):
 if c['cell_type']!='code':continue
 outputs=[];buffer=io.StringIO()
 def display(value):
  data={'text/plain':[repr(value)]}
  if hasattr(value,'_repr_html_'):
   html=value._repr_html_()
   if html:data['text/html']=[html]
  outputs.append({'output_type':'display_data','metadata':{},'data':data})
 def show(*args,**kwargs):
  for num in plt.get_fignums():
   fig=plt.figure(num);b=io.BytesIO();fig.savefig(b,format='png',dpi=160,bbox_inches='tight')
   outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(b.getvalue()).decode()}})
   plt.close(fig)
 ns['display']=display;plt.show=show
 with contextlib.redirect_stdout(buffer):exec(compile(''.join(c['source']),f'{p}:cell{i}','exec'),ns)
 if buffer.getvalue():outputs.insert(0,{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(keepends=True)})
 c['outputs']=outputs;c['execution_count']=sum(x['cell_type']=='code' for x in n['cells'][:i+1])
 p.write_text(json.dumps(n,indent=1)+'\n')
 print('Executed cell',i,flush=True)
print('Executed notebook:',p,flush=True)
