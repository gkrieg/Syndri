"""Replay a saved metrics config, changing only input/output locations and runtime threads."""
import json,sys,os
from pathlib import Path
import pandas as pd
sys.path.insert(0,'/home/jwinchel/Syndri/eval')
import metrics_eval
config_path,input_path,output_path=sys.argv[1:]
cfg=json.loads(Path(config_path).read_text())
out=Path(output_path)
out.mkdir(parents=True,exist_ok=False)
parser=metrics_eval.build_parser()
known={a.dest for a in parser._actions}
# Extra provenance fields are not evaluator arguments.
settings={k:v for k,v in cfg.items() if k in known}
settings.update(input_files=[input_path],input_list_file=None,output_dir=str(out),
                append_output=False,n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK',8)))
parser.set_defaults(**settings)
metrics_eval.build_parser=lambda:parser
sys.argv=[sys.argv[0]]
metrics_eval.main()
for path in out.glob('*.csv'):
    try: table=pd.read_csv(path)
    except pd.errors.EmptyDataError: continue
    if 'error' in table and table.error.fillna('').astype(str).ne('').any():
        raise RuntimeError(f'Recorded metric errors in {path}')
if not settings.get('skip_cell_type_separation'):
    table=pd.read_csv(out/settings['cell_type_output'])
    cols=['silhouette_score','calinski_harabasz','davies_bouldin','leiden_nmi_vs_label',
          'leiden_adjusted_rand_vs_label','median_genes_per_cell','median_transcripts_per_cell']
    if len(table)!=1 or table[cols].isna().any().any():
        raise RuntimeError('Incomplete separation metrics')
print('Completed metrics:',out,flush=True)
