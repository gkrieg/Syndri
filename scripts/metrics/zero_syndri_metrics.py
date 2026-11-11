"""Replay prior metric settings with hard zeroing in an isolated output directory."""
import json
import os
from pathlib import Path
import sys

manifest, task = sys.argv[1:]
entry = json.loads(Path(manifest).read_text())[int(task)]
config = json.loads(Path(entry['config']).read_text())
output = Path(entry['output'])
output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, '/home/jwinchel/Syndri/eval')
import metrics_eval
parser = metrics_eval.build_parser()
known = {action.dest for action in parser._actions}
settings = {key: value for key, value in config.items() if key in known}
settings.update(cleaning_mode='zero', output_dir=str(output), append_output=False,
                n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
parser.set_defaults(**settings)
metrics_eval.build_parser = lambda: parser
sys.argv = [sys.argv[0]]
metrics_eval.main()
import pandas as pd
for path in output.glob('*.csv'):
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        continue
    if 'error' in table and table['error'].fillna('').astype(str).ne('').any():
        raise RuntimeError(f'Recorded metric errors: {path}')
if not settings.get('skip_cell_type_separation', False):
    table = pd.read_csv(output / settings['cell_type_output'])
    cleaned = table[table['case'].str.startswith('cleaned_admixed_removed_frac_', na=False)]
    expected = len(settings['target_removed_fractions'].split(','))
    columns = ['silhouette_score', 'calinski_harabasz', 'davies_bouldin',
               'leiden_nmi_vs_label', 'leiden_adjusted_rand_vs_label']
    if len(cleaned) != expected or cleaned[columns].isna().any().any():
        raise RuntimeError('Incomplete Syndri hard-zero metrics')
    if not cleaned['cleaning_mode'].eq('zero').all():
        raise RuntimeError('Unexpected cleaning mode')
(output / 'complete.json').write_text(json.dumps({'source_config':entry['source_config'],
    'cleaning_mode':'zero','n_jobs':settings['n_jobs']}, indent=2) + '\n')
print('Completed:', output, flush=True)
