#!/usr/bin/env bash
#SBATCH --job-name=figure4-cancer-gsea
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/figure4-cancer-gsea-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/figure4-cancer-gsea-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PATH=/home/jwinchel/miniconda3/envs/rctd/bin:$PATH
cd /home/jwinchel/Syndri/figures
/home/jwinchel/miniconda3/envs/rctd/bin/python -u - <<'PY'
from pathlib import Path
import hashlib, os
import nbformat
from nbclient import NotebookClient
source = Path('/home/jwinchel/Syndri/figures/Figure4_Cancer.ipynb')
snapshot = source.read_bytes()
output = source.parent / 'outputs/cancer_tcell_enrichment' / ('execution_' + os.environ['SLURM_JOB_ID'])
output.mkdir(parents=True, exist_ok=False)
(output/'source.ipynb').write_bytes(snapshot)
notebook = nbformat.reads(snapshot.decode(), as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(source.parent)}})
try:
    client.execute()
except BaseException:
    nbformat.write(notebook, output/'failed.ipynb')
    raise
nbformat.write(notebook, output/'Figure4_Cancer.executed.ipynb')
if source.read_bytes() == snapshot:
    temporary = output/'with_outputs.ipynb'
    nbformat.write(notebook, temporary)
    temporary.replace(source)
    print('Saved executed cells and figures to', source, flush=True)
else:
    print('Source notebook changed during execution; executed copy saved in', output, flush=True)
PY
