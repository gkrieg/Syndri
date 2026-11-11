"""Export type-specific cellAdmix factor removal as a count layer."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from common import output_path, save_new, validate_counts

def run(final: Path, assignments: Path, factor_map: Path, root: Path | None,
        method_name: str = 'celladmix'):
    target = output_path(final, method_name, root)
    if target.exists():
        raise FileExistsError(target)
    a = ad.read_h5ad(final)
    validate_counts(a)
    factor = pd.read_csv(factor_map)
    if list(factor.columns) != ['cell_type', 'factor']:
        raise ValueError('Factor map must have cell_type,factor columns')
    if factor.duplicated().any():
        raise ValueError('Duplicate factor mapping')
    table = pd.read_csv(assignments, dtype={'mol_id': str, 'cell_id': str, 'gene': str,
                                            'cell_type': str})
    if table.mol_id.duplicated().any():
        raise ValueError('Duplicate molecule assignments')
    if factor.empty:
        removed = table.iloc[:0]
    else:
        flagged = table.merge(factor.assign(remove=True), on=['cell_type', 'factor'], how='left')
        removed = flagged.loc[flagged['remove'] == True]
    cell = pd.Categorical(removed.cell_id, categories=a.obs_names.astype(str))
    gene = pd.Categorical(removed.gene, categories=a.var_names.astype(str))
    matched = (cell.codes >= 0) & (gene.codes >= 0)
    if not matched.all():
        raise ValueError('Removed transcript IDs do not all match final AnnData')
    removed_counts = sparse.coo_matrix((np.ones(matched.sum(), dtype=np.int32),
                                        (cell.codes[matched], gene.codes[matched])), shape=a.shape).tocsr()
    raw = a.X.tocsr() if sparse.issparse(a.X) else sparse.csr_matrix(a.X)
    if np.any((removed_counts - raw).data > 0):
        raise ValueError('Removed molecule counts exceed final AnnData counts')
    corrected = raw - removed_counts
    a.layers['celladmix_corrected_counts'] = corrected
    commit = subprocess.check_output(['git', '-C', '/home/jwinchel/cellAdmix', 'rev-parse', 'HEAD'],
                                     text=True).strip()
    a.uns['celladmix_competitor'] = {'factor_map': str(factor_map), 'assignments': str(assignments),
                                    'source_final': str(final), 'celladmix_commit': commit}
    save_new(a, target)
    print(target)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('final_h5ad', type=Path)
    p.add_argument('assignments_csv', type=Path)
    p.add_argument('factor_map_csv', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--method-name', default='celladmix')
    args = p.parse_args()
    run(args.final_h5ad, args.assignments_csv, args.factor_map_csv,
        args.output_root, args.method_name)
