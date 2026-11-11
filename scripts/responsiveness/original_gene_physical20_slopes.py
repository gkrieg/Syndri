"""Across-query original-logit slopes against actual physical malignant k20 count."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from physical20_context_counts import ensure_physical20_context_counts
from gene_context_slopes import T_CELL_TYPES

def compute_original_gene_physical20_slopes(run, output_dir, marker_genes, pool_t_cells=True, min_queries=10):
    import anndata as ad
    from scipy import sparse
    run, output_dir = Path(run), Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    config = dict(marker_genes=list(marker_genes), pool_t_cells=pool_t_cells,
        min_queries_per_section_gene=min_queries, min_distinct_neighbor_counts=3,
        response='original-context predicted admixture logit', logit_clip=1e-6,
        predictor='query actual malignant count among 20 nearest non-self cells',
        slope_unit='logit / additional malignant neighbor', fit='OLS across queries separately by section/type/gene',
        gene_filter='raw original query X > 0')
    complete = output_dir / 'complete.json'
    if complete.exists():
        if json.loads(complete.read_text()) != config:
            raise ValueError('Cached original-logit slope definition differs')
        return pd.read_csv(output_dir / 'slice_gene_slopes.csv')
    count_paths = ensure_physical20_context_counts(run)
    frames, audits = [], []
    expected_names = None
    for entry in json.loads((run / 'design.json').read_text())['slices']:
        a = ad.read_h5ad(entry['predictions'])
        obs = a.obs.reset_index(drop=True)
        original = obs.loc[obs.context_kind.eq('original')].copy()
        if original.focal_cell_id.duplicated().any():
            raise ValueError('Expected one original context per query')
        names = a.var.get('feature_name', pd.Series(a.var_names, index=a.var_names)).fillna('').astype(str).to_numpy()
        names = np.where(names == '', a.var_names.astype(str), names)
        if len(set(names)) != len(names):
            raise ValueError('Gene names must be unique')
        if expected_names is None:
            expected_names = names.copy()
        elif not np.array_equal(names, expected_names):
            raise ValueError('Gene panels differ between sections')
        p = a.layers['admix_probs'][original.index.to_numpy()]
        p = (p.toarray() if sparse.issparse(p) else np.asarray(p)).astype(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid probabilities')
        clipped = np.clip(p, 1e-6, 1-1e-6)
        y = np.log(clipped)-np.log1p(-clipped)
        raw = a.X[original.index.to_numpy()]
        expressed = (raw.toarray() if sparse.issparse(raw) else np.asarray(raw)) > 0
        counts = pd.read_csv(count_paths[entry['slice']]).set_index('cell_id')
        x = original.focal_cell_id.astype(str).map(counts.malignant_neighbors_k20).to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Unmatched query physical counts')
        original = original.reset_index(drop=True)
        original['query_group'] = original.query_cell_type.replace(
            {t:'T cells (pooled)' for t in T_CELL_TYPES}) if pool_t_cells else original.query_cell_type
        np.savez_compressed(output_dir / (entry['slice']+'_original_query_logits.npz'),
            genes=names.astype(str), query_cell_id=original.focal_cell_id.to_numpy(dtype=str),
            query_cell_type=original.query_cell_type.to_numpy(dtype=str),
            query_group=original.query_group.to_numpy(dtype=str), malignant_neighbors_k20=x.astype(np.int16),
            original_logit=y.astype(np.float32), expressed=expressed)
        for group, ix in original.groupby('query_group',observed=True).groups.items():
            ix = np.asarray(ix)
            xx, yy, valid = x[ix], y[ix], expressed[ix]
            n = valid.sum(axis=0)
            sx = (valid * xx[:,None]).sum(axis=0)
            sy = np.where(valid, yy, 0).sum(axis=0)
            sxx = (valid * xx[:,None]**2).sum(axis=0)
            sxy = np.where(valid, yy * xx[:,None], 0).sum(axis=0)
            distinct = np.stack([(valid & (xx[:,None] == value)).any(axis=0) for value in range(21)]).sum(axis=0)
            denominator = n*sxx - sx*sx
            supported = (n >= min_queries) & (distinct >= 3) & (denominator > 0)
            beta = np.divide(n*sxy-sx*sy, denominator, out=np.full(len(names),np.nan), where=supported)
            status = np.where(n < min_queries, 'insufficient_queries',
                     np.where(distinct < 3, 'insufficient_neighbor_count_variation',
                     np.where(denominator <= 0, 'constant_predictor', 'ok')))
            frames.append(pd.DataFrame(dict(slice=entry['slice'],query_group=str(group),gene=names,
                median_slope=beta,n_queries=n,n_distinct_neighbor_counts=distinct,status=status,
                gene_set=np.where(np.isin(names,marker_genes),'Selected marker panel','Other measured genes'))))
            audits.append(dict(slice=entry['slice'],query_group=str(group),n_original_queries=len(ix),
                n_distinct_neighbor_counts=len(np.unique(xx)),minimum_count=float(xx.min()),maximum_count=float(xx.max())))
        print('Across-cell original-logit slopes:',entry['label'],len(original),'queries',flush=True)
        del a, p, y, raw, expressed
    result = pd.concat(frames,ignore_index=True)
    result.to_csv(output_dir/'slice_gene_slopes.csv',index=False)
    pd.DataFrame(audits).to_csv(output_dir/'query_count_support.csv',index=False)
    complete.write_text(json.dumps(config,indent=2)+'\n')
    return result
