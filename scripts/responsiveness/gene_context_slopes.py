"""Gene-level logit slopes versus malignant composition of matched swapped contexts."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

T_CELL_TYPES = ['CD4-positive, alpha-beta memory T cell', 'cytotoxic T cell', 'natural T-regulatory cell']

def compute_gene_context_slopes(run, output_dir, marker_genes, pool_t_cells=True, min_queries=10, context_predictor="model_proportion"):
    if context_predictor not in {"model_proportion", "donor_physical20"}:
        raise ValueError(context_predictor)
    physical20 = context_predictor == "donor_physical20"
    import anndata as ad
    from scipy import sparse
    run, output_dir = Path(run), Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    config = dict(marker_genes=list(marker_genes), pool_t_cells=pool_t_cells,
        min_queries_per_section_gene=min_queries, response_scale='logit', logit_clip=1e-6,
        slope_unit='logit / additional malignant neighbor' if physical20 else 'logit / +0.10 malignant model-context proportion',
        predictor='substituted context donor malignant count among 20 nearest non-self cells' if physical20 else 'malignant cell column in neighborhood_proportions',
        contexts='same matched donor panel within each section and original query type',
        gene_filter='expressed in raw query X > 0', min_contexts=10,
        min_distinct_predictor_values=3, min_predictor_range=2 if physical20 else 0.2)
    complete = output_dir / 'complete.json'
    if complete.exists():
        if json.loads(complete.read_text()) != config:
            raise ValueError('Cached gene slope definition differs')
        return pd.read_csv(output_dir / 'slice_gene_slopes.csv')
    frames, audits = [], []
    count_paths = None
    if physical20:
        from physical20_context_counts import ensure_physical20_context_counts
        count_paths = ensure_physical20_context_counts(run)
    expected_names = None
    for entry in json.loads((run / 'design.json').read_text())['slices']:
        a = ad.read_h5ad(entry['predictions'])
        obs = a.obs.reset_index(drop=True)
        names = a.var.get('feature_name', pd.Series(a.var_names, index=a.var_names)).fillna('').astype(str).to_numpy()
        names = np.where(names == '', a.var_names.astype(str), names)
        if len(set(names)) != len(names):
            raise ValueError('Gene names must be unique for gene-level heatmaps')
        if expected_names is None:
            expected_names = names.copy()
        elif not np.array_equal(names, expected_names):
            raise ValueError('Gene panels differ between sections')
        p = a.layers['admix_probs']
        p = p.toarray() if sparse.issparse(p) else np.asarray(p)
        if physical20:
            count_lookup = pd.read_csv(count_paths[entry['slice']]).set_index('cell_id').malignant_neighbors_k20
            comp = obs.context_source_cell_id.astype(str).map(count_lookup).to_numpy(dtype=float)
            if not np.isfinite(comp).all():
                raise ValueError('Unmatched physical context donor counts')
        else:
            comp = a.obsm['neighborhood_proportions']['malignant cell'].to_numpy(dtype=float)
        slopes, query_tables = [], []
        for cell_type, part in obs.groupby('query_cell_type', observed=True):
            donors = sorted(part.loc[part.context_kind.eq('representative'), 'context_source_cell_id'].astype(str).unique())
            current = part.loc[~part.context_kind.eq('unchanged_control') &
                               part.context_source_cell_id.astype(str).isin(donors)].copy()
            current = current.sort_values(['focal_cell_id', 'context_source_cell_id'])
            if current.duplicated(['focal_cell_id','context_source_cell_id']).any():
                raise ValueError('Duplicate matched query/context')
            sizes = current.groupby('focal_cell_id', observed=True).size()
            if not sizes.eq(len(donors)).all():
                raise ValueError('Queries do not share the matched context panel')
            ids = current.focal_cell_id.drop_duplicates().astype(str).to_numpy()
            nq, nc = len(ids), len(donors)
            original = part.loc[part.context_kind.eq('original')].copy()
            if original.focal_cell_id.duplicated().any():
                raise ValueError('Duplicate original query rows')
            original['source_position'] = original.index
            original_positions = original.set_index('focal_cell_id').reindex(ids).source_position.to_numpy()
            if pd.isna(original_positions).any():
                raise ValueError('Missing original query')
            original_positions = original_positions.astype(int)
            x = comp[current.index.to_numpy()].reshape(nq, nc)
            if not np.allclose(x, x[:1], equal_nan=False):
                raise ValueError('Context composition differs across matched queries')
            x = x[0]
            valid = (nc >= 10 and np.unique(x).size >= 3 and np.ptp(x) >= (2 if physical20 else 0.2))
            beta = np.full((nq, len(names)), np.nan)
            raw = a.X[original_positions]
            expressed = (raw.toarray() if sparse.issparse(raw) else np.asarray(raw)) > 0
            if valid:
                selected = p[current.index.to_numpy()].astype(float)
                if not np.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
                    raise ValueError('Invalid admixture probabilities')
                selected = np.clip(selected, 1e-6, 1-1e-6)
                logits = (np.log(selected) - np.log1p(-selected)).reshape(nq, nc, len(names))
                centered = x - x.mean()
                beta = (1.0 if physical20 else 0.1) * np.einsum('qcg,c->qg', logits, centered) / np.dot(centered, centered)
                beta[~expressed] = np.nan
            slopes.append(beta)
            query_tables.append(pd.DataFrame(dict(query_cell_id=ids, query_cell_type=str(cell_type))))
            audits.append(dict(slice=entry['slice'], query_cell_type=str(cell_type),
                n_queries=nq, n_contexts=nc, n_distinct_predictor_values=len(np.unique(x)),
                predictor_range=float(np.ptp(x)), status='ok' if valid else 'insufficient_context_variation',
                donor_ids_json=json.dumps(donors)))
        beta = np.vstack(slopes)
        queries = pd.concat(query_tables, ignore_index=True)
        queries['query_group'] = queries.query_cell_type.replace(
            {t:'T cells (pooled)' for t in T_CELL_TYPES}) if pool_t_cells else queries.query_cell_type
        # Store the query-level values compactly, without expanding millions of CSV rows.
        np.savez_compressed(output_dir / (entry['slice'] + '_query_gene_slopes.npz'),
            slopes=beta.astype(np.float32), genes=names.astype(str),
            query_cell_id=queries.query_cell_id.to_numpy(dtype=str),
            query_cell_type=queries.query_cell_type.to_numpy(dtype=str),
            query_group=queries.query_group.to_numpy(dtype=str))
        for query_group, ix in queries.groupby('query_group', observed=True).groups.items():
            values = beta[np.asarray(ix)]
            count = np.isfinite(values).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median = np.nanmedian(values, axis=0)
            median[count < min_queries] = np.nan
            frames.append(pd.DataFrame(dict(slice=entry['slice'], query_group=str(query_group),
                gene=names, median_slope=median, n_queries=count,
                gene_set=np.where(np.isin(names, marker_genes),'Selected marker panel','Other measured genes'))))
        print('Gene slopes complete:', entry['label'], len(queries), 'queries', len(names), 'genes', flush=True)
        del a, p, beta
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir / 'slice_gene_slopes.csv', index=False)
    pd.DataFrame(audits).to_csv(output_dir / 'context_support.csv', index=False)
    complete.write_text(json.dumps(config, indent=2) + '\n')
    return result
