"""Marker-score probability SD across the same saved matched context panels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def compute_probability_marker_sensitivity(run, marker_genes, output_dir):
    import anndata as ad
    from scipy import sparse
    run, output_dir = Path(run), Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    metadata = dict(marker_genes=list(marker_genes), response_scale='probability',
                    definition='SD across matched contexts of mean probabilities for expressed selected genes', ddof=0)
    result_path = output_dir / 'query_probability_sensitivity.csv'
    complete = output_dir / 'complete.json'
    if complete.exists():
        if json.loads(complete.read_text()) != metadata:
            raise ValueError('Probability cache definition differs')
        return pd.read_csv(result_path)
    if result_path.exists():
        raise FileExistsError(result_path)
    result_frames, audits = [], []
    for entry in json.loads((run / 'design.json').read_text())['slices']:
        a = ad.read_h5ad(entry['predictions'])
        obs = a.obs.reset_index(drop=True)
        names = a.var.get('feature_name', pd.Series(a.var_names, index=a.var_names)).fillna('').astype(str).to_numpy()
        names = np.where(names == '', a.var_names.astype(str), names)
        selected = np.flatnonzero(np.isin(names, marker_genes))
        raw = a.X[:, selected]
        expressed = (raw.toarray() if sparse.issparse(raw) else np.asarray(raw)) > 0
        p = a.layers['admix_probs'][:, selected]
        p = (p.toarray() if sparse.issparse(p) else np.asarray(p)).astype(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid predicted probabilities')
        n_markers = expressed.sum(axis=1)
        scores = np.divide((p * expressed).sum(axis=1), n_markers,
                           out=np.full(len(obs), np.nan), where=n_markers > 0)
        for cell_type, group in obs.groupby('query_cell_type', observed=True):
            donors = sorted(group.loc[group.context_kind.eq('representative'), 'context_source_cell_id'].astype(str).unique())
            if len(donors) < 10:
                raise ValueError('Insufficient matched context panel')
            current = group.loc[~group.context_kind.eq('unchanged_control') &
                                group.context_source_cell_id.astype(str).isin(donors)].copy()
            if current.duplicated(['focal_cell_id', 'context_source_cell_id']).any():
                raise ValueError('Duplicate query/donor pairs')
            support = current.groupby('focal_cell_id', observed=True).size()
            if not support.eq(len(donors)).all():
                raise ValueError('Queries do not share the same context panel')
            current['marker_probability_score'] = scores[current.index]
            current['n_expressed_selected_markers'] = n_markers[current.index]
            summary = current.groupby('focal_cell_id', observed=True).agg(
                marker_probability_score_sd=('marker_probability_score', lambda x: x.std(ddof=0)),
                n_panel_contexts=('context_source_cell_id', 'size'),
                n_expressed_selected_markers=('n_expressed_selected_markers', 'first')).reset_index().rename(columns={'focal_cell_id':'query_cell_id'})
            result_frames.append(summary.assign(slice=entry['slice'], query_cell_type=str(cell_type)))
            audits.append(dict(slice=entry['slice'], query_cell_type=str(cell_type),
                               n_queries=len(summary), n_contexts=len(donors), donor_ids_json=json.dumps(donors)))
        print('Probability sensitivity:', entry['label'], flush=True)
        del a, raw, expressed, p
    frame = pd.concat(result_frames, ignore_index=True)
    distances = pd.read_csv(run / 'summaries_distance/query_original_distance_scores.csv')
    frame = frame.merge(distances[['slice', 'query_cell_id', 'query_distance_um']],
                        on=['slice', 'query_cell_id'], how='left', validate='one_to_one')
    if frame.query_distance_um.isna().any():
        raise ValueError('Unmatched query distances')
    supported = frame.n_expressed_selected_markers.gt(0)
    if not frame.loc[supported, 'marker_probability_score_sd'].between(0, 0.5 + 1e-12).all():
        raise ValueError('Probability SD outside its valid range')
    frame.to_csv(result_path, index=False)
    pd.DataFrame(audits).to_csv(output_dir / 'matched_context_panels.csv', index=False)
    complete.write_text(json.dumps(metadata, indent=2) + '\n')
    return frame
