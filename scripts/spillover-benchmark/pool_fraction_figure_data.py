"""Cached figure data for the optimized donor-pool-fraction benchmark.

Only reads existing counts and prediction files. Never trains or runs inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve
from sklearn.neighbors import NearestNeighbors

from diagnose_test_admixture import read_metadata, read_counts, TEST_SLICES, TYPE_ORDER
from prepare_weighted_test_admixture import matched_real, pilot_rows
from search_admixture_parameters import normalized_log_counts, rbf_bandwidth_squared
from search_pool_fraction_admixture import squared_distances, mmd_squared

METHODS = ['Raw', 'ResolVI', 'Syndri', 'SPLIT', 'CellAdmix', 'ovrlpy']
PR_METHODS = [name for method in METHODS[1:] for name in (method, method+' (binarized)')]
CURVE_COLUMNS = ['curve', 'method', 'x', 'y', 'threshold', 'metric_value', 'auprc',
    'average_precision', 'positive_prevalence', 'best_f1', 'best_f1_threshold']
REMOVAL_COLUMNS = ['slice', 'method', 'auprc', 'known_removed_fraction',
    'unadmixed_removed_fraction', 'n_cells', 'n_observed_entries', 'n_positive_entries', 'status']


def signature(paths, settings):
    inputs = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) if p.exists()
              else (str(p.resolve()), None, None) for p in map(Path, paths)]
    return hashlib.sha256(json.dumps(dict(inputs=inputs, settings=settings,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()


def write_manifest(path, digest):
    path.write_text(json.dumps({'signature': digest}, indent=2)+'\n')


def cache_matches(path, digest):
    return path.exists() and json.loads(path.read_text()).get('signature') == digest


def calibration_mmd(root):
    selected = json.loads((root/'search/selected_parameters.json').read_text())['lambda_max']
    data = pd.read_csv(root/'search/stratum_metrics.csv')
    before = data.loc[data.lambda_max.eq(0), ['slice', 'cell_type', 'seed', 'mmd2']].rename(columns={'mmd2': 'mmd2_before'})
    after = data.loc[np.isclose(data.lambda_max, selected), ['slice', 'cell_type', 'seed', 'mmd2', 'n_query']]
    paired = after.rename(columns={'mmd2': 'mmd2_after'}).merge(before,
        on=['slice', 'cell_type', 'seed'], validate='one_to_one')
    summary = paired.groupby(['slice', 'cell_type'], as_index=False).agg(
        mmd2_before=('mmd2_before', 'first'), mmd2_after_mean=('mmd2_after', 'mean'),
        mmd2_after_sd=('mmd2_after', 'std'), n_query=('n_query', 'first'))
    summary['delta_mmd2'] = summary.mmd2_after_mean - summary.mmd2_before
    return summary


def test_mmd(root, spatial_dir, output_dir, max_queries=500, query_seed=91827, recompute=False):
    path, manifest = output_dir/'test_mmd_by_slice_celltype.csv', output_dir/'test_mmd_manifest.json'
    slices = sorted(TEST_SLICES)
    inputs = [root/'slices'/f'{s}.h5ad' for s in slices] + [spatial_dir/f'st_integrated_slice-{s}.h5ad' for s in slices]
    digest = signature(inputs, dict(max_queries=max_queries, query_seed=query_seed,
        preprocessing='CP10K_log1p', bandwidth='median_clean_real_squared_distance'))
    if not recompute and path.exists() and cache_matches(manifest, digest):
        return pd.read_csv(path)
    records = []
    for name in slices:
        source = root/'slices'/f'{name}.h5ad'
        obs, var, config = read_metadata(source)
        if config.get('admixture_target_mode') != 'pool_fraction' or not obs.split.astype(str).eq('test').all():
            raise ValueError(f'Expected held-out pool-fraction data: {source}')
        rows = pilot_rows(obs, max_queries, query_seed)
        selected = obs.iloc[rows]
        clean = read_counts(source, 'layers/original_counts', var.index, rows)
        mixed = read_counts(source, 'layers/admixed_counts', var.index, rows)
        real, _ = matched_real(spatial_dir/f'st_integrated_slice-{name}.h5ad', selected, var.index)
        x0, x1, y = map(normalized_log_counts, (clean, mixed, real))
        for ct in sorted(selected.cell_type.astype(str).unique()):
            mask = selected.cell_type.astype(str).eq(ct).to_numpy()
            if mask.sum() < 2:
                continue
            bandwidth = rbf_bandwidth_squared(x0[mask], y[mask])
            metric = dict(target=y[mask], bandwidth_squared=bandwidth,
                yy_mean=np.exp(-squared_distances(y[mask], y[mask])/(2*bandwidth)).mean())
            before, after = mmd_squared(x0[mask], metric), mmd_squared(x1[mask], metric)
            records.append(dict(slice=name, cell_type=ct, mmd2_before=before,
                mmd2_after_mean=after, mmd2_after_sd=0., delta_mmd2=after-before,
                n_query=int(mask.sum()), bandwidth_squared=bandwidth, n_saved_simulations=1))
        print(f'Held-out MMD²: {name}, {len(rows):,} fixed queries', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(path, index=False)
    write_manifest(manifest, digest)
    return result


def method_paths(root, ovrlpy_threshold=.7):
    if ovrlpy_threshold not in {.5, .7}:
        raise ValueError('Saved ovrlpy corrections are available at VSI thresholds 0.5 and 0.7')
    ovrlpy_layer = 'ovrlpy_corrected_counts_vsi_lt_'+str(ovrlpy_threshold).replace('.', '_')
    return {name: {
        'ResolVI': (root/'resolvi'/f'{name}.h5ad', 'resolvi_corrected_counts'),
        'Syndri': (root/'syndri'/f'{name}_syndri_predictions.h5ad', 'admix_probs'),
        'SPLIT': (root/'split/merged_h5ad'/f'{name}.h5ad', 'split_purified_counts_X'),
        'CellAdmix': (root/'celladmix'/f'{name}.h5ad', 'celladmix_corrected_counts'),
        'ovrlpy': (root/'ovrlpy'/f'{name}.h5ad', ovrlpy_layer),
    } for name in sorted(TEST_SLICES)}


def availability(root, ovrlpy_threshold=.7):
    rows = []
    for name, methods in method_paths(root, ovrlpy_threshold).items():
        for method, (path, layer) in methods.items():
            status = 'awaiting predictions'
            if path.exists():
                with h5py.File(path, 'r') as handle:
                    status = 'available' if layer in handle['layers'] else f'missing {layer}'
            rows.append(dict(slice=name, method=method, status=status, path=str(path)))
    return pd.DataFrame(rows)


def matrix(path, key):
    with h5py.File(path, 'r') as handle:
        value = ad.io.read_elem(handle[key])
    return sparse.csr_matrix(value)


def values_at(value, rows, columns):
    return np.asarray(value[rows, columns]).ravel()


def checked_output(source_path, output_path, layer):
    """Align predictions and verify their input counts belong to this simulation."""
    obs, var, _ = read_metadata(source_path)
    out_obs, out_var, _ = read_metadata(output_path)
    row_idx, col_idx = obs.index.get_indexer(out_obs.index), var.index.get_indexer(out_var.index)
    if (row_idx < 0).any() or (col_idx < 0).any():
        raise ValueError(f'Prediction IDs do not match the new simulation: {output_path}')
    admixed = read_counts(source_path, 'layers/admixed_counts', var.index, row_idx)[:, col_idx]
    added = read_counts(source_path, 'layers/synthetic_admixture_counts', var.index, row_idx)[:, col_idx]
    native = read_counts(source_path, 'layers/original_counts', var.index, row_idx)[:, col_idx]
    with h5py.File(output_path, 'r') as handle:
        input_key = 'layers/admixed_counts' if 'admixed_counts' in handle['layers'] else 'X'
    supplied = matrix(output_path, input_key)
    difference = supplied-admixed
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError(f'Prediction input differs from the optimized synthetic counts: {output_path}')
    return admixed, added, native, matrix(output_path, 'layers/'+layer)


def score_output(source, path, layer, method, threshold, batch_size=4096):
    admixed, added, native, output = checked_output(source, path, layer)
    label_parts, score_parts, removal_score_parts = [], [], []
    known_removed = native_removed = known_total = native_total = 0.
    for start in range(0, admixed.shape[0], batch_size):
        stop = min(start+batch_size, admixed.shape[0])
        observed = admixed[start:stop].tocoo()
        keep = observed.data > 0
        rows, genes, raw = observed.row[keep], observed.col[keep], observed.data[keep]
        y_added = values_at(added[start:stop], rows, genes)
        y_native = values_at(native[start:stop], rows, genes)
        prediction = values_at(output[start:stop], rows, genes)
        finite = np.isfinite(prediction)
        if not finite.all():
            raise ValueError(f'Nonfinite predictions on observed entries: {path}')
        score = prediction if method == 'Syndri' else 1-prediction/raw
        corrected = np.where(prediction >= threshold, 0., raw) if method == 'Syndri' else prediction
        removed = np.maximum(raw-corrected, 0.)
        known_removed += np.minimum(y_added, removed).sum()
        known_total += y_added.sum()
        clean_entries = y_added <= 0
        native_removed += np.minimum(y_native[clean_entries], removed[clean_entries]).sum()
        native_total += y_native[clean_entries].sum()
        label_parts.append((y_added > 0).astype(np.uint8))
        score_parts.append(score.astype(np.float32))
        if method == 'Syndri':
            # The original notebook's per-slice removal AP uses hard-zeroed
            # counts at the configured threshold; PR/ROC still use admix_probs.
            removal_score_parts.append((prediction >= threshold).astype(np.uint8))
    labels, scores = np.concatenate(label_parts), np.concatenate(score_parts)
    removal_scores = np.concatenate(removal_score_parts) if method == 'Syndri' else scores
    ap = float(average_precision_score(labels, removal_scores)) if labels.any() else np.nan
    result = dict(auprc=ap, known_removed_fraction=known_removed/known_total if known_total else np.nan,
        unadmixed_removed_fraction=native_removed/native_total if native_total else np.nan,
        n_cells=admixed.shape[0], n_observed_entries=len(labels), n_positive_entries=int(labels.sum()), status='ok')
    return result, labels, scores


def curve_rows(labels, scores, method, max_points):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # sklearn's thresholds align with precision[:-1] and recall[:-1].
    denominator = precision[:-1]+recall[:-1]
    f1 = np.divide(2*precision[:-1]*recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(f1))
    summary = dict(auprc=float(auc(recall, precision)), average_precision=float(average_precision_score(labels, scores)),
        positive_prevalence=float(labels.mean()), best_f1=float(f1[best]), best_f1_threshold=float(thresholds[best]))
    take = np.unique(np.r_[np.linspace(0, len(recall)-1, min(max_points, len(recall)), dtype=int), best])
    padded = np.r_[thresholds, np.nan]
    rows = [dict(curve='PR', method=method, x=recall[i], y=precision[i], threshold=padded[i],
                 metric_value=summary['auprc'], **summary) for i in take]
    if method == 'Syndri' and np.unique(labels).size == 2:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        take = np.unique(np.linspace(0, len(fpr)-1, min(max_points, len(fpr)), dtype=int))
        rows.extend(dict(curve='ROC', method=method, x=fpr[i], y=tpr[i], threshold=roc_thresholds[i],
            metric_value=float(auc(fpr, tpr)), **summary) for i in take)
    return rows, summary


def benchmark_metrics(root, output_dir, threshold=.5, max_points=2000, recompute=False,
                      require_complete_methods=True, ovrlpy_threshold=.7):
    specs = method_paths(root, ovrlpy_threshold)
    inputs = [root/'slices'/f'{s}.h5ad' for s in sorted(TEST_SLICES)] + [
        path for methods in specs.values() for path, _ in methods.values()]
    digest = signature(inputs, dict(threshold=threshold, max_points=max_points,
                                   require_complete_methods=require_complete_methods,
                                   ovrlpy_threshold=ovrlpy_threshold))
    manifest = output_dir/'method_metrics_manifest.json'
    paths = [output_dir/name for name in ['removal_metrics.csv', 'method_curves.csv', 'pr_metric_summary.csv']]
    if not recompute and all(p.exists() for p in paths) and cache_matches(manifest, digest):
        return tuple(pd.read_csv(p) for p in paths)
    removal_rows, curves, summaries = [], [], []
    status = availability(root, ovrlpy_threshold)
    complete = {method: status.loc[status.method.eq(method), 'status'].eq('available').all()
                for method in METHODS[1:]}
    for name in sorted(TEST_SLICES):
        source = root/'slices'/f'{name}.h5ad'
        _, var, _ = read_metadata(source)
        admixed = read_counts(source, 'layers/admixed_counts', var.index)
        added = read_counts(source, 'layers/synthetic_admixture_counts', var.index)
        n_observed, n_positive = admixed.count_nonzero(), added.count_nonzero()
        removal_rows.append(dict(slice=name, method='Raw', auprc=n_positive/n_observed,
            known_removed_fraction=0., unadmixed_removed_fraction=0., n_cells=admixed.shape[0],
            n_observed_entries=n_observed, n_positive_entries=n_positive, status='ok'))
    for method in METHODS[1:]:
        # A newly completed comparator must not force every existing method's
        # full-resolution scores and curves to be calculated again.
        method_inputs = [root/'slices'/f'{name}.h5ad' for name in sorted(TEST_SLICES)] + [
            specs[name][method][0] for name in sorted(TEST_SLICES)]
        method_digest = signature(method_inputs, dict(method=method, threshold=threshold,
            max_points=max_points, require_complete_methods=require_complete_methods,
            ovrlpy_threshold=ovrlpy_threshold if method == 'ovrlpy' else None))
        cache_dir = output_dir/'method_metrics_cache'/method.lower()
        cache_paths = [cache_dir/name for name in ['removal.csv', 'curves.csv', 'summary.csv']]
        method_manifest = cache_dir/'manifest.json'
        if not recompute and all(p.exists() for p in cache_paths) and cache_matches(method_manifest, method_digest):
            cached = [pd.read_csv(p).to_dict(orient='records') for p in cache_paths]
            removal_rows.extend(cached[0]); curves.extend(cached[1]); summaries.extend(cached[2])
            print(f'Reusing cached method metrics: {method}', flush=True)
            continue
        first_removal, first_curve, first_summary = len(removal_rows), len(curves), len(summaries)
        labels, scores = [], []
        for name in sorted(TEST_SLICES):
            path, layer = specs[name][method]
            available = status.loc[status.slice.eq(name) & status.method.eq(method), 'status'].iloc[0] == 'available'
            if not available or (require_complete_methods and not complete[method]):
                removal_rows.append(dict(slice=name, method=method, status='awaiting predictions',
                    n_cells=0, n_observed_entries=0, n_positive_entries=0,
                    auprc=np.nan, known_removed_fraction=np.nan, unadmixed_removed_fraction=np.nan))
                continue
            result, y, s = score_output(root/'slices'/f'{name}.h5ad', path, layer, method, threshold)
            removal_rows.append(dict(slice=name, method=method, **result))
            labels.append(y)
            scores.append(s)
        if labels:
            y, s = np.concatenate(labels), np.concatenate(scores)
            if y.any():
                rows, summary = curve_rows(y, s, method, max_points)
                curves.extend(rows)
                summaries.append(dict(method=method, **summary))
                binary_scores = (s >= summary['best_f1_threshold']).astype(np.float32)
                rows, binary_summary = curve_rows(y, binary_scores, method+' (binarized)', max_points)
                tp = np.count_nonzero((binary_scores > 0) & (y > 0))
                fp = np.count_nonzero((binary_scores > 0) & (y == 0))
                fn = np.count_nonzero((binary_scores == 0) & (y > 0))
                binary_summary['best_f1'] = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
                binary_summary['best_f1_threshold'] = summary['best_f1_threshold']
                for row in rows:
                    row.update(best_f1=binary_summary['best_f1'], best_f1_threshold=summary['best_f1_threshold'])
                curves.extend(rows)
                summaries.append(dict(method=method+' (binarized)', **binary_summary))
        cache_dir.mkdir(parents=True, exist_ok=True)
        for frame, path in zip([
                pd.DataFrame(removal_rows[first_removal:], columns=REMOVAL_COLUMNS),
                pd.DataFrame(curves[first_curve:], columns=CURVE_COLUMNS),
                pd.DataFrame(summaries[first_summary:], columns=['method', 'auprc', 'average_precision',
                    'positive_prevalence', 'best_f1', 'best_f1_threshold'])], cache_paths):
            frame.to_csv(path, index=False)
        write_manifest(method_manifest, method_digest)
        print(f'Updated method metrics: {method}', flush=True)
    results = (pd.DataFrame(removal_rows, columns=REMOVAL_COLUMNS),
               pd.DataFrame(curves, columns=CURVE_COLUMNS),
               pd.DataFrame(summaries, columns=['method', 'auprc', 'average_precision',
                   'positive_prevalence', 'best_f1', 'best_f1_threshold']))
    for frame, path in zip(results, paths):
        frame.to_csv(path, index=False)
    write_manifest(manifest, digest)
    return results


def adjacency(root, spatial_dir, slice_name, output_dir, k=6, eps=1e-8, recompute=False):
    source, real_path = root/'slices'/f'{slice_name}.h5ad', spatial_dir/f'st_integrated_slice-{slice_name}.h5ad'
    path, manifest = output_dir/f'{slice_name}_adjacency.csv', output_dir/f'{slice_name}_adjacency_manifest.json'
    digest = signature([source, real_path], dict(k=k, eps=eps))
    if not recompute and path.exists() and cache_matches(manifest, digest):
        data = pd.read_csv(path)
    else:
        simulated, _, _ = read_metadata(source)
        real, _, _ = read_metadata(real_path)
        indices = real.index.get_indexer(simulated.matched_st_id.astype(str))
        if (indices < 0).any():
            raise ValueError('Missing real recipients for adjacency')
        real = real.iloc[indices].copy()
        if not np.array_equal(simulated.cell_type.astype(str), real.cell_type_harmonized.astype(str)):
            raise ValueError('Adjacency annotations differ between matched cells')
        order = [ct for ct in TYPE_ORDER if ct in set(simulated.cell_type.astype(str))]
        records = []
        for name, obs, label in [('Simulated ST slice', simulated, 'cell_type'), ('Real ST slice', real, 'cell_type_harmonized')]:
            xy = obs[['CenterX_global_px', 'CenterY_global_px']].to_numpy(float)
            if not np.isfinite(xy).all():
                raise ValueError('Invalid spatial coordinates')
            codes = pd.Index(order).get_indexer(obs[label].astype(str))
            candidates = NearestNeighbors(n_neighbors=k+1).fit(xy).kneighbors(xy, return_distance=False)
            neighbors = np.array([row[row != i][:k] for i, row in enumerate(candidates)])
            pairs = np.bincount((codes[:,None]*len(order)+codes[neighbors]).ravel(), minlength=len(order)**2).reshape(len(order),len(order))
            observed = pairs/pairs.sum()
            frequency = np.bincount(codes, minlength=len(order))/len(codes)
            enrichment = np.log2((observed+eps)/(np.outer(frequency,frequency)+eps))
            records.append(pd.DataFrame(dict(dataset=name, source_cell_type=np.repeat(order,len(order)),
                neighbor_cell_type=np.tile(order,len(order)), log2_observed_expected=enrichment.ravel(), n_cells=len(obs))))
        data = pd.concat(records, ignore_index=True)
        data.to_csv(path, index=False)
        write_manifest(manifest, digest)
    order = [ct for ct in TYPE_ORDER if ct in set(data.source_cell_type)]
    matrices = {name: part.pivot(index='source_cell_type', columns='neighbor_cell_type', values='log2_observed_expected').reindex(
        index=order, columns=order).to_numpy() for name,part in data.groupby('dataset')}
    return order, matrices
