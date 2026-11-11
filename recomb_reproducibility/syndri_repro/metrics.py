"""Syndri/Raw metrics, with no dependency on other methods or their rankings."""
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from . import ROOT
from .io import config, dataset, data_dir, spatial_files, predicted_file, read, valid_labels, qc, csr
from .correction import correct


def normalize_log(counts):
    counts = csr(counts).astype(np.float64)
    totals = np.asarray(counts.sum(1)).ravel()
    result = sparse.diags(np.divide(10000., totals, out=np.zeros_like(totals), where=totals > 0)) @ counts
    result.data = np.log1p(result.data)
    return result.tocsr()


def variance(counts):
    n = counts.shape[0]
    mean = np.asarray(counts.mean(0)).ravel()
    return np.maximum(0, np.asarray(counts.power(2).mean(0)).ravel() - mean**2) * n / max(n-1, 1)


def matrices(obj, name, unit=None):
    if 'admix_probs' not in obj.layers:
        raise KeyError('Missing admix_probs; run inference first')
    settings = dict(config()['correction']); settings['unit'] = unit or settings['unit']
    cleaned, _ = correct(obj.X, obj.layers['admix_probs'], obj.obs[dataset(name)['label']], **settings)
    return {'Raw': csr(obj.X), 'Syndri': cleaned}


def adjacency(obj, label, coords, order, k=None, eps=1e-8):
    k = k or config()['metrics']['adjacency_k']
    labels = obj.obs[label].astype(str).to_numpy()
    xy = obj.obs[coords].to_numpy(float)
    keep = np.isfinite(xy).all(1) & np.isin(labels, order)
    labels, xy = labels[keep], xy[keep]
    k = min(k, len(xy)-1)
    if k < 1:
        raise ValueError('At least two cells with valid coordinates are required')
    graph = NearestNeighbors(n_neighbors=k+1).fit(xy).kneighbors(xy, return_distance=False)
    # Exclude each query by index, including when coordinates are tied.
    neighbors = np.array([row[row != i][:k] for i, row in enumerate(graph)])
    counts = np.zeros((len(order), len(order)), dtype=np.float64)
    freq = np.array([(labels == value).mean() for value in order])
    for i, query in enumerate(order):
        selected = labels[neighbors[labels == query]].ravel()
        if not len(selected):
            continue
        counts[i] = np.array([(selected == neighbor).sum() for neighbor in order])
    observed = counts / counts.sum()
    return np.log2((observed + eps) / (np.outer(freq, freq) + eps))


def separation(counts, labels, max_cells=None):
    from scib_metrics import nmi_ari_cluster_labels_leiden
    from scib_metrics.nearest_neighbors import pynndescent
    if max_cells and counts.shape[0] > max_cells:
        selected = np.sort(np.random.default_rng(42).choice(counts.shape[0], max_cells, replace=False))
        counts, labels = counts[selected], labels[selected]
    obj = ad.AnnData(normalize_log(counts))
    n_pcs = min(50, min(obj.shape)-1)
    if n_pcs < 2 or len(set(labels)) < 2:
        raise ValueError('Too few cells, genes, or cell types for separation metrics')
    sc.pp.pca(obj, n_comps=n_pcs, random_state=0)
    embedding = obj.obsm['X_pca']
    neighbors = pynndescent(embedding, n_neighbors=min(15, len(labels)-1), random_state=0, n_jobs=8)
    scores = nmi_ari_cluster_labels_leiden(neighbors, labels, optimize_resolution=False,
                                         resolution=1.0, n_jobs=8, seed=0)
    return {'ARI': float(scores['ari']), 'NMI': float(scores['nmi']),
            'Silhouette': float(silhouette_score(embedding, labels, metric='cosine')),
            'separation_n_cells': len(labels)}


def evaluate(name, recompute=False, max_cells=None, unit=None):
    output = ROOT / 'outputs/metrics' / name; output.mkdir(parents=True, exist_ok=True)
    reference = read(data_dir(name)/'reference.h5ad')
    ref_labels = reference.obs[dataset(name)['reference_label']].astype(str).to_numpy()
    for path in spatial_files(name):
        dest = output / f'{path.stem}.csv'
        if dest.exists() and not recompute:
            continue
        print('Metrics:', name, path.stem, flush=True)
        obj = read(predicted_file(name, path), layers=['admix_probs'])
        valid = valid_labels(obj.obs[dataset(name)['label']])
        obj = obj[valid].copy()
        labels = obj.obs[dataset(name)['label']].astype(str).to_numpy()
        shared = obj.var_names.intersection(reference.var_names, sort=False)
        if len(shared) < 3:
            raise ValueError(f'{name}/{path.stem}: shared gene identifiers missing')
        col = obj.var_names.get_indexer(shared)
        ref = csr(reference[:, shared].X); ref_norm = normalize_log(ref)
        records, type_records = [], []
        for method, counts in matrices(obj, name, unit).items():
            passing = qc(counts) if config()['metrics']['apply_qc'] else np.ones(obj.n_obs, dtype=bool)
            score = separation(counts[passing], labels[passing], max_cells)
            score.update(dataset=name, slice=path.stem, method=method,
                median_genes=float(np.median(np.diff(counts[passing].indptr))),
                median_transcripts=float(np.median(np.asarray(counts[passing].sum(1)).ravel())),
                correction_unit=unit or config()['correction']['unit'], max_cells=max_cells)
            norm = normalize_log(counts)[:, col]
            for cell_type in sorted(set(labels) & set(ref_labels)):
                st_mask, ref_mask = labels == cell_type, ref_labels == cell_type
                left = np.asarray(counts[:, col][st_mask].sum(0)).ravel()
                right = np.asarray(ref[ref_mask].sum(0)).ravel()
                left = np.log1p(left * 10000 / max(left.sum(), 1))
                right = np.log1p(right * 10000 / max(right.sum(), 1))
                cosine = float(left @ right / max(np.linalg.norm(left)*np.linalg.norm(right), 1e-15))
                rho = np.nan
                if st_mask.sum() >= 50 and ref_mask.sum() >= 50:
                    ref_var = variance(ref_norm[ref_mask])
                    genes = (np.asarray((ref[ref_mask] > 0).mean(0)).ravel() >= .05) & (ref_var > 0)
                    if genes.sum() >= 2:
                        rho = float(spearmanr(variance(norm[st_mask])[genes], ref_var[genes]).statistic)
                type_records.append(dict(dataset=name, slice=path.stem, method=method,
                    cell_type=cell_type, reference_cosine=cosine, variance_spearman=rho,
                    n_spatial=int(st_mask.sum()), n_reference=int(ref_mask.sum())))
            current = pd.DataFrame(type_records)
            score['reference_cosine'] = current.loc[current.method == method, 'reference_cosine'].mean()
            score['variance_spearman'] = current.loc[current.method == method, 'variance_spearman'].mean()
            records.append(score)
        pd.DataFrame(records).to_csv(dest, index=False)
        pd.DataFrame(type_records).to_csv(output/f'{path.stem}_celltypes.csv', index=False)

