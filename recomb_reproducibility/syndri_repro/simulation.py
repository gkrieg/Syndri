"""Synthetic admixture, MMD-squared, and known-spillover detection curves."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import precision_recall_curve, average_precision_score
from . import ROOT
from .io import config, data_dir, read, write, spatial_files, predicted_file, csr
from .metrics import normalize_log


def squared_distances(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T, 0)


def bandwidth(x, y, seed=87):
    pooled = np.vstack([x, y]).astype(np.float32)
    rng = np.random.default_rng(seed); values = []
    for _ in range(10):
        i = rng.integers(0, len(pooled), 1000); j = rng.integers(0, len(pooled), 1000)
        d = np.square(pooled[i] - pooled[j], dtype=np.float32).sum(1)
        values.append(d[d > 0])
    positive = np.concatenate(values)
    return float(np.median(positive)) if len(positive) else 1.


def mmd2(x, y, sigma2):
    return float(max(0., np.exp(-squared_distances(x,x)/(2*sigma2)).mean() +
        np.exp(-squared_distances(y,y)/(2*sigma2)).mean() -
        2*np.exp(-squared_distances(x,y)/(2*sigma2)).mean()))


def sample(obj, seed=0):
    from Heimdall.admixture_sampling import build_neighbor_dict, sample_admixture_from_neighbor_dict
    cfg = config()['simulation']
    obj = build_neighbor_dict(obj, k_neighbors=cfg['k'], coord_cols=tuple(config()['datasets']['SIM']['coords']),
        tissue_col='spatial_slice' if 'spatial_slice' in obj.obs else None,
        cell_type_col='cell_type', min_source_counts=cfg['min_source_counts'], use_same_cell_type_neighbors=False)
    added, _ = sample_admixture_from_neighbor_dict(obj, tau_range=tuple(cfg['tau']), n_repeats=1,
        min_sampled_transcript_threshold=cfg['min_added_counts'], only_sample_unexpressed_genes=True,
        rng=np.random.default_rng(seed), return_sparse=True)
    added = csr(added)
    obj.layers['original_counts'] = csr(obj.X)
    obj.layers['synthetic_admixture_counts'] = added
    obj.X = obj.X + added
    total = np.asarray(obj.X.sum(1)).ravel()
    obj.obs['achieved_admixture_burden'] = np.divide(np.asarray(added.sum(1)).ravel(), total,
        out=np.zeros_like(total, dtype=float), where=total > 0)
    return obj


def generate_test(rebuilt=False, seed=0):
    path = ROOT/'outputs/preprocessed/SIM/test.h5ad' if rebuilt else data_dir('SIM')/'test.h5ad'
    obj = read(path)
    for i, sl in enumerate(config()['simulation']['test_slices']):
        subset = obj[obj.obs['spatial_slice'].astype(str) == sl].copy()
        result = sample(subset, seed + i*1000)
        write(result, ROOT/'outputs/simulated_test'/f'{sl}.h5ad')


def mmd_table(recompute=False):
    dest = ROOT/'outputs/tables/figure3_mmd.csv'
    if dest.exists() and not recompute:
        return pd.read_csv(dest)
    cfg = config()['simulation']
    training = read(data_dir('SIM')/'training.h5ad')
    training = training[(training.obs['split'].astype(str) == 'train') &
                        (training.obs['cell_type'].astype(str) != 'Treg')].copy()
    records = []
    for sl_index, sl in enumerate(sorted(training.obs['spatial_slice'].astype(str).unique())):
        mapped = training[training.obs['spatial_slice'].astype(str) == sl].copy()
        real = read(data_dir('NC')/'spatial'/f'{sl}.h5ad')
        ids = [str(value).removeprefix(f'{sl}:') for value in mapped.obs_names]
        rows = real.obs_names.get_indexer(ids)
        if (rows < 0).any():
            raise ValueError(f'{sl}: mapped spatial cell IDs not found')
        real = real[rows].copy()
        shared = mapped.var_names.intersection(real.var_names, sort=False)
        mapped, real = mapped[:, shared].copy(), real[:, shared].copy()
        labels = mapped.obs['cell_type'].astype(str).to_numpy()
        if not np.array_equal(labels, real.obs['cell_type_harmonized'].astype(str).to_numpy()):
            raise ValueError(f'{sl}: annotation mismatch')
        rng = np.random.default_rng(91827)
        queries = []
        for label in sorted(set(labels)):
            indices = np.flatnonzero(labels == label)
            if len(indices) >= 2:
                queries.extend(rng.choice(indices, min(cfg['max_query_per_cell_type'], len(indices)), replace=False))
        queries = np.asarray(queries)
        base = normalize_log(mapped.X[queries]).toarray()
        target = normalize_log(real.X[queries]).toarray()
        strata = {}
        for label in sorted(set(labels[queries])):
            mask = labels[queries] == label
            strata[label] = (mask, bandwidth(base[mask], target[mask], 91827+sl_index))
        # Query order and RNG streams match the notebook; sources remain available.
        from Heimdall.admixture_sampling import build_neighbor_dict, sample_admixture_from_neighbor_dict
        neighbors = build_neighbor_dict(mapped, k_neighbors=9, coord_cols=tuple(config()['datasets']['SIM']['coords']),
            tissue_col='spatial_slice', cell_type_col='cell_type', min_source_counts=400, use_same_cell_type_neighbors=False)
        query_ids = mapped.obs_names[queries].tolist()
        qdict = {value: neighbors.uns['neighbor_dict'].get(value, []) for value in query_ids}
        source_ids = list(dict.fromkeys(value for values in qdict.values() for value in values))
        selected = list(dict.fromkeys(query_ids + source_ids))
        sampling = mapped[selected].copy(); sampling.uns['neighbor_dict'] = qdict
        qrows = sampling.obs_names.get_indexer(query_ids)
        for seed in cfg['seeds']:
            added, _ = sample_admixture_from_neighbor_dict(sampling, tau_range=tuple(cfg['tau']),
                n_repeats=1, min_sampled_transcript_threshold=10, only_sample_unexpressed_genes=True,
                rng=np.random.default_rng(seed+1000*sl_index), return_sparse=True)
            admixed = normalize_log(sampling.X[qrows] + csr(added)[qrows]).toarray()
            for label, (mask, sigma2) in strata.items():
                records.append(dict(slice=sl, cell_type=label, seed=seed, n_query=int(mask.sum()),
                    mmd2_before=mmd2(base[mask], target[mask], sigma2),
                    mmd2_after=mmd2(admixed[mask], target[mask], sigma2)))
        print('MMD:', sl, flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records); table.to_csv(dest, index=False)
    return table


def pr_table(recompute=False):
    folder = ROOT/'outputs/tables'; folder.mkdir(parents=True, exist_ok=True)
    cache, summary_path = folder/'figure3_pr_curves.csv', folder/'figure3_detection.csv'
    if cache.exists() and summary_path.exists() and (folder/'figure3_removed_counts.csv').exists() and not recompute:
        return pd.read_csv(cache), pd.read_csv(summary_path)
    labels, scores, known_removed, native_removed = [], [], [], []
    for path in spatial_files('SIM'):
        if path.stem not in config()['simulation']['test_slices']:
            continue
        original = read(path, layers=['original_counts', 'synthetic_admixture_counts', 'admix_probs'])
        pred_path = predicted_file('SIM', path)
        obj = original if pred_path == path else read(pred_path, layers=['admix_probs','original_counts','synthetic_admixture_counts'])
        if 'synthetic_admixture_counts' in obj.layers:
            original = obj
        rows = obj.obs_names.get_indexer(original.obs_names); cols = obj.var_names.get_indexer(original.var_names)
        if (rows < 0).any() or (cols < 0).any():
            raise ValueError('Prediction identifiers do not cover synthetic test input')
        prob = obj.layers['admix_probs'][rows][:, cols]
        entries = original.X.tocoo()
        labels.append(np.asarray(original.layers['synthetic_admixture_counts'][entries.row, entries.col]).ravel() > 0)
        scores.append(np.asarray(prob[entries.row, entries.col]).ravel().astype(np.float32))
    y = np.concatenate(labels); p = np.concatenate(scores)
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(precision), where=precision+recall > 0)
    best = int(np.argmax(f1[:-1])); threshold = float(thresholds[best])
    summary = []; curves = []
    for method, values in [('Syndri', p), ('Syndri (binarized)', (p >= threshold).astype(float))]:
        pr, rc, th = precision_recall_curve(y, values)
        fs = np.divide(2*pr*rc, pr+rc, out=np.zeros_like(pr), where=pr+rc > 0)
        optimum = int(np.argmax(fs[:-1]))
        summary.append(dict(method=method, average_precision=float(average_precision_score(y, values)),
            best_f1=float(fs[optimum]), threshold=float(th[optimum]),
            best_precision=float(pr[optimum]), best_recall=float(rc[optimum]),
            positive_prevalence=float(y.mean()), n_entries=len(y),
            threshold_selected_on='pooled test data; descriptive oracle, not validation selection'))
        # Keep exact AP/F1 above; thin only the plotted curve.
        keep = np.unique(np.r_[np.linspace(0, len(pr)-1, min(2000, len(pr)), dtype=int), optimum])
        curves.append(pd.DataFrame(dict(method=method, precision=pr[keep], recall=rc[keep])))
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    pd.concat(curves).to_csv(cache, index=False)
    # Whole-entry removal fractions at the descriptive test-selected cutoff.
    removal = []
    for path in spatial_files('SIM'):
        if path.stem not in config()['simulation']['test_slices']: continue
        truth = read(path, layers=['original_counts', 'synthetic_admixture_counts', 'admix_probs'])
        pred = read(predicted_file('SIM', path), layers=['admix_probs','original_counts','synthetic_admixture_counts'])
        if 'synthetic_admixture_counts' in pred.layers:
            truth = pred
        rows = pred.obs_names.get_indexer(truth.obs_names); cols = pred.var_names.get_indexer(truth.var_names)
        mask = csr(pred.layers['admix_probs'][rows][:, cols] >= threshold)
        for key in ['original_counts', 'synthetic_admixture_counts']:
            total = float(truth.layers[key].sum())
            removal.append(dict(slice=path.stem, count_class=key, threshold=threshold,
                fraction_removed=float(truth.layers[key].multiply(mask).sum()) / total if total else np.nan))
    pd.DataFrame(removal).to_csv(folder/'figure3_removed_counts.csv', index=False)
    return pd.concat(curves), pd.DataFrame(summary)

