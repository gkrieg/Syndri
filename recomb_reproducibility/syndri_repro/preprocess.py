"""Validate archived inputs or rebuild the pseudo-spatial training data."""
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
from . import ROOT
from .io import config, dataset, data_dir, spatial_files, read, write, valid_labels


def validate(name):
    records = []
    paths = [data_dir(name)/'training.h5ad', data_dir(name)/'reference.h5ad', *spatial_files(name)]
    for path in paths:
        obj = read(path)
        if not obj.obs_names.is_unique or not obj.var_names.is_unique:
            raise ValueError(f'Duplicate cell/gene identifiers: {path}')
        if not np.isfinite(obj.X.data).all() or (obj.X.data < 0).any():
            raise ValueError(f'Invalid count matrix: {path}')
        if path.name == 'training.h5ad':
            if 'split' not in obj.obs:
                raise ValueError(f'No training partitions: {path}')
            source = obj.obs.get('source_sc_id', pd.Series(obj.obs_names, index=obj.obs_names))
            groups = obj.obs['split'].astype(str)
            overlap = set(source[groups == 'train']) & set(source[groups == 'val'])
            if overlap:
                raise ValueError(f'{len(overlap)} source reference cells occur in train and validation')
        records.append({'path': str(path.relative_to(ROOT)), 'cells': obj.n_obs,
                        'genes': obj.n_vars, 'counts': float(obj.X.sum())})
    output = ROOT / f'outputs/preprocessed/{name}'; output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output/'input_audit.csv', index=False)
    print(pd.DataFrame(records).to_string(index=False))


def _map(reference, spatial, name, matching_label=None):
    from .mapping import integrate_single_st_file_with_harmony, map_sc_reference_to_unique_xenium_cells
    cfg = dataset(name); settings = config()['mapping']
    common = spatial.var_names.intersection(reference.var_names, sort=False)
    if len(common) < 3:
        raise ValueError('Reference/spatial gene identifiers do not overlap')
    spatial = spatial[:, common].copy(); reference = reference[:, common].copy()
    reference.obs['cell_type'] = reference.obs[cfg['reference_label']].astype(str)
    spatial.obs['cell_type'] = spatial.obs[cfg['label']].astype(str)
    match_sc = matching_label or cfg['reference_label']
    match_st = matching_label if matching_label and matching_label in spatial.obs else cfg['label']
    valid = valid_labels(spatial.obs[match_st]) & spatial.obs[match_st].isin(reference.obs[match_sc]).to_numpy()
    spatial = spatial[valid].copy()
    reference, spatial, key = integrate_single_st_file_with_harmony(
        spatial, reference, tissue_filter=None, n_pcs=settings['n_pcs'], tissue_name='representative')
    mapped = map_sc_reference_to_unique_xenium_cells(reference, spatial, embedding_key=key,
        spatial_cols=tuple(cfg['coords']), n_matches=settings['n_matches'], reuse_penalty=settings['reuse_penalty'],
        max_reuse_per_sc=settings['max_reuse_per_sc'], match_cell_type=True, ensure_all_sc_used=False,
        sc_cell_type_col=match_sc, st_cell_type_col=match_st, rng=np.random.default_rng(settings['seed']))
    # Split source identities, never remapped rows, into train and validation.
    folds = reference.obs.groupby(cfg['reference_label'], observed=True).cumcount() % 5
    val_ids = set(reference.obs_names[folds == 4])
    mapped.obs['split'] = np.where(mapped.obs['source_sc_id'].isin(val_ids), 'val', 'train')
    mapped.obs['cell_type'] = mapped.obs[cfg['reference_label']].astype(str)
    labels = sorted(mapped.obs['cell_type'].unique())
    mapped.uns['cell_type_dict'] = {label: -(i + 1) for i, label in enumerate(labels)}
    metadata = mapped.uns.get('sc_reference_to_spatial_mapping', {})
    if 'spatial_cols' in metadata:
        metadata['spatial_cols'] = list(metadata['spatial_cols'])
    return mapped


def select_representative(name):
    from .metrics import adjacency
    cfg = dataset(name); objects = [read(path) for path in spatial_files(name)]
    order = sorted(set().union(*(set(obj.obs.loc[valid_labels(obj.obs[cfg['label']]), cfg['label']].astype(str)) for obj in objects)))
    profiles = [adjacency(obj, cfg['label'], cfg['coords'], order).ravel() for obj in objects]
    similarity = np.full((len(profiles), len(profiles)), np.nan)
    for i, left in enumerate(profiles):
        for j, right in enumerate(profiles):
            if i != j:
                similarity[i,j] = spearmanr(left, right, nan_policy='omit').statistic
    names = [path.stem for path in spatial_files(name)]
    destination = ROOT/f'outputs/preprocessed/{name}'; destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(similarity,index=names,columns=names).to_csv(destination/'adjacency_spearman.csv')
    selected = names[int(np.nanargmax(np.nanmean(similarity,axis=1)))]
    print('Selected representative:', selected)
    return selected


def rebuild(name, representative=None):
    if name == 'SIM':
        return rebuild_simulation()
    cfg = dataset(name)
    reference = read(data_dir(name)/'reference.h5ad')
    spatial = read(data_dir(name)/'spatial'/f'{representative or cfg["representative"]}.h5ad')
    mapped = _map(reference, spatial, name, cfg.get('matching_label') if name == 'BM' else None)
    write(mapped, ROOT/f'outputs/preprocessed/{name}/training.h5ad')


def rebuild_simulation():
    reference = read(data_dir('NC')/'reference.h5ad')
    reference = reference[reference.obs['cell_type_harmonized'].astype(str) != 'Treg'].copy()
    reference.obs['split'] = np.where(reference.obs['Patient'].isin(['p1']), 'test',
                                   np.where(reference.obs['Patient'].isin(['p5']), 'val', 'train'))
    parts = []
    for partition, slices in config()['simulation']['partitions'].items():
        subset = reference[reference.obs['split'] == partition].copy()
        for sl in slices:
            spatial = read(data_dir('NC')/'spatial'/f'{sl}.h5ad')
            spatial = spatial[spatial.obs['cell_type_harmonized'].astype(str) != 'Treg'].copy()
            mapped = _map(subset, spatial, 'NC')
            mapped.obs_names = [f'{sl}:{cell}' for cell in mapped.obs_names]
            mapped.obs['spatial_slice'] = sl; mapped.obs['split'] = partition
            parts.append(mapped)
    for label, partitions in [('training', ['train', 'val']), ('test', ['test'])]:
        combined = ad.concat([obj for obj in parts if obj.obs['split'].iloc[0] in partitions],
                             join='inner', uns_merge='same')
        labels = sorted(reference.obs['cell_type_harmonized'].astype(str).unique())
        combined.uns['cell_type_dict'] = {value: -(i+1) for i, value in enumerate(labels)}
        write(combined, ROOT/f'outputs/preprocessed/SIM/{label}.h5ad')

