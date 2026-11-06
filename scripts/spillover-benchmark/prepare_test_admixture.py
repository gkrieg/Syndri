#!/usr/bin/env python3
"""Make per-slice test inputs with synthetic admixture and a unique SC reference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

ROOT = Path('/home/jwinchel/Syndri')
sys.path.insert(0, str(ROOT / 'Heimdall'))
from Heimdall.admixture_sampling import build_neighbor_dict, sample_admixture_from_neighbor_dict


def csr(x):
    return x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--sc-reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reference-output', type=Path)
    p.add_argument('--seed', type=int, default=20261103)
    a = p.parse_args()

    test = ad.read_h5ad(a.test)
    if 'spatial_slice' not in test.obs or 'cell_type' not in test.obs:
        raise KeyError("test.h5ad needs obs columns 'spatial_slice' and 'cell_type'")
    test = test[test.obs['cell_type'].astype(str) != 'Treg'].copy()
    if not test.obs_names.is_unique:
        raise ValueError('test.h5ad must have unique observation names')

    # Recreate the train+validation source-cell pool directly from the original
    # SC AnnData. This intentionally avoids repeated rows from spatial remapping.
    ref = ad.read_h5ad(a.sc_reference)
    required = {'Patient', 'cell_type_harmonized'}
    missing = required - set(ref.obs)
    if missing:
        raise KeyError(f'SC reference is missing obs columns: {sorted(missing)}')
    ref = ref[(~ref.obs['Patient'].astype(str).isin(['p1'])) &
              (ref.obs['cell_type_harmonized'].astype(str) != 'Treg')].copy()
    ref.obs['cell_type'] = ref.obs['cell_type_harmonized'].astype(str)
    if not ref.obs_names.is_unique:
        raise ValueError('Original SC reference observation names are not unique')

    a.output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = a.reference_output or (a.output_dir.parent / 'train_val_sc_reference_unique.h5ad')
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref.write_h5ad(ref_path, compression='gzip')

    for i, slice_name in enumerate(sorted(test.obs['spatial_slice'].astype(str).unique())):
        part = test[test.obs['spatial_slice'].astype(str) == slice_name].copy()
        if part.n_obs == 0:
            continue
        raw = csr(part.X).astype(np.int32)
        part.X = raw
        part = build_neighbor_dict(
            part,
            k_neighbors=9,
            coord_cols=('CenterX_global_px', 'CenterY_global_px'),
            tissue_col='spatial_slice',
            cell_type_col='cell_type',
            min_source_counts=400,
            use_same_cell_type_neighbors=False,
        )
        added, _ = sample_admixture_from_neighbor_dict(
            part,
            tau_range=(0.0, 0.95),
            n_repeats=1,
            min_sampled_transcript_threshold=10,
            only_sample_unexpressed_genes=True,
            rng=np.random.default_rng(a.seed + i),
            return_sparse=True,
        )
        added = csr(added).astype(np.int32)
        raw_total = np.asarray(raw.sum(axis=1)).ravel().astype(np.float64)
        added_total = np.asarray(added.sum(axis=1)).ravel().astype(np.float64)
        part.layers['original_counts'] = raw
        part.layers['synthetic_admixture_counts'] = added
        part.layers['admixed_counts'] = raw + added
        part.obs['achieved_admixture_burden'] = np.divide(
            added_total, raw_total + added_total,
            out=np.zeros_like(added_total), where=(raw_total + added_total) > 0,
        )
        part.X = part.layers['admixed_counts'].copy()
        part.uns['synthetic_admixture_config'] = {
            'k_neighbors': 9, 'tau_range': [0.0, 0.95],
            'min_source_counts': 400, 'min_sampled_transcript_threshold': 10,
            'only_sample_unexpressed_genes': True,
            'use_same_cell_type_neighbors': False,
            'seed': int(a.seed + i), 'reference_is_unique_original_sc': True,
        }
        # Neighbors are only needed to sample the synthetic contamination.
        # Avoid carrying the large ID-to-ID mapping into method inputs.
        part.uns.pop('neighbor_dict', None)
        part.uns.pop('neighbor_dict_config', None)
        out = a.output_dir / f'{slice_name}.h5ad'
        part.write_h5ad(out, compression='gzip')
        print(f'{out}: cells={part.n_obs:,}, genes={part.n_vars:,}, '
              f'mean achieved burden={part.obs["achieved_admixture_burden"].mean():.4f}', flush=True)

    print(f'Unique train/val SC reference: {ref_path}', flush=True)


if __name__ == '__main__':
    main()
