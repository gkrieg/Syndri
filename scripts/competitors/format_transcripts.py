"""Normalize raw CosMx/Xenium molecules and join final AnnData annotations."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path('/work/magroup/skrieger/Syndri/data/datasets')
LABELS = {'nsclc_cosmx': 'cell_type_harmonized', 'breast_xenium': 'harmony_cell_type'}


def paths(dataset: str, final: Path, output_root: Path | None):
    root = BASE / dataset
    out = (output_root or root / 'raw/competitors/transcripts') / (final.stem + '.parquet')
    if dataset == 'nsclc_cosmx':
        sample = final.stem.removeprefix('st_integrated_slice-')
        source = root / 'raw/spatial' / sample / f'{sample}-Flat_files_and_images' / f'{sample}_tx_file.csv'
    else:
        sample = final.stem.removesuffix('_rctd_annotated')
        source = root / 'raw/spatial' / f'{sample}_transcripts.parquet'
    return source, out


def format_one(dataset: str, final: Path, output_root: Path | None = None):
    if dataset not in LABELS:
        raise ValueError(f'Unsupported dataset {dataset}')
    source, out = paths(dataset, final, output_root)
    if out.exists():
        raise FileExistsError(out)
    if not source.is_file():
        raise FileNotFoundError(source)
    a = ad.read_h5ad(final, backed='r')
    labels = a.obs[LABELS[dataset]].astype('string')
    if labels.isna().any():
        raise ValueError('Final AnnData has missing cell type labels')
    obs = pd.DataFrame({'cell_id': a.obs_names.astype(str), 'cell_type': labels.to_numpy()})
    if obs.cell_id.duplicated().any():
        raise ValueError('Duplicate final cell IDs')
    gene_col = 'gene_symbol' if dataset == 'nsclc_cosmx' else 'feature_name'
    gene_map = pd.Series(a.var_names.astype(str).to_numpy(), index=a.var[gene_col].astype(str).to_numpy())
    gene_map = gene_map.loc[~gene_map.index.isin(['N/A', 'nan', ''])]
    if gene_map.index.duplicated().any():
        raise ValueError(f'Duplicate {gene_col} symbols in final AnnData')
    if dataset == 'nsclc_cosmx':
        obs['fov'] = a.obs['fov'].to_numpy().astype(str)
        obs['raw_cell_id'] = a.obs['cell_ID'].to_numpy().astype(str)
        if obs[['fov', 'raw_cell_id']].duplicated().any():
            raise ValueError('Ambiguous CosMx (fov, cell_ID) keys')
        tx = pd.read_csv(source, usecols=['fov', 'cell_ID', 'x_global_px', 'y_global_px', 'z', 'target'],
                         dtype={'fov': str, 'cell_ID': str, 'target': str})
        tx = tx.rename(columns={'cell_ID': 'raw_cell_id', 'target': 'gene'})
        tx = tx.merge(obs, on=['fov', 'raw_cell_id'], how='inner', validate='many_to_one')
        tx['x'] = pd.to_numeric(tx.pop('x_global_px')) * 0.12028
        tx['y'] = pd.to_numeric(tx.pop('y_global_px')) * 0.12028
        tx['z'] = pd.to_numeric(tx['z']) * 0.8
    else:
        obs['raw_cell_id'] = a.obs['cell_id'].astype(str).to_numpy()
        if obs.raw_cell_id.duplicated().any():
            raise ValueError('Ambiguous Xenium cell_id keys')
        tx = pq.read_table(source, columns=['cell_id', 'feature_name', 'x_location',
                                           'y_location', 'z_location', 'qv']).to_pandas()
        tx = tx.rename(columns={'cell_id': 'raw_cell_id', 'feature_name': 'gene', 'x_location': 'x',
                                'y_location': 'y', 'z_location': 'z'})
        tx = tx.loc[tx['qv'] >= 20].drop(columns='qv')
        tx = tx.merge(obs, on='raw_cell_id', how='inner', validate='many_to_one')
    tx['gene'] = tx['gene'].map(gene_map)
    tx = tx.dropna(subset=['gene'])[['cell_id', 'cell_type', 'gene', 'x', 'y', 'z']]
    tx = tx.loc[np.isfinite(tx[['x', 'y', 'z']].to_numpy()).all(axis=1)].copy()
    if tx.empty:
        raise ValueError('No transcripts match the final AnnData')
    tx.insert(0, 'mol_id', np.arange(len(tx), dtype=np.int64).astype(str))
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{out.stem}.', suffix='.parquet', dir=out.parent)
    os.close(fd)
    try:
        tx.to_parquet(tmp, index=False)
        os.link(tmp, out)  # exclusive atomic publish
    finally:
        os.unlink(tmp)
    print(f'{len(tx):,} molecules -> {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=LABELS)
    p.add_argument('final_h5ad', type=Path)
    p.add_argument('--output-root', type=Path)
    args = p.parse_args()
    format_one(args.dataset, args.final_h5ad, args.output_root)
