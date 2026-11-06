"""Shared input validation and collision-safe AnnData output."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

BASE = Path('/work/magroup/skrieger/Syndri/data/datasets')
LABELS = {'nsclc_cosmx': 'cell_type_harmonized', 'breast_xenium': 'harmony_cell_type',
          'agingbrain_merfish': 'cell_type', 'brain_merfish': 'class'}


def dataset_of(path: Path) -> str:
    parent = path.resolve().parent
    for name in LABELS:
        dataset_root = BASE / name
        if parent in (
            dataset_root / 'results/final',
            dataset_root / 'interim/spatial',
        ):
            return name
    raise ValueError(f'Expected a supported final or interim spatial AnnData: {path}')


def output_path(path: Path, method: str, root: Path | None = None,
                dataset: str | None = None) -> Path:
    dataset = dataset or dataset_of(path)
    return (root or BASE / dataset / 'results/competitors') / method / path.name


def transcript_path(path: Path, root: Path | None = None) -> Path:
    dataset = dataset_of(path)
    return (root or BASE / dataset / 'raw/competitors/transcripts') / (path.stem + '.parquet')


def validate_counts(adata: ad.AnnData) -> None:
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('AnnData must have unique cell and gene names')
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.allclose(values, np.round(values)):
        raise ValueError('Input X must contain raw integer counts')


def save_new(adata: ad.AnnData, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{target.stem}.', suffix='.h5ad', dir=target.parent)
    os.close(fd)
    try:
        if hasattr(ad.settings, 'allow_write_nullable_strings'):
            ad.settings.allow_write_nullable_strings = True
        adata.write_h5ad(tmp, compression='gzip')
        os.link(tmp, target)  # atomic and fails if another job wrote the result
    finally:
        os.unlink(tmp)
