"""Compute ovrlpy cell integrity scores from formatted transcripts."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import ovrlpy
from ovrlpy import Ovrlp, cell_integrity_from_transcripts

from common import dataset_of, output_path, save_new, transcript_path


def run(final: Path, output_root: Path | None, transcript_root: Path | None, workers: int):
    target = output_path(final, 'ovrlpy', output_root)
    if target.exists():
        raise FileExistsError(target)
    source = transcript_path(final, transcript_root)
    tx = pl.from_arrow(pq.read_table(source, columns=['cell_id', 'gene', 'x', 'y', 'z']))
    a = ad.read_h5ad(final)
    if not a.obs_names.is_unique:
        raise ValueError('Duplicate cell IDs')
    scores = []
    if dataset_of(final) == 'nsclc_cosmx':
        # CosMx global coordinates contain large gaps between fields of view.
        # Work one FOV at a time to bound the KDE raster size.
        lookup = a.obs[['fov']].copy()
        lookup['cell_id'] = a.obs_names.astype(str)
        tx = tx.join(pl.from_pandas(lookup.reset_index(drop=True)), on='cell_id', how='inner')
        groups = [g.drop('fov') for _, g in tx.partition_by('fov', as_dict=True).items()]
    else:
        groups = [tx]
    for group in groups:
        if group.height < 100 or group['gene'].n_unique() < 4:
            continue
        # CosMx uses z=-1 for some molecules. ovrlpy's KDE expects every
        # coordinate axis to be nonnegative; shifting preserves distances.
        group = group.with_columns((pl.col('z') - pl.col('z').min()).alias('z'))
        n_components = min(30, group['gene'].n_unique() - 1)
        ov = Ovrlp(group, n_components=n_components, n_workers=workers, random_state=0)
        ov.analyse(fit_umap=False)
        pixels = cell_integrity_from_transcripts(ov, cell_id='cell_id', unassigned=None)
        per_cell = pixels.group_by('cell_id').agg(pl.col('vsi').mean().alias('ovrlpy_vsi'))
        scores.append(per_cell.to_pandas())
    if not scores:
        raise ValueError('No ovrlpy scores computed')
    score = pd.concat(scores).groupby('cell_id').ovrlpy_vsi.mean()
    a.obs['ovrlpy_vsi'] = score.reindex(a.obs_names.astype(str)).to_numpy(dtype=np.float32)
    a.uns['ovrlpy_competitor'] = {'source_transcripts': str(source), 'score': 'mean pixel VSI per cell',
                                  'missing_score': 'NaN', 'workers': workers,
                                  'ovrlpy_version': ovrlpy.__version__,
                                  'layer_semantics': 'cell score broadcast over genes; not expression'}
    # A layer requires cell x gene shape. This broadcasts a cell-level diagnostic;
    # it must never be interpreted as an expression count matrix.
    a.layers['ovrlpy_vsi'] = np.broadcast_to(a.obs['ovrlpy_vsi'].to_numpy(dtype=np.float32)[:, None], a.shape).copy()
    save_new(a, target)
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('final_h5ad', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--transcript-root', type=Path)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    run(a.final_h5ad, a.output_root, a.transcript_root, a.workers)
