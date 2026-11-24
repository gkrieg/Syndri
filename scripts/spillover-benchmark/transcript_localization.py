"""Cached boundary-localization summaries for saved SYNDRI-SIM transcripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'recomb_reproducibility'))
from syndri_repro.transcript_locations import MaskGeometry


def signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def boundary_distance(points_um, region, pixel_size_um):
    """Exact distance to exposed raster-pixel edges, including interior points.

    The location simulator's region-distance function returns zero for interior
    points; here the target is the boundary itself, as in Figure 1.
    """
    result = np.empty(len(points_um), dtype=float)
    pending = np.arange(len(points_um))
    k = min(8, len(region.edge_centers))
    while len(pending):
        distances, indices = region.tree.query(points_um[pending], k=k)
        distances = np.asarray(distances).reshape(len(pending), k)
        indices = np.asarray(indices).reshape(len(pending), k)
        delta = np.maximum(np.abs(points_um[pending, None, :] - region.edge_centers[indices])
                           - region.edge_halfwidths[indices], 0.)
        best = np.sqrt(np.square(delta).sum(axis=2)).min(axis=1)
        done = ((distances[:, -1] - pixel_size_um/2 >= best-1e-12)
                | (k == len(region.edge_centers)))
        result[pending[done]] = best[done]
        pending = pending[~done]
        k = min(2*k, len(region.edge_centers))
    return result


def slice_localization(benchmark_root, output_dir, slice_name, n_cells=1000,
                       seed=20261121, cell_types=None, recompute=False):
    """Stream only four CSV columns; retain per-cell sums rather than all transcripts."""
    benchmark_root, output_dir = Path(benchmark_root), Path(output_dir)
    folder = benchmark_root/'transcript_coordinates'/slice_name/'all_fovs'
    csv_path, count_path = folder/f'{slice_name}_tx_file.csv', folder/'simulated_counts.h5ad'
    provenance = json.loads((folder/'simulation.json').read_text())
    validation = json.loads((folder/'validation.json').read_text())
    if not provenance['validation_passed'] or not all(validation['checks'].values()):
        raise ValueError(f'{slice_name}: coordinate generation has not passed validation')
    data_root = Path(json.loads((benchmark_root/'parameters.json').read_text())['data_root'])
    raw = data_root/'raw/spatial'/slice_name/f'{slice_name}-Flat_files_and_images'
    settings = dict(slice=slice_name, n_cells=n_cells, seed=seed,
                    cell_types=sorted(cell_types) if cell_types is not None else None,
                    cell_selection='uniform without replacement among cells with both populations',
                    csv=signature(csv_path), counts=signature(count_path),
                    provenance=signature(folder/'simulation.json'),
                    positions=signature(raw/f'{slice_name}_fov_positions_file.csv'),
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    cache = output_dir/'cache'/f'{slice_name}_{key}'
    table_path = cache/'cell_distance_summaries.parquet'
    if table_path.exists() and (cache/'complete.json').exists() and not recompute:
        print(f'{slice_name}: reusing cached localization summaries', flush=True)
        return pd.read_parquet(table_path)
    with h5py.File(count_path) as handle:
        obs = ad.io.read_elem(handle['obs'])
    eligible = obs.loc[(obs.native_counts > 0) & (obs.added_counts > 0)].copy()
    if cell_types is not None:
        eligible = eligible.loc[eligible.cell_type.astype(str).isin(cell_types)]
    if eligible.empty:
        raise ValueError(f'{slice_name}: no eligible cells for the requested cell-type filter')
    if n_cells is not None and n_cells <= 0:
        raise ValueError('N_CELLS_PER_SLICE must be positive or None for all eligible cells')
    slice_seed = int.from_bytes(hashlib.sha256(slice_name.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, slice_seed]))
    size = len(eligible) if n_cells is None else min(n_cells, len(eligible))
    sampled = eligible.iloc[np.sort(rng.choice(len(eligible), size, replace=False))].copy()
    sampled.index = sampled.index.astype(str)
    cache.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(cache/'sampled_cells.csv', index_label='assigned_cell')
    geometry = MaskGeometry(raw, slice_name,
                            benchmark_root/'transcript_coordinates/cache'/slice_name,
                            pixel_size_um=provenance['pixel_size_um'])
    geometry.flip_y = {int(row['fov']): bool(row['flip_image_rows'])
                       for row in provenance['fov_calibration']}
    sums = np.zeros((len(sampled), 2), dtype=np.float64)
    counts = np.zeros((len(sampled), 2), dtype=np.int64)
    radii = np.zeros(len(sampled))
    ids = pd.Index(sampled.index)
    reader = pacsv.open_csv(csv_path,
        read_options=pacsv.ReadOptions(block_size=32*1024**2, use_threads=False),
        convert_options=pacsv.ConvertOptions(include_columns=[
            'assigned_cell', 'is_admixed', 'x_global_px', 'y_global_px']))
    print(f'{slice_name}: scanning CSV for {len(sampled):,} sampled cells...', flush=True)
    selected_transcripts = 0
    for batch in reader:
        frame = batch.to_pandas()
        rows = ids.get_indexer(frame.assigned_cell)
        keep = rows >= 0
        if not keep.any():
            continue
        frame = frame.loc[keep].copy()
        rows = rows[keep]
        frame['_sample_row'] = rows
        for row, positions in frame.groupby('_sample_row', sort=False).indices.items():
            meta = sampled.iloc[int(row)]
            region = geometry.region(int(meta.fov), int(meta.cell_ID))
            xy = frame.iloc[positions][['x_global_px', 'y_global_px']].to_numpy(float)
            local = xy - geometry.origin(int(meta.fov))
            if not geometry.contains(int(meta.fov), int(meta.cell_ID), local).all():
                raise AssertionError(f'{slice_name}: transcript outside sampled recipient mask')
            distance = boundary_distance(xy*geometry.scale, region, geometry.scale)
            kind = frame.iloc[positions].is_admixed.to_numpy(dtype=bool).astype(np.int64)
            counts[row] += np.bincount(kind, minlength=2)
            sums[row] += np.bincount(kind, weights=distance, minlength=2)
            radii[row] = np.sqrt(len(region.rows)*geometry.scale**2/np.pi)
        selected_transcripts += len(frame)
    expected = sampled[['native_counts', 'added_counts']].to_numpy(np.int64)
    if not np.array_equal(counts, expected):
        raise AssertionError(f'{slice_name}: sampled transcript totals do not match saved native/added counts')
    if (radii <= 0).any() or (counts <= 0).any():
        raise AssertionError('Missing geometry or a missing transcript population')
    result = pd.DataFrame(dict(
        slice=np.repeat(slice_name, 2*len(sampled)),
        assigned_cell=np.repeat(ids.to_numpy(), 2),
        cell_type=np.repeat(sampled.cell_type.astype(str).to_numpy(), 2),
        fov=np.repeat(sampled.fov.to_numpy(), 2),
        cell_ID=np.repeat(sampled.cell_ID.to_numpy(), 2),
        transcript_type=np.tile(['Native', 'Admixed'], len(sampled)),
        n_transcripts=counts.ravel(), equivalent_radius_um=np.repeat(radii, 2),
        mean_distance_um=(sums/counts).ravel(),
        mean_normalized_distance=(sums/counts/radii[:, None]).ravel()))
    result.to_parquet(table_path, index=False)
    (cache/'complete.json').write_text(json.dumps(dict(settings=settings,
        n_eligible_cells=len(eligible), n_sampled_cells=len(sampled),
        n_transcripts=selected_transcripts, sampled_count_reconstruction_exact=True), indent=2)+'\n')
    print(f'{slice_name}: summarized {selected_transcripts:,} transcripts in {len(sampled):,} paired cells', flush=True)
    return result
