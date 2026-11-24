#!/usr/bin/env python3
"""Disk-backed cellAdmix annotation: exact global KNN queries, compact cell pairs.

No transcript subsampling or FOV isolation. Only the cell-pair sufficient
information consumed by get_adj_pairs is retained, not a molecule graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def signature(path):
    p = Path(path).resolve()
    s = p.stat()
    return [str(p), s.st_size, s.st_mtime_ns]


def stage(transcripts, assignments, metadata, work, k, chunk_rows=250000):
    """Read paired CSVs once; sort by cell then original molecule ID on disk."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    spec = dict(schema=1, transcripts=signature(transcripts),
                assignments=signature(assignments), metadata=signature(metadata), factors=k)
    complete = work/'staged.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['specification'] != spec:
            raise ValueError('Annotation staging input changed')
        return saved
    meta = pd.read_csv(metadata, dtype={'cell': str, 'celltype': str}).sort_values('cell')
    if meta.cell.duplicated().any():
        raise ValueError('Nonunique metadata cells')
    ids = pd.Index(meta.cell)
    # The exact original CSV row count is recorded by preparation.
    ready = json.loads((Path(transcripts).parent/'ready.json').read_text())
    maximum = int(ready['n_transcripts'])
    xyz = np.memmap(work/'unsorted_xyz.dat', mode='w+', dtype='<f8', shape=(maximum, 3))
    cells = np.memmap(work/'unsorted_cells.dat', mode='w+', dtype='<i4', shape=(maximum,))
    factors = np.memmap(work/'unsorted_factors.dat', mode='w+', dtype='<i4', shape=(maximum,))
    counts = np.zeros((len(meta), k), np.int64)
    sums = np.zeros((len(meta), 3), np.float64)
    source = pd.read_csv(transcripts, chunksize=chunk_rows,
                         dtype={'cell_id': str, 'cell_type': str, 'gene': str})
    assignment = pd.read_csv(assignments, chunksize=chunk_rows,
                             dtype={'cell_id': str, 'cell_type': str, 'gene': str})
    total = valid = 0
    for tx in source:
        ass = next(assignment, None)
        if ass is None or len(tx) != len(ass):
            raise ValueError('Transcript/assignment lengths differ')
        for col in ['mol_id', 'cell_id', 'cell_type', 'gene']:
            if not np.array_equal(tx[col].to_numpy(), ass[col].to_numpy()):
                raise ValueError(f'Transcript/assignment mismatch: {col}')
        if not np.array_equal(tx.mol_id, np.arange(total, total+len(tx))):
            raise ValueError('Molecule order differs from original preparation')
        keep = ass.factor.notna().to_numpy()
        code = ids.get_indexer(tx.cell_id.to_numpy()[keep])
        f = ass.factor.to_numpy()[keep].astype(np.int32)
        coords = tx[['x', 'y', 'z']].to_numpy(np.float64)[keep]
        if (code < 0).any() or not np.isfinite(coords).all() or ((f < 1) | (f > k)).any():
            raise ValueError('Invalid cell, coordinate, or factor')
        if not np.array_equal(tx.cell_type.to_numpy()[keep], meta.celltype.to_numpy()[code]):
            raise ValueError('Cell-type mismatch')
        end = valid+len(code)
        xyz[valid:end], cells[valid:end], factors[valid:end] = coords, code, f
        np.add.at(counts, (code, f-1), 1)
        for axis in range(3):
            sums[:, axis] += np.bincount(code, weights=coords[:, axis], minlength=len(meta))
        valid = end
        total += len(tx)
        if total//5000000 != (total-len(tx))//5000000:
            print(f'STAGE {total:,}/{maximum:,} molecules', flush=True)
    if next(assignment, None) is not None or total != maximum:
        raise ValueError('Unconsumed/missing molecules')
    if not (counts.sum(0) > 0).all():
        raise ValueError('Original annotation assumes contiguous globally observed factors 1..k')
    active = counts.sum(1) > 0
    remap = np.cumsum(active)-1
    # Stable sorting preserves the original per-cell mol_id order.
    order = np.argsort(cells[:valid], kind='stable')
    final_xyz = np.memmap(work/'xyz.dat', mode='w+', dtype='<f8', shape=(valid, 3))
    final_f = np.memmap(work/'factors.dat', mode='w+', dtype='<i4', shape=(valid,))
    final_c = np.memmap(work/'cells.dat', mode='w+', dtype='<i4', shape=(valid,))
    for start in range(0, valid, chunk_rows):
        ix = order[start:start+chunk_rows]
        final_xyz[start:start+len(ix)] = xyz[ix]
        final_f[start:start+len(ix)] = factors[ix]
        final_c[start:start+len(ix)] = remap[cells[ix]]
    for a in (final_xyz, final_f, final_c):
        a.flush()
    sizes = counts.sum(1)[active]
    meta = meta.loc[active].copy()
    meta['start'] = np.r_[0, np.cumsum(sizes)[:-1]]
    meta['n'] = sizes
    for axis, name in enumerate(['x', 'y', 'z']):
        meta[name] = sums[active, axis]/sizes
    meta.to_csv(work/'cells.csv', index=False, float_format='%.17g')
    pd.DataFrame(counts[active], columns=[str(i) for i in range(1, k+1)]).to_csv(
        work/'factor_counts.csv', index=False)
    # Release mappings before removing temporary staging files.
    del xyz, cells, factors, order, final_xyz, final_f, final_c
    for name in ('unsorted_xyz.dat', 'unsorted_cells.dat', 'unsorted_factors.dat'):
        (work/name).unlink()
    saved = dict(specification=spec, n_transcripts=valid, n_input_transcripts=total,
                 n_excluded_unassigned=total-valid, n_cells=len(meta), factors=k)
    atomic_json(complete, saved)
    return saved


def reduce_neighbors(query_cells, neighbors, cell_types, first, exposure, offset):
    """Reproduce get_adj_pairs' first molecule and lexical neighbor choices."""
    n_types = first.shape[1]
    neighbor_types = cell_types[neighbors]
    query_types = cell_types[query_cells]
    sentinel = len(cell_types)
    for d in range(n_types):
        selected = neighbor_types == d
        exposed = selected.any(1)
        exposure[:, d] += np.bincount(query_types[exposed], minlength=n_types)
        same = query_types == d
        # Original same-type logic requires >1 DISTINCT neighbor cells,
        # then removes the query's own cell and picks the first lexical cell.
        least = np.where(selected, neighbors, sentinel).min(1)
        greatest = np.where(selected, neighbors, -1).max(1)
        allowed = exposed & (~same | (least != greatest))
        eligible = selected & (~same[:, None] | (neighbors != query_cells[:, None]))
        chosen = np.where(eligible, neighbors, sentinel).min(1)
        valid = np.flatnonzero(allowed & (chosen != sentinel))
        if len(valid):
            cells, ix = np.unique(query_cells[valid], return_index=True)
            new = first[cells, d] < 0
            first[cells[new], d] = chosen[valid[ix[new]]]


def make_pairs(meta, first, exposure):
    types = sorted(meta.celltype.unique())
    codes = pd.Index(types).get_indexer(meta.celltype)
    supported = [i for i in range(len(types)) if np.count_nonzero(codes == i) > 1]
    combinations = [(i, i) for i in supported] + [(i, j) for i in supported for j in supported if i != j]
    pairs, table = [], []
    for index, (i, j) in enumerate(combinations, 1):
        q = np.flatnonzero((codes == i) & (first[:, j] >= 0))
        d = first[q, j]
        if exposure[i, j] < 5:
            q, d = q[:0], d[:0]
        if len(d):
            keep = ~pd.Index(d).duplicated()
            q, d = q[keep], d[keep]
            if i == j:
                keep = ~np.isin(d, q)
                q, d = q[keep], d[keep]
        pairs.extend(dict(pair_index=index, self_cells=meta.cell.iloc[a], adj_cells=meta.cell.iloc[b])
                     for a, b in zip(q, d))
        table.append(dict(pair_index=index, ct1=types[i], ct2=types[j],
                          name=types[i]+'_'+types[j], n_pairs=len(q)))
    return pd.DataFrame(pairs, columns=['pair_index', 'self_cells', 'adj_cells']), pd.DataFrame(table)


def global_pairs(work, workers=1, chunk_rows=100000):
    work = Path(work)
    info = json.loads((work/'staged.json').read_text())
    if (work/'neighbors_complete.json').exists():
        return
    n = info['n_transcripts']
    meta = pd.read_csv(work/'cells.csv', dtype={'cell': str, 'celltype': str})
    types = sorted(meta.celltype.unique())
    cell_types = pd.Index(types).get_indexer(meta.celltype)
    xyz = np.memmap(work/'xyz.dat', mode='r', dtype='<f8', shape=(n, 3))
    cell = np.memmap(work/'cells.dat', mode='r', dtype='<i4', shape=(n,))
    print(f'GLOBAL KNN: building exact 3D index for {n:,} transcripts', flush=True)
    tree = cKDTree(xyz, copy_data=False)
    first = np.full((len(meta), len(types)), -1, np.int32)
    exposure = np.zeros((len(types), len(types)), np.int64)
    cursor = 0
    cache = work/'neighbors_progress.npz'
    if cache.exists():
        with np.load(cache) as old:
            first[:], exposure[:], cursor = old['first'], old['exposure'], int(old['cursor'])
    began = time.monotonic()
    for start in range(cursor, n, chunk_rows):
        stop = min(n, start+chunk_rows)
        # Exact query against ALL transcripts; remove the query itself, not
        # the first returned point (important when coordinates are tied).
        _, indices = tree.query(xyz[start:stop], k=11, eps=0, workers=workers)
        query = np.arange(start, stop)
        not_self = indices != query[:, None]
        rank = np.cumsum(not_self, axis=1)
        keep = not_self & (rank <= 10)
        indices = indices[keep].reshape(stop-start, 10)
        reduce_neighbors(cell[start:stop], cell[indices], cell_types, first, exposure, start)
        if stop % (10*chunk_rows) == 0 or stop == n:
            with (work/'neighbors_progress.tmp').open('wb') as f:
                np.savez(f, first=first, exposure=exposure, cursor=stop)
            (work/'neighbors_progress.tmp').replace(cache)
            atomic_json(work/'progress.json', dict(stage='global_neighbors', done=stop, total=n,
                                                   elapsed_seconds=time.monotonic()-began))
            print(f'GLOBAL KNN {stop:,}/{n:,} ({100*stop/n:.1f}%)', flush=True)
    pairs, plan = make_pairs(meta, first, exposure)
    pairs.to_csv(work/'pairs.csv', index=False)
    plan.to_csv(work/'pair_plan.csv', index=False)
    atomic_json(work/'neighbors_complete.json', dict(n_transcripts=n, k=10,
        exact_global_queries=True, spatial_dimension=3, graph_materialized=False,
        pair_tests=len(plan), cell_pairs=len(pairs)))


def run_annotation(transcripts, assignments, metadata, factors, output, workers=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    work = output/'bounded_cache'
    work.mkdir(parents=True, exist_ok=True)
    recipe = dict(schema=1, global_knn=10, bridge_knn=20, bridge_sample_cells=200,
                  seed=0, false_positive_neighbors=100, p_threshold=.1,
                  transcript_downsampling=False, bridge_workers=1,
                  python_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  r_sha256=hashlib.sha256((HERE/'celladmix_annotation_bounded.R').read_bytes()).hexdigest())
    recipe_path = work/'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Annotation implementation changed; preserve old cache and use a fresh output directory')
    atomic_json(recipe_path, recipe)
    stage(transcripts, assignments, metadata, work, factors)
    global_pairs(work, workers=workers)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(['/home/jwinchel/miniconda3/envs/r_env/bin/Rscript',
                    str(HERE/'celladmix_annotation_bounded.R'), str(work), str(output), str(factors)],
                   env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', type=Path, required=True)
    parser.add_argument('--assignments', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--factors', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    run_annotation(args.transcripts, args.assignments, args.metadata,
                   args.factors, args.output, args.workers)
