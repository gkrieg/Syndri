#!/usr/bin/env python3
"""Measure saved SYNDRI-SIM test data, unique SC reference, and matched real ST.

Reads existing matrices only. No admixture or transcript locations are generated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '8')

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Heimdall'))
sys.path.insert(0, str(ROOT / 'recomb_reproducibility'))
from Heimdall.admixture_sampling import build_neighbor_dict
from syndri_repro.transcript_locations import TEST_SLICES, integer_counts, _same_counts

REFERENCE = 'scRNA-seq reference'
CLEAN = 'Mapped scRNA-seq (clean)'
SIMULATED = 'Simulated ST'
REAL = 'Real CosMx'
DATASETS = [REFERENCE, CLEAN, SIMULATED, REAL]
METRICS = ['detected_genes', 'non_native_marker_genes',
           'achieved_admixture_fraction', 'eligible_genes_per_query']
TYPE_ORDER = ['T CD4+', 'T CD8+', 'NK', 'B cells', 'Plasmablast', 'Monocyte',
              'Macrophage', 'DC', 'Neutrophil', 'Mast', 'Endothelial',
              'Fibroblast', 'Epithelial', 'Malignant']


def read_metadata(path):
    """Read only useful metadata, without loading saved correction layers."""
    columns = ['cell_type', 'cell_type_harmonized', 'spatial_slice', 'split',
               'matched_st_id', 'source_sc_id', 'CenterX_global_px', 'CenterY_global_px',
               'achieved_admixture_burden']
    with h5py.File(path, 'r') as handle:
        group = handle['obs']
        index = pd.Index(ad.io.read_elem(group[group.attrs['_index']])).astype(str)
        obs = pd.DataFrame(index=index)
        for column in columns:
            if column in group:
                obs[column] = ad.io.read_elem(group[column])
        var = ad.io.read_elem(handle['var'])
        config = (ad.io.read_elem(handle['uns/synthetic_admixture_config'])
                  if 'synthetic_admixture_config' in handle['uns'] else None)
    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError(f'Nonunique observation or gene identifiers: {path}')
    return obs, var, config


def read_counts(path, key, genes, rows=None):
    with h5py.File(path, 'r') as handle:
        var = ad.io.read_elem(handle['var'])
        positions = var.index.get_indexer(genes)
        if (positions < 0).any():
            raise ValueError(f'Missing diagnostic genes: {path}')
        counts = integer_counts(ad.io.read_elem(handle[key]))
    if rows is not None:
        counts = counts[rows]
    return counts[:, positions].tocsr()


def select_markers(path, genes, types, aliases, n_markers, max_padj, unique_only=False):
    table = pd.read_csv(path).dropna(subset=['cell_type', 'gene', 'padj', 'log2FoldChange'])
    if not set(types).issubset(set(table.cell_type)):
        raise ValueError(f'Cell types absent from the DESeq2 table: {set(types) - set(table.cell_type)}')
    table = table.loc[table.gene.isin(genes) & table.padj.le(max_padj) & table.log2FoldChange.gt(0)]
    # Determine uniqueness across the full supplied DESeq2 table before
    # restricting to evaluated cell types or truncating a ranked marker set.
    n_types = table.groupby('gene').cell_type.nunique()
    table = table.assign(n_significant_cell_types=table.gene.map(n_types))
    if unique_only:
        table = table.loc[table.n_significant_cell_types.eq(1)]
    table = table.sort_values(['cell_type', 'padj', 'log2FoldChange'],
                              ascending=[True, True, False], kind='stable')
    selected = []
    for cell_type in types:
        part = table.loc[table.cell_type.eq(cell_type)].drop_duplicates('gene')
        if n_markers is not None:
            part = part.head(n_markers)
        part = part.copy()
        if n_markers is not None and len(part) != n_markers:
            raise ValueError(f'{cell_type}: only {len(part)} positive significant shared markers; '
                             f'{n_markers} requested')
        part['rank'] = np.arange(1, len(part) + 1)
        part['gene_symbol'] = part.gene.map(aliases)
        selected.append(part)
    result = pd.concat(selected, ignore_index=True)
    if unique_only and not result.gene.is_unique:
        raise AssertionError('Exclusive marker selection contains a shared gene')
    return result


def expression_diagnostics(counts, obs, dataset, slice_name, genes, markers, label_column):
    labels = obs[label_column].astype(str)
    detected = counts > 0
    universe = set(markers.gene)
    non_native, native, n_candidates = (np.zeros(len(obs), dtype=np.int32) for _ in range(3))
    for cell_type in labels.unique():
        rows = np.flatnonzero(labels.to_numpy() == cell_type)
        own = set(markers.loc[markers.cell_type.eq(cell_type), 'gene'])
        # Shared markers of the query's own type are never counted as non-native.
        other_indices = genes.get_indexer(sorted(universe - own))
        own_indices = genes.get_indexer(sorted(own))
        non_native[rows] = np.asarray(detected[rows][:, other_indices].sum(axis=1)).ravel()
        native[rows] = np.asarray(detected[rows][:, own_indices].sum(axis=1)).ravel()
        n_candidates[rows] = len(other_indices)
    result = pd.DataFrame(dict(dataset=dataset, slice=slice_name, cell_id=obs.index.astype(str),
        cell_type=labels.to_numpy(), total_counts=np.asarray(counts.sum(axis=1)).ravel(),
        detected_genes=detected.getnnz(axis=1), non_native_marker_genes=non_native,
        native_marker_genes=native, non_native_marker_candidates=n_candidates))
    if 'source_sc_id' in obs:
        result['source_sc_id'] = obs.source_sc_id.astype(str).to_numpy()
    result['achieved_admixture_fraction'] = np.nan
    result['eligible_genes_per_query'] = np.nan
    return result


def eligibility(native, added, obs, config, batch_size):
    """Replay original neighbor selection, then count eligible genes in batches."""
    if config is None:
        raise ValueError('Missing saved synthetic_admixture_config')
    base = ad.AnnData(native, obs=obs[['cell_type', 'spatial_slice',
        'CenterX_global_px', 'CenterY_global_px']].copy())
    base = build_neighbor_dict(base, k_neighbors=int(config['k_neighbors']),
        coord_cols=('CenterX_global_px', 'CenterY_global_px'), tissue_col='spatial_slice',
        cell_type_col='cell_type', min_source_counts=int(config['min_source_counts']),
        use_same_cell_type_neighbors=bool(config['use_same_cell_type_neighbors']))
    neighbors = base.uns['neighbor_dict']
    ids = obs.index.astype(str)
    lengths = np.array([len(neighbors.get(key, [])) for key in ids], dtype=np.int32)
    flat = [source for key in ids for source in neighbors.get(key, [])]
    columns = ids.get_indexer(flat)
    if (columns < 0).any():
        raise AssertionError('Unknown neighbor cell')
    graph = sparse.csr_matrix((np.ones(len(columns), dtype=np.int64), columns,
        np.r_[0, np.cumsum(lengths)]), shape=(len(obs), len(obs)))
    del base, neighbors, flat
    eligible_genes = np.empty(len(obs), dtype=np.int32)
    eligible_counts = np.empty(len(obs), dtype=np.int64)
    only_unexpressed = bool(config['only_sample_unexpressed_genes'])
    for start in range(0, len(obs), batch_size):
        stop = min(start + batch_size, len(obs))
        pool = (graph[start:stop] @ native).toarray()
        query = native[start:stop].toarray()
        injected = added[start:stop].toarray()
        mask = pool > 0
        if only_unexpressed:
            mask &= query == 0
        if np.any((injected > 0) & ~mask):
            raise AssertionError('Saved admixture contains an ineligible gene')
        if not np.array_equal(injected[injected > 0], pool[injected > 0]):
            raise AssertionError('Original neighbor pools do not reconstruct saved whole-gene additions')
        eligible_genes[start:stop] = mask.sum(axis=1)
        eligible_counts[start:stop] = np.where(mask, pool, 0).sum(axis=1)
    selected = added.getnnz(axis=1)
    if (selected > eligible_genes).any():
        raise AssertionError('More added than eligible genes')
    return dict(eligible_genes_per_query=eligible_genes,
                eligible_neighbor_transcripts=eligible_counts,
                selected_neighbor_count=lengths,
                added_genes_per_query=selected,
                eligible_genes_remaining=eligible_genes-selected,
                fraction_eligible_genes_selected=np.divide(selected, eligible_genes,
                    out=np.full(len(obs), np.nan), where=eligible_genes > 0))


def summarize(cells):
    records = []
    for grouping in [('dataset',), ('dataset', 'slice'), ('dataset', 'slice', 'cell_type')]:
        for values, part in cells.groupby(list(grouping), observed=True, sort=False):
            values = values if isinstance(values, tuple) else (values,)
            identifiers = dict(zip(grouping, values))
            identifiers.update(level={1: 'dataset', 2: 'slice', 3: 'cell_type'}[len(grouping)])
            for metric in METRICS:
                array = part[metric].to_numpy(float)
                array = array[np.isfinite(array)]
                quantiles = np.quantile(array, [.05, .25, .5, .75, .95]) if len(array) else np.full(5, np.nan)
                records.append(dict(**identifiers, metric=metric, n_cells=len(part), n_finite=len(array),
                    mean=float(array.mean()) if len(array) else np.nan,
                    std=float(array.std(ddof=1)) if len(array) > 1 else np.nan,
                    p05=quantiles[0], q25=quantiles[1], median=quantiles[2], q75=quantiles[3], p95=quantiles[4]))
    return pd.DataFrame(records)


def plot_results(cells, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.titlesize': 10, 'svg.fonttype': 'none'})
    colors = ['#999999', '#4c78a8', '#f28e2b', '#59a14f']
    labels = ['scRNA-seq', 'Clean mapped', 'Simulated', 'Real CosMx']
    slices = sorted(cells.loc[cells.dataset.eq(SIMULATED), 'slice'].unique())
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 3.05))
    specifications = [
        ('detected_genes', DATASETS, labels, 'Gene detection', 'Detected genes/cell'),
        ('non_native_marker_genes', DATASETS, labels, 'Cross-type markers', 'Non-native markers/cell'),
        ('achieved_admixture_fraction', slices, slices, 'Achieved admixture', 'Added / total counts'),
        ('eligible_genes_per_query', slices, slices, 'Available genes', 'Eligible genes/query')]
    for number, (ax, specification) in enumerate(zip(axes, specifications)):
        metric, groups, ticks, title, ylabel = specification
        arrays = [(cells.loc[cells.dataset.eq(group), metric] if number < 2 else
                   cells.loc[cells.dataset.eq(SIMULATED) & cells['slice'].eq(group), metric]).dropna().to_numpy()
                  for group in groups]
        statistics = []
        for values in arrays:
            q = np.quantile(values, [.05, .25, .5, .75, .95])
            statistics.append(dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[]))
        artists = ax.bxp(statistics, showfliers=False, patch_artist=True, widths=.6,
                         medianprops=dict(color='black', linewidth=.8))
        for patch, color in zip(artists['boxes'], colors):
            patch.set(facecolor=color, alpha=.7)
        ax.set_xticks(np.arange(1, len(groups)+1), ticks, rotation=60, ha='right')
        ax.set(title=title, ylabel=ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        if metric == 'achieved_admixture_fraction':
            ax.set_ylim(0, 1)
    fig.tight_layout(w_pad=.8)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'priority_diagnostics.{extension}', dpi=400, bbox_inches='tight')
    plt.close(fig)

    types = [name for name in TYPE_ORDER if name in set(cells.cell_type)]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))
    for index, (ax, metric) in enumerate(zip(axes, METRICS[:2])):
        means = cells.groupby(['cell_type', 'dataset'], observed=True)[metric].mean()
        matrix = means.unstack('dataset').reindex(index=types, columns=DATASETS).to_numpy()
        image = ax.imshow(matrix, aspect='auto', cmap='Blues', vmin=0)
        ax.set_xticks(np.arange(4), labels, rotation=35, ha='right')
        ax.set_yticks(np.arange(len(types)), types if index == 0 else ['']*len(types))
        ax.set_title(['Mean detected genes', 'Mean non-native markers'][index])
        for row, col in np.ndindex(matrix.shape):
            ax.text(col, row, f'{matrix[row,col]:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if matrix[row,col] > np.nanmax(matrix)*.6 else 'black')
        fig.colorbar(image, ax=ax, fraction=.04, pad=.03)
    fig.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'diagnostics_by_cell_type.{extension}', dpi=400, bbox_inches='tight')
    plt.close(fig)


def signature(paths, settings):
    files = []
    for path in paths:
        stat = path.stat()
        files.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return dict(files=files, settings=settings,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def marker_count(value):
    if value.lower() == 'all':
        return None
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Use a positive integer or "all"')
    if result < 1:
        raise argparse.ArgumentTypeError('Use a positive integer or "all"')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--spatial-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--markers', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--slices', nargs='+', default=list(TEST_SLICES))
    parser.add_argument('--markers-per-cell-type', type=marker_count, default=3,
                        help='Positive integer or "all" (default: 3)')
    parser.add_argument('--unique-markers-only', action='store_true',
                        help='Keep genes positive/significant for exactly one type in the supplied DESeq2 table')
    parser.add_argument('--marker-padj', type=float, default=.05)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--replot-only', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.marker_padj <= 1:
        parser.error('Batch size, marker count, and padj threshold must be positive')
    if len(set(args.slices)) != len(args.slices) or not set(args.slices).issubset(TEST_SLICES):
        parser.error('Select unique held-out test slices only')
    started = time.perf_counter()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cell_path = output/'per_cell_diagnostics.parquet'
    if args.replot_only:
        plot_results(pd.read_parquet(cell_path), output)
        print(f'Replotted cached per-cell diagnostics: {output}', flush=True)
        return
    simulated_paths = {name: args.test_dir/f'{name}.h5ad' for name in args.slices}
    real_paths = {name: args.spatial_dir/f'st_integrated_slice-{name}.h5ad' for name in args.slices}
    paths = [args.reference, args.markers, *simulated_paths.values(), *real_paths.values()]
    expected_signature = signature(paths, dict(slices=args.slices,
        markers_per_cell_type=args.markers_per_cell_type, marker_padj=args.marker_padj,
        unique_markers_only=args.unique_markers_only))
    cache_signature = output/'input_signature.json'
    cached = (cell_path.exists() and cache_signature.exists() and
              json.loads(cache_signature.read_text()) == expected_signature)
    if cached and not args.recompute:
        print('Reusing cached per-cell diagnostics', flush=True)
        cells = pd.read_parquet(cell_path)
    else:
        metadata = {name: read_metadata(path) for name, path in simulated_paths.items()}
        real_metadata = {name: read_metadata(path) for name, path in real_paths.items()}
        ref_obs, ref_var, _ = read_metadata(args.reference)
        ref_obs = ref_obs.loc[ref_obs.cell_type.astype(str).ne('Treg')].copy()
        gene_sets = [set(ref_var.index)] + [set(item[1].index) for item in metadata.values()]
        gene_sets += [set(item[1].index) for item in real_metadata.values()]
        shared = set.intersection(*gene_sets)
        genes = pd.Index([gene for gene in ref_var.index if gene in shared])
        if not len(genes):
            raise ValueError('No shared panel genes')
        aliases = ref_var['gene_symbol'].astype(str).to_dict()
        types = sorted(set(ref_obs.cell_type.astype(str)) |
                       set.union(*(set(item[0].cell_type.astype(str)) for item in metadata.values())))
        markers = select_markers(args.markers, genes, types, aliases,
                                 args.markers_per_cell_type, args.marker_padj,
                                 args.unique_markers_only)
        markers.to_csv(output/'selected_markers.csv', index=False)
        marker_counts = markers.groupby('cell_type').size().reindex(types, fill_value=0)
        marker_counts.rename('n_markers').to_csv(output/'marker_counts_by_cell_type.csv')
        (output/'marker_selection.json').write_text(json.dumps(dict(
            marker_padj=args.marker_padj, log2_fold_change_greater_than=0,
            markers_per_cell_type=args.markers_per_cell_type,
            unique_markers_only=args.unique_markers_only,
            uniqueness_scope='all cell types in the supplied DESeq2 table, before ranking/truncation',
            n_selected_markers=len(markers), n_unique_genes=markers.gene.nunique(),
            n_markers_by_cell_type=marker_counts.to_dict()), indent=2)+'\n')
        pd.DataFrame(dict(gene_id=genes, gene_symbol=[aliases.get(gene, gene) for gene in genes])).to_csv(
            output/'shared_genes.csv', index=False)
        print(f'{len(genes)} shared genes; {len(markers)} marker assignments; '
              f'{markers.gene.nunique()} unique marker genes', flush=True)
        reference_index = read_metadata(args.reference)[0].index.get_indexer(ref_obs.index)
        counts = read_counts(args.reference, 'X', genes, reference_index)
        frames = [expression_diagnostics(counts, ref_obs, REFERENCE, 'Reference', genes,
                                         markers, 'cell_type')]
        print(f'Unique scRNA-seq reference: {len(ref_obs):,} cells', flush=True)
        del counts
        checks = []
        for name in args.slices:
            obs, var, config = metadata[name]
            if not obs.split.astype(str).eq('test').all() or not obs.spatial_slice.astype(str).eq(name).all():
                raise ValueError(f'{name}: input is not the expected held-out test slice')
            if obs.cell_type.astype(str).eq('Treg').any():
                raise ValueError('Unexpected Treg cells in test input')
            # Eligibility and source thresholds require the full original panel.
            if set(var.index) != set(genes):
                raise ValueError('The full simulation panel is required for exact neighbor recovery')
            native = read_counts(simulated_paths[name], 'layers/original_counts', genes)
            added = read_counts(simulated_paths[name], 'layers/synthetic_admixture_counts', genes)
            existing = read_counts(simulated_paths[name], 'X', genes)
            _same_counts(existing, native+added, f'{name}: clean + added != saved X')
            clean_frame = expression_diagnostics(native, obs, CLEAN, name, genes, markers, 'cell_type')
            sim_frame = expression_diagnostics(existing, obs, SIMULATED, name, genes, markers, 'cell_type')
            del existing
            print(f'{name}: measuring native neighbor eligibility for {len(obs):,} cells...', flush=True)
            capacity = eligibility(native, added, obs, config, args.batch_size)
            for key, values in capacity.items():
                sim_frame[key] = values
            clean_frame['eligible_genes_per_query'] = capacity['eligible_genes_per_query']
            native_total = np.asarray(native.sum(axis=1)).ravel()
            added_total = np.asarray(added.sum(axis=1)).ravel()
            total = native_total + added_total
            fraction = np.divide(added_total, total, out=np.full(len(obs), np.nan), where=total > 0)
            sim_frame['achieved_admixture_fraction'] = fraction
            sim_frame['native_counts'] = native_total
            sim_frame['added_counts'] = added_total
            if 'achieved_admixture_burden' in obs and not np.allclose(
                    np.nan_to_num(fraction), obs.achieved_admixture_burden.to_numpy(), atol=1e-8, rtol=0):
                raise AssertionError('Recomputed fraction differs from the saved achieved burden')
            real_obs, real_var, _ = real_metadata[name]
            rows = real_obs.index.get_indexer(obs.matched_st_id.astype(str))
            if (rows < 0).any() or len(np.unique(rows)) != len(rows):
                raise ValueError('Real cells cannot be matched uniquely to simulated recipients')
            matched = real_obs.iloc[rows]
            if not np.array_equal(matched.cell_type_harmonized.astype(str), obs.cell_type.astype(str)):
                raise ValueError('Matched real/simulated cell-type labels differ')
            real = read_counts(real_paths[name], 'X', genes, rows)
            real_frame = expression_diagnostics(real, matched, REAL, name, genes, markers, 'cell_type_harmonized')
            frames.extend([clean_frame, sim_frame, real_frame])
            checks.append(dict(slice=name, n_cells=len(obs), matrix_reconstruction_exact=True,
                original_neighbor_pools_verified=True, real_cells_and_types_matched=True,
                saved_achieved_fraction_verified=True, **{key: np.asarray(config[key]).tolist()
                    for key in ['k_neighbors', 'tau_range', 'min_source_counts', 'only_sample_unexpressed_genes']}))
            print(f'{name}: mean detected genes {clean_frame.detected_genes.mean():.1f} -> '
                  f'{sim_frame.detected_genes.mean():.1f}; real {real_frame.detected_genes.mean():.1f}; '
                  f'achieved fraction {np.nanmean(fraction):.3f}', flush=True)
            del native, added, real, clean_frame, sim_frame, real_frame
            gc.collect()
        cells = pd.concat(frames, ignore_index=True)
        for key in ['dataset', 'slice', 'cell_type']:
            cells[key] = cells[key].astype('category')
        cells.to_parquet(cell_path, index=False)
        (output/'validation.json').write_text(json.dumps(checks, indent=2)+'\n')
        cache_signature.write_text(json.dumps(expected_signature, indent=2)+'\n')
    summary = summarize(cells)
    summary.to_csv(output/'diagnostic_summary.csv', index=False)
    balanced = (cells.groupby(['dataset', 'cell_type'], observed=True)[METRICS[:2]].mean()
                .groupby('dataset', observed=True).mean().reindex(DATASETS))
    balanced.to_csv(output/'cell_type_balanced_means.csv')
    plot_results(cells, output)
    elapsed = time.perf_counter() - started
    text = '''# SYNDRI-SIM test diagnostics

All saved test cells are used; no new admixture is generated. The real CosMx
cells are matched to simulated recipients by `matched_st_id`, with identical
cell-type composition and gene panels. The unique original train/validation
scRNA-seq reference is included once, without repeating its rows across slices.
Clean mapped test scRNA-seq cells provide the paired before-admixture baseline.

- **Detected genes:** number of shared-panel genes with counts > 0 per cell.
- **Non-native marker genes:** detected genes in the union of selected markers
  for other cell types, excluding every marker selected for the query's own type.
  Markers are positive DESeq2 hits with adjusted p <= the configured threshold,
  ranked by adjusted p and then descending log2 fold change. They are a proxy
  for cross-type expression, not proof of contamination in real data.
- **Achieved admixture fraction:** added counts / (native + added counts), only
  for simulated ST. It is not added / native counts. Unknown fractions for real
  data and the reference are missing, not set to zero.
- **Eligible genes per query:** genes present in the original selected native
  neighbor pool and absent from the clean query, before adding admixture.
  Neighbor selection reuses the original builder and saved settings. Every added
  whole-gene count is verified against its original neighbor pool. The same
  eligibility values accompany the clean and admixed versions of each query;
  they do not count the residual zeros after admixture. No comparable native
  spatial neighbor pool is known for real data or the unique reference, so those
  values are missing.

`per_cell_diagnostics.parquet` caches every measurement. Additional columns
record added genes, eligible transcript counts, remaining eligible genes, and
the fraction of eligible genes selected. `diagnostic_summary.csv` gives pooled,
slice, and slice/cell-type summaries; `cell_type_balanced_means.csv` averages
within-type means equally to reduce cell-composition confounding.
`selected_markers.csv`, `shared_genes.csv`, `input_signature.json`, and
`validation.json` record definitions, inputs, and checks.

Boxes show medians, interquartile ranges, and 5th–95th percentile whiskers, with
all cells contributing. `diagnostics_by_cell_type` shows within-type means.
PNG, PDF, and SVG figures use a maximum width of 7.5 inches.

Rerunning the same command reuses cached measurements when the input metadata,
settings, and script match. Use `--recompute` to rebuild them or `--replot-only`
to render the existing measurements without reading input count matrices.
'''
    count_description = ('all qualifying markers' if args.markers_per_cell_type is None
                         else f'up to {args.markers_per_cell_type} markers per cell type')
    text += f'\n## Marker selection for this run\n\nUses {count_description}, with adjusted p <= {args.marker_padj} and log2 fold change > 0.\n'
    if args.unique_markers_only:
        text += ('A gene is retained only when it is a positive significant marker of exactly one\n'
                 'cell type across the full supplied DESeq2 table, before selecting evaluated types.\n'
                 'Significant downregulation in another type does not disqualify an upregulated marker.\n')
    text += ('`marker_counts_by_cell_type.csv` reports the resulting marker-set sizes;\n'
             '`marker_selection.json` records the selection rules.\n')
    (output/'README.md').write_text(text)
    pooled = summary.loc[summary.level.eq('dataset')]
    print(pooled.pivot(index='dataset', columns='metric', values='mean').to_string(), flush=True)
    (output/'run_info.json').write_text(json.dumps(dict(elapsed_seconds=elapsed,
        n_cells_by_dataset=cells.groupby('dataset', observed=True).size().to_dict()), indent=2)+'\n')
    print(f'Completed in {elapsed:.1f} seconds. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
