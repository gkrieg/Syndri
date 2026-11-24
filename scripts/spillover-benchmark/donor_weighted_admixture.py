"""Donor-type calibration and integer whole-gene sampling for SYNDRI-SIM.

This alternative does not modify Heimdall's original admixture sampler.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from Heimdall.admixture_sampling import build_neighbor_dict, sample_tau_admixture_from_neighbors


def neighbor_graph(native, obs, config, cache_path=None):
    """Use the existing source-selection rules on the complete clean slice."""
    if cache_path is not None and Path(cache_path).exists():
        graph = sparse.load_npz(cache_path).tocsr()
        if graph.shape != (len(obs), len(obs)):
            raise ValueError('Cached graph has the wrong shape')
        return graph
    part = ad.AnnData(native, obs=obs[['cell_type', 'spatial_slice',
        'CenterX_global_px', 'CenterY_global_px']].copy())
    part = build_neighbor_dict(part, k_neighbors=int(config['k_neighbors']),
        coord_cols=('CenterX_global_px', 'CenterY_global_px'), tissue_col='spatial_slice',
        cell_type_col='cell_type', min_source_counts=int(config['min_source_counts']),
        use_same_cell_type_neighbors=bool(config['use_same_cell_type_neighbors']))
    neighbors = part.uns['neighbor_dict']
    ids = obs.index.astype(str)
    sizes = np.array([len(neighbors.get(key, [])) for key in ids], dtype=np.int64)
    columns = ids.get_indexer([source for key in ids for source in neighbors.get(key, [])])
    if (columns < 0).any():
        raise AssertionError('Unknown source cell in neighbor graph')
    graph = sparse.csr_matrix((np.ones(len(columns), dtype=np.int64), columns,
        np.r_[0, np.cumsum(sizes)]), shape=(len(obs), len(obs)))
    if graph.diagonal().any():
        raise AssertionError('A query occurs in its own neighbor pool')
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache_path, graph)
    return graph


def marker_detection(counts, labels, types, genes, markers):
    """C[r,d]: mean detection probability over the marker set for donor d."""
    labels = np.asarray(labels, dtype=str)
    codes = pd.Index(types).get_indexer(labels)
    if (codes < 0).any():
        raise ValueError('A cell type has no calibration definition')
    grouping = sparse.csr_matrix((np.ones(len(labels), dtype=np.int64),
        (codes, np.arange(len(labels)))), shape=(len(types), len(labels)))
    n_cells = np.bincount(codes, minlength=len(types))
    detections = (grouping @ (counts > 0)).toarray()
    values = np.full((len(types), len(types)), np.nan)
    for donor, name in enumerate(types):
        indices = genes.get_indexer(markers.loc[markers.cell_type.eq(name), 'gene'])
        if len(indices) == 0 or (indices < 0).any():
            raise ValueError(f'No available exclusive markers for {name}')
        values[:, donor] = np.divide(detections[:, indices].sum(axis=1),
            n_cells * len(indices), out=np.full(len(types), np.nan), where=n_cells > 0)
    return values, n_cells


def marker_availability(native, obs, graph, types, genes, markers):
    """Conditional E[r,d], per query, marker, and selected type-d donor.

    Queries with no selected type-d donor do not contribute. Exposed queries
    with zero eligible marker expression do contribute zero. Return exposure
    counts as well so real CosMx detection can use the identical recipients.
    """
    labels = obs.cell_type.astype(str).to_numpy()
    values = np.full((len(types), len(types)), np.nan)
    support = np.zeros((len(types), len(types)), dtype=np.int64)
    donor_counts = np.zeros((len(obs), len(types)), dtype=np.int32)
    for donor, name in enumerate(types):
        indices = genes.get_indexer(markers.loc[markers.cell_type.eq(name), 'gene'])
        if len(indices) == 0 or (indices < 0).any():
            raise ValueError(f'No available exclusive markers for {name}')
        n_donors = np.asarray(graph @ (labels == name).astype(np.int64)).ravel()
        donor_counts[:, donor] = n_donors
        source = sparse.diags((labels == name).astype(np.int64)) @ native[:, indices]
        pool = graph @ source
        expressed = native[:, indices] > 0
        eligible_total = np.asarray(pool.sum(axis=1) - pool.multiply(expressed).sum(axis=1)).ravel()
        per_query = np.divide(eligible_total, len(indices) * n_donors,
            out=np.full(len(obs), np.nan), where=n_donors > 0)
        for recipient, recipient_name in enumerate(types):
            rows = (labels == recipient_name) & (n_donors > 0)
            support[recipient, donor] = int(rows.sum())
            if rows.any():
                values[recipient, donor] = per_query[rows].mean()
    return values, support, donor_counts


def exposed_marker_detection(counts, labels, types, genes, markers, donor_counts):
    """C[r,d] over the same matched, donor-exposed real recipients as E[r,d]."""
    labels = np.asarray(labels, dtype=str)
    if donor_counts.shape != (len(labels), len(types)):
        raise ValueError('Exposure rows must match real recipient rows exactly')
    values = np.full((len(types), len(types)), np.nan)
    support = np.zeros((len(types), len(types)), dtype=np.int64)
    for d, donor_name in enumerate(types):
        indices = genes.get_indexer(markers.loc[markers.cell_type.eq(donor_name), 'gene'])
        if len(indices) == 0 or (indices < 0).any():
            raise ValueError(f'No available exclusive markers for {donor_name}')
        per_cell = np.asarray((counts[:, indices] > 0).sum(axis=1)).ravel() / len(indices)
        for r, recipient_name in enumerate(types):
            rows = (labels == recipient_name) & (donor_counts[:, d] > 0)
            support[r,d] = int(rows.sum())
            if rows.any():
                values[r,d] = per_cell[rows].mean()
    return values, support


def donor_weights(strata, min_cells=50, shrinkage=.25, epsilon=1e-8, min_slices=2):
    """Balance exposed strata; retain unsupported donor weights at exactly 1."""
    if not 0 <= shrinkage <= 1 or epsilon <= 0:
        raise ValueError('Shrinkage must be in [0,1] and epsilon must be positive')
    if min_cells < 1 or min_slices < 1:
        raise ValueError('Support thresholds must be positive')
    all_types = pd.Index(sorted(strata.donor_type.unique()), name='donor_type')
    cross = strata.loc[strata.recipient_type.ne(strata.donor_type)]
    use = strata.loc[strata.recipient_type.ne(strata.donor_type) &
        strata.n_spatial_cells.ge(min_cells) & strata.n_native_cells.ge(min_cells) &
        strata.n_reference_cells.ge(min_cells) &
        np.isfinite(strata.cosmx_detection) &
        np.isfinite(strata.available_marker_counts)].copy()
    use['excess_detection'] = np.maximum(0., use.cosmx_detection - use.reference_detection)
    by_slice = use.groupby(['slice', 'donor_type'], as_index=False).agg(
        excess_detection=('excess_detection', 'mean'),
        available_marker_counts=('available_marker_counts', 'mean'),
        n_recipient_types=('recipient_type', 'nunique'),
        n_exposed_neighborhoods=('n_native_cells', 'sum'))
    weights = by_slice.groupby('donor_type').agg(
        excess_detection=('excess_detection', 'mean'),
        available_marker_counts=('available_marker_counts', 'mean'),
        n_calibration_slices=('slice', 'nunique'),
        n_exposed_neighborhoods_used=('n_exposed_neighborhoods', 'sum')).reindex(all_types)
    weights['n_exposed_neighborhoods_total'] = cross.groupby('donor_type').n_native_cells.sum().reindex(all_types, fill_value=0)
    weights['n_real_exposed_neighborhoods_total'] = cross.groupby('donor_type').n_spatial_cells.sum().reindex(all_types, fill_value=0)
    weights['n_eligible_recipient_strata'] = use.groupby('donor_type').size().reindex(all_types, fill_value=0)
    for column in ['n_calibration_slices', 'n_exposed_neighborhoods_used']:
        weights[column] = weights[column].fillna(0).astype(np.int64)
    weights['supported'] = (weights.n_calibration_slices.ge(min_slices) &
        weights.available_marker_counts.gt(0) & np.isfinite(weights.available_marker_counts) &
        np.isfinite(weights.excess_detection))
    weights['neutral_reason'] = ''
    weights.loc[weights.n_calibration_slices.lt(min_slices), 'neutral_reason'] = 'insufficient_donor_exposed_strata_or_slices'
    weights.loc[weights.n_calibration_slices.ge(min_slices) &
        ~weights.supported, 'neutral_reason'] = 'no_positive_eligible_marker_availability'
    weights['raw_weight'] = np.nan
    supported = weights.supported
    weights.loc[supported, 'raw_weight'] = weights.loc[supported, 'excess_detection'] / (
        weights.loc[supported, 'available_marker_counts'] + epsilon)
    weights['normalized_weight'] = 1.
    weights['weight'] = 1.
    mean_raw = weights.loc[supported, 'raw_weight'].mean()
    if supported.any() and mean_raw > 0:
        # Supported weights have mean 1, while every unsupported type stays 1;
        # therefore all donor types together also have mean 1 after shrinkage.
        weights.loc[supported, 'normalized_weight'] = weights.loc[supported, 'raw_weight'] / mean_raw
        weights.loc[supported, 'weight'] = (1-shrinkage) * weights.loc[supported, 'normalized_weight'] + shrinkage
        weights.loc[supported, 'weight'] /= weights.loc[supported, 'weight'].mean()
    elif supported.any():
        weights.loc[supported, 'neutral_reason'] = 'no_positive_supported_excess'
        weights.loc[supported, 'supported'] = False
    if not np.isfinite(weights.weight).all() or (weights.weight < 0).any() or not np.isclose(weights.weight.mean(), 1):
        raise AssertionError('Invalid normalized donor weights')
    if not weights.loc[~weights.supported, 'weight'].eq(1.).all():
        raise AssertionError('An unsupported donor type lost its neutral weight')
    return weights.reset_index(), by_slice, use


ALLOCATION_SCHEMA = pa.schema([
    ('assigned_row', pa.int64()), ('source_row', pa.int64()), ('gene_index', pa.int64()),
    ('count', pa.int64()), ('native_donor_count', pa.int64()), ('donor_weight', pa.float64()),
    ('assigned_cell', pa.string()), ('source_cell', pa.string()), ('gene_id', pa.string()),
    ('assigned_cell_type', pa.string()), ('source_cell_type', pa.string()),
    ('assigned_sc_id', pa.string()), ('source_sc_id', pa.string()),
    ('assigned_matched_st_id', pa.string()), ('source_matched_st_id', pa.string()),
])


def sample_pool_fraction_from_neighbors(pool, fraction, rng, min_n=10):
    """Select whole pooled genes toward round(fraction * eligible pool total)."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('The donor-pool fraction must be in [0,1]')
    eligible = np.flatnonzero(pool)
    available = int(pool[eligible].sum())
    target = int(round(float(fraction) * available))
    if not available or target < min_n or target <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), target
    order = rng.permutation(len(eligible))
    genes = eligible[order]
    counts = pool[genes]
    stop = int(np.searchsorted(np.cumsum(counts), target, side='left')) + 1
    return genes[:stop], counts[:stop], target


def sample_weighted(native, obs, genes, graph, recipients, weights, config, seed,
                    allocations_path=None, batch_size=256):
    """Round each weighted donor/gene contribution, then add whole pooled genes.

    Rounding is floor(w*X) + Bernoulli(frac(w*X)); expected contribution is w*X.
    Per-query streams are invariant to batching and pilot/full query selection.
    """
    native = native.tocsr()
    target_mode = config.get('admixture_target_mode', 'final_fraction')
    if target_mode not in {'final_fraction', 'pool_fraction'}:
        raise ValueError(f'Unknown admixture target mode: {target_mode}')
    if target_mode == 'pool_fraction' and not 0 <= float(config['lambda_max']) <= 1:
        raise ValueError('lambda_max must be in [0,1]')
    recipients = np.asarray(recipients, dtype=np.int64)
    labels = obs.cell_type.astype(str).to_numpy()
    ids = obs.index.astype(str).to_numpy()
    source_sc = obs.source_sc_id.astype(str).to_numpy()
    matched = obs.matched_st_id.astype(str).to_numpy()
    weights_by_cell = np.array([weights[label] for label in labels], dtype=float)
    totals = np.asarray(native.sum(axis=1)).ravel()
    rows_out, cols_out, counts_out, diagnostics = [], [], [], []
    writer = pq.ParquetWriter(allocations_path, ALLOCATION_SCHEMA, compression='zstd') if allocations_path else None
    try:
        for start in range(0, len(recipients), batch_size):
            queries = recipients[start:start+batch_size]
            edges = graph[queries]
            sources = edges.indices
            donor_counts = native[sources]
            record_parts = []
            for offset, query in enumerate(queries):
                edge_start, edge_end = edges.indptr[offset:offset+2]
                begin, end = donor_counts.indptr[[edge_start, edge_end]]
                gene_indices = donor_counts.indices[begin:end]
                original = donor_counts.data[begin:end]
                source_rows = np.repeat(sources[edge_start:edge_end],
                    np.diff(donor_counts.indptr[edge_start:edge_end+1]))
                expected = original * weights_by_cell[source_rows]
                lower = np.floor(expected).astype(np.int64)
                rounding_rng = np.random.default_rng(np.random.SeedSequence([seed, int(query), 401]))
                contribution = lower + (rounding_rng.random(len(lower)) < expected-lower)
                pool = np.bincount(gene_indices, weights=contribution, minlength=native.shape[1]).astype(np.int64)
                if bool(config['only_sample_unexpressed_genes']):
                    pool[native.indices[native.indptr[query]:native.indptr[query+1]]] = 0
                eligible = np.flatnonzero(pool)
                sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, int(query), 503]))
                if target_mode == 'pool_fraction':
                    sampled_fraction = sampling_rng.uniform(0., float(config['lambda_max']))
                    selected, counts, target = sample_pool_fraction_from_neighbors(
                        pool, sampled_fraction, sampling_rng,
                        min_n=int(config['min_sampled_transcript_threshold']))
                else:
                    sampled_fraction = sampling_rng.uniform(*config['tau_range'])
                    selected, counts = sample_tau_admixture_from_neighbors(
                        {int(g): int(pool[g]) for g in eligible}, int(totals[query]), float(sampled_fraction),
                        rng=sampling_rng, min_n=int(config['min_sampled_transcript_threshold']))
                    target = int(round(sampled_fraction / (1-sampled_fraction) * totals[query]))
                added_total = int(counts.sum())
                diagnostics.append((int(query), sampled_fraction, target, int(totals[query]), added_total,
                    len(eligible), int(pool.sum()), len(selected), int(edge_end-edge_start)))
                if len(selected):
                    rows_out.append(np.full(len(selected), start+offset, dtype=np.int64))
                    cols_out.append(selected)
                    counts_out.append(counts)
                    if writer is not None:
                        keep = (contribution > 0) & np.isin(gene_indices, selected)
                        record_parts.append((np.full(int(keep.sum()), query, dtype=np.int64),
                            source_rows[keep], gene_indices[keep], contribution[keep], original[keep]))
            if writer is not None and record_parts:
                q, d, g, count, original = [np.concatenate([part[i] for part in record_parts]) for i in range(5)]
                table = pa.Table.from_pydict(dict(assigned_row=q, source_row=d, gene_index=g,
                    count=count, native_donor_count=original, donor_weight=weights_by_cell[d],
                    assigned_cell=ids[q], source_cell=ids[d], gene_id=np.asarray(genes)[g],
                    assigned_cell_type=labels[q], source_cell_type=labels[d],
                    assigned_sc_id=source_sc[q], source_sc_id=source_sc[d],
                    assigned_matched_st_id=matched[q], source_matched_st_id=matched[d]), schema=ALLOCATION_SCHEMA)
                writer.write_table(table)
            if start == 0 or (start // batch_size) % 20 == 0 or start+batch_size >= len(recipients):
                print(f'  Sampled {min(start+batch_size, len(recipients)):,}/{len(recipients):,} recipients', flush=True)
    finally:
        if writer is not None:
            writer.close()
    shape = len(recipients), native.shape[1]
    if counts_out:
        added = sparse.coo_matrix((np.concatenate(counts_out),
            (np.concatenate(rows_out), np.concatenate(cols_out))), shape=shape).tocsr()
    else:
        added = sparse.csr_matrix(shape, dtype=np.int64)
    fraction_column = 'sampled_lambda' if target_mode == 'pool_fraction' else 'sampled_tau'
    stats = pd.DataFrame(diagnostics, columns=['input_row', fraction_column, 'target_added_counts',
        'native_counts', 'added_counts', 'eligible_genes_per_query', 'eligible_neighbor_transcripts',
        'added_genes_per_query', 'selected_neighbor_count'])
    total = stats.native_counts.to_numpy() + stats.added_counts.to_numpy()
    stats['achieved_admixture_burden'] = np.divide(stats.added_counts.to_numpy(), total,
        out=np.zeros(len(stats)), where=total > 0)
    if target_mode == 'pool_fraction':
        stats['achieved_pool_fraction'] = np.divide(stats.added_counts.to_numpy(),
            stats.eligible_neighbor_transcripts.to_numpy(), out=np.zeros(len(stats)),
            where=stats.eligible_neighbor_transcripts.to_numpy() > 0)
        expected_target = np.rint(stats.sampled_lambda * stats.eligible_neighbor_transcripts).astype(np.int64)
        if not np.array_equal(stats.target_added_counts, expected_target):
            raise AssertionError('Pool-fraction targets do not match sampled lambda')
        active = (expected_target >= int(config['min_sampled_transcript_threshold'])) & (expected_target > 0)
        if (stats.loc[active, 'added_counts'] < expected_target[active]).any():
            raise AssertionError('Whole-gene sampling failed to reach its pool target')
        if (stats.added_counts > stats.eligible_neighbor_transcripts).any():
            raise AssertionError('More counts added than the eligible weighted pool contains')
        if stats.loc[~active, 'added_counts'].ne(0).any():
            raise AssertionError('Counts were added below the minimum target threshold')
    return added, stats


def validate_allocations(path, native, obs, genes, recipients, added, weights):
    """Stream known weighted donor assignments and reconstruct added counts."""
    inverse = np.full(native.shape[0], -1, dtype=np.int64)
    inverse[recipients] = np.arange(len(recipients))
    recovered = sparse.csr_matrix(added.shape, dtype=np.int64)
    labels = obs.cell_type.astype(str).to_numpy()
    ids = obs.index.astype(str).to_numpy()
    n_allocations = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250000):
        frame = batch.to_pandas()
        q, d, g = (frame[key].to_numpy(np.int64) for key in ['assigned_row', 'source_row', 'gene_index'])
        count = frame['count'].to_numpy(np.int64)
        if (q < 0).any() or (q >= len(obs)).any() or (d < 0).any() or (d >= len(obs)).any() or (g < 0).any() or (g >= len(genes)).any():
            raise AssertionError('Invalid allocation indices')
        if (inverse[q] < 0).any() or (q == d).any() or (labels[q] == labels[d]).any() or (count <= 0).any():
            raise AssertionError('Invalid donor-recipient assignment')
        for column, expected in [('assigned_cell', ids[q]), ('source_cell', ids[d]),
                ('gene_id', np.asarray(genes)[g]), ('assigned_cell_type', labels[q]), ('source_cell_type', labels[d])]:
            if not np.array_equal(frame[column].to_numpy(), expected):
                raise AssertionError(f'Inconsistent allocation identifiers: {column}')
        original = np.asarray(native[d, g]).ravel()
        if not np.array_equal(original, frame.native_donor_count):
            raise AssertionError('Allocation native donor counts differ from source counts')
        w = np.array([weights[label] for label in labels[d]])
        expectation = original * w
        if not np.allclose(w, frame.donor_weight, atol=1e-12, rtol=0) or np.any(
                (count < np.floor(expectation)) | (count > np.ceil(expectation))):
            raise AssertionError('Donor contribution is inconsistent with stochastic rounding')
        recovered += sparse.coo_matrix((count, (inverse[q], g)), shape=added.shape).tocsr()
        n_allocations += len(frame)
    difference = recovered-added
    difference.eliminate_zeros()
    if difference.nnz:
        raise AssertionError('Donor allocations do not reconstruct the added matrix')
    if native[recipients].multiply(added).nnz:
        raise AssertionError('Admixture was added to a natively expressed gene')
    return dict(donor_allocations_exact=True, stochastic_rounding_valid=True,
        only_unexpressed_genes_added=True, source_recipient_ids_valid=True,
        n_donor_recipient_gene_allocations=n_allocations,
        n_native_transcripts=int(native[recipients].sum()), n_added_transcripts=int(added.sum()))
