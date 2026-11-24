"""Independent transcript donation using the existing SYNDRI-SIM donor graph.

This sampler does not round weighted expression or select whole genes. It uses
original integer donor counts and normalizes the frozen type weights across all
selected donors of each query, including donors without eligible expression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from donor_weighted_admixture import ALLOCATION_SCHEMA


BINOMIAL_ALLOCATION_SCHEMA = ALLOCATION_SCHEMA.append(
    pa.field('normalized_donor_weight', pa.float64())).append(
    pa.field('donation_probability', pa.float64())).append(
    pa.field('sampled_lambda', pa.float64())).append(
    pa.field('neighborhood_weight_mean', pa.float64()))


def sample_binomial(native, obs, genes, graph, recipients, weights, config, seed,
                    allocations_path=None, batch_size=256):
    """D[j->i,g] ~ Binomial(X[j,g], min(1, lambda[i]*w[j]/mean(w[N[i]]))).

    One vectorized binomial draw per query covers all eligible donor/gene entries.
    No target budget, minimum-count cutoff, or depletion of source cells applies.
    Query-indexed streams make pilot/full results and batch sizes reproducible.
    """
    if config.get('admixture_target_mode') != 'binomial_transcripts':
        raise ValueError('Binomial sampling requires binomial_transcripts mode')
    maximum = float(config['lambda_max'])
    if not np.isfinite(maximum) or not 0 <= maximum <= 1:
        raise ValueError('lambda_max must be in [0,1]')
    if not config.get('only_sample_unexpressed_genes', False):
        raise ValueError('This controlled experiment requires native-zero genes')
    native = native.tocsr()
    if not np.issubdtype(native.dtype, np.integer) or (native.data < 0).any():
        raise ValueError('Binomial source counts must be nonnegative integers')
    if not native.has_canonical_format or (native.data == 0).any():
        native = native.copy()
        native.sum_duplicates()
        native.eliminate_zeros()
    recipients = np.asarray(recipients, dtype=np.int64)
    if len(np.unique(recipients)) != len(recipients):
        raise ValueError('Recipient rows must be unique')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    labels = obs.cell_type.astype(str).to_numpy()
    ids = obs.index.astype(str).to_numpy()
    source_sc = obs.source_sc_id.astype(str).to_numpy()
    matched = obs.matched_st_id.astype(str).to_numpy()
    weight_by_cell = np.array([weights[ct] for ct in labels], dtype=np.float64)
    if not np.isfinite(weight_by_cell).all() or (weight_by_cell <= 0).any():
        raise ValueError('Frozen donor weights must be finite and positive')
    totals = np.asarray(native.sum(axis=1)).ravel()
    row_parts, gene_parts, count_parts, diagnostics = [], [], [], []
    writer = (pq.ParquetWriter(allocations_path, BINOMIAL_ALLOCATION_SCHEMA,
                              compression='zstd') if allocations_path else None)
    try:
        for start in range(0, len(recipients), batch_size):
            queries = recipients[start:start+batch_size]
            edges = graph[queries]
            sources = edges.indices
            donor_counts = native[sources]
            record_parts = []
            for offset, query in enumerate(queries):
                edge_start, edge_end = edges.indptr[offset:offset+2]
                selected_sources = sources[edge_start:edge_end]
                raw_weights = weight_by_cell[selected_sources]
                mean_weight = float(raw_weights.mean()) if len(raw_weights) else 1.
                normalized = raw_weights / mean_weight
                if len(normalized) and not np.isclose(normalized.mean(), 1., atol=1e-12):
                    raise AssertionError('Neighborhood weights do not have mean one')
                fraction_rng = np.random.default_rng(np.random.SeedSequence([seed, int(query), 503]))
                lam = float(fraction_rng.uniform(0., maximum))
                probability = np.minimum(1., lam * normalized)
                begin, end = donor_counts.indptr[[edge_start, edge_end]]
                genes_all = donor_counts.indices[begin:end]
                counts_all = donor_counts.data[begin:end]
                lengths = np.diff(donor_counts.indptr[edge_start:edge_end+1])
                donors_all = np.repeat(selected_sources, lengths)
                probability_all = np.repeat(probability, lengths)
                native_positive = native.indices[native.indptr[query]:native.indptr[query+1]]
                eligible = ~np.isin(genes_all, native_positive)
                g, original, donors, p = (array[eligible] for array in
                    (genes_all, counts_all, donors_all, probability_all))
                donation_rng = np.random.default_rng(np.random.SeedSequence([seed, int(query), 607]))
                donated = donation_rng.binomial(original, p)
                positive = donated > 0
                pooled = np.bincount(g[positive], weights=donated[positive],
                                     minlength=native.shape[1]).astype(np.int64)
                selected = np.flatnonzero(pooled)
                available = int(original.sum())
                expected = float(np.dot(original, p))
                diagnostics.append((int(query), lam, int(totals[query]), int(donated.sum()),
                    int(len(np.unique(g))), available, len(selected), len(selected_sources),
                    mean_weight, float(normalized.mean()) if len(normalized) else np.nan,
                    expected, float(np.dot(original, p*(1-p))),
                    int(np.count_nonzero(probability == 1)),
                    float(probability.max()) if len(probability) else 0.))
                if len(selected):
                    row_parts.append(np.full(len(selected), start+offset, dtype=np.int64))
                    gene_parts.append(selected)
                    count_parts.append(pooled[selected])
                    if writer is not None:
                        n = int(positive.sum())
                        record_parts.append((np.full(n, query, dtype=np.int64), donors[positive],
                            g[positive], donated[positive], original[positive], p[positive],
                            np.full(n, lam), np.full(n, mean_weight)))
            if writer is not None and record_parts:
                q, d, g, count, original, p, lam, mean = [
                    np.concatenate([part[i] for part in record_parts]) for i in range(8)]
                writer.write_table(pa.Table.from_pydict(dict(
                    assigned_row=q, source_row=d, gene_index=g, count=count,
                    native_donor_count=original, donor_weight=weight_by_cell[d],
                    assigned_cell=ids[q], source_cell=ids[d], gene_id=np.asarray(genes)[g],
                    assigned_cell_type=labels[q], source_cell_type=labels[d],
                    assigned_sc_id=source_sc[q], source_sc_id=source_sc[d],
                    assigned_matched_st_id=matched[q], source_matched_st_id=matched[d],
                    normalized_donor_weight=weight_by_cell[d]/mean,
                    donation_probability=p, sampled_lambda=lam,
                    neighborhood_weight_mean=mean), schema=BINOMIAL_ALLOCATION_SCHEMA))
            if start == 0 or (start // batch_size) % 20 == 0 or start+batch_size >= len(recipients):
                print(f'  Binomial sampled {min(start+batch_size, len(recipients)):,}/{len(recipients):,} recipients', flush=True)
    finally:
        if writer is not None:
            writer.close()
    shape = len(recipients), native.shape[1]
    added = (sparse.coo_matrix((np.concatenate(count_parts),
        (np.concatenate(row_parts), np.concatenate(gene_parts))), shape=shape).tocsr()
        if count_parts else sparse.csr_matrix(shape, dtype=np.int64))
    stats = pd.DataFrame(diagnostics, columns=['input_row', 'sampled_lambda',
        'native_counts', 'added_counts', 'eligible_genes_per_query',
        'eligible_neighbor_transcripts', 'added_genes_per_query', 'selected_neighbor_count',
        'neighborhood_weight_mean', 'normalized_neighbor_weight_mean',
        'expected_added_counts', 'added_counts_variance', 'n_saturated_neighbors',
        'max_donation_probability'])
    total = stats.native_counts.to_numpy() + stats.added_counts.to_numpy()
    stats['achieved_admixture_burden'] = np.divide(stats.added_counts.to_numpy(), total,
        out=np.zeros(len(stats)), where=total > 0)
    stats['achieved_pool_fraction'] = np.divide(stats.added_counts.to_numpy(),
        stats.eligible_neighbor_transcripts.to_numpy(), out=np.zeros(len(stats)),
        where=stats.eligible_neighbor_transcripts.to_numpy() > 0)
    if (stats.added_counts > stats.eligible_neighbor_transcripts).any():
        raise AssertionError('Donation exceeds the unweighted native donor pool')
    if not np.array_equal(np.asarray(added.sum(axis=1)).ravel(), stats.added_counts):
        raise AssertionError('Sparse additions lost donated transcripts')
    return added, stats


def validate_binomial_allocations(path, native, obs, genes, recipients, added, weights,
                                  *, graph, stats, config, seed):
    """Reconstruct additions and verify every positive donor allocation."""
    recipients = np.asarray(recipients, dtype=np.int64)
    inverse = np.full(len(obs), -1, dtype=np.int64)
    inverse[recipients] = np.arange(len(recipients))
    labels = obs.cell_type.astype(str).to_numpy()
    ids = obs.index.astype(str).to_numpy()
    sc_ids = obs.source_sc_id.astype(str).to_numpy()
    matched_ids = obs.matched_st_id.astype(str).to_numpy()
    gene_ids = np.asarray(genes)
    weight_by_cell = np.array([weights[ct] for ct in labels])
    degree = np.diff(graph.indptr)
    mean_weights = np.divide(np.asarray(graph @ weight_by_cell).ravel(), degree,
                            out=np.ones(len(obs)), where=degree > 0)
    if not np.array_equal(stats.input_row.to_numpy(), recipients):
        raise AssertionError('Sampling stats are not aligned with recipients')
    expected_lambda = np.array([np.random.default_rng(np.random.SeedSequence([
        seed, int(q), 503])).uniform(0., config['lambda_max']) for q in recipients])
    if not np.array_equal(expected_lambda, stats.sampled_lambda):
        raise AssertionError('Query-specific lambda streams are inconsistent')
    if not np.allclose(stats.neighborhood_weight_mean, mean_weights[recipients], atol=1e-12, rtol=0):
        raise AssertionError('Local normalization did not use the full selected neighborhood')
    nonempty = degree[recipients] > 0
    if not np.allclose(stats.normalized_neighbor_weight_mean[nonempty], 1., atol=1e-12, rtol=0):
        raise AssertionError('Normalized weights do not have neighborhood mean one')
    recovered = sparse.csr_matrix(added.shape, dtype=np.int64)
    n_allocations = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250000):
        frame = batch.to_pandas()
        q, d, g = (frame[key].to_numpy(np.int64) for key in ['assigned_row', 'source_row', 'gene_index'])
        count = frame['count'].to_numpy(np.int64)
        if (q < 0).any() or (q >= len(obs)).any() or (d < 0).any() or (d >= len(obs)).any() or (g < 0).any() or (g >= len(genes)).any():
            raise AssertionError('Invalid allocation indices')
        if (inverse[q] < 0).any() or (q == d).any() or (labels[q] == labels[d]).any() or (count <= 0).any():
            raise AssertionError('Invalid donor-recipient assignment')
        if not np.asarray(graph[q, d]).ravel().all():
            raise AssertionError('An allocation source is absent from the fixed donor graph')
        for column, expected in [
            ('assigned_cell', ids[q]), ('source_cell', ids[d]), ('gene_id', gene_ids[g]),
            ('assigned_cell_type', labels[q]), ('source_cell_type', labels[d]),
            ('assigned_sc_id', sc_ids[q]), ('source_sc_id', sc_ids[d]),
            ('assigned_matched_st_id', matched_ids[q]), ('source_matched_st_id', matched_ids[d])]:
            if not np.array_equal(frame[column].to_numpy(), expected):
                raise AssertionError(f'Inconsistent allocation identifiers: {column}')
        original = np.asarray(native[d, g]).ravel()
        if not np.array_equal(original, frame.native_donor_count) or (count > original).any():
            raise AssertionError('Binomial donation exceeds original donor counts')
        normalized = weight_by_cell[d] / mean_weights[q]
        lam = stats.sampled_lambda.to_numpy()[inverse[q]]
        probability = np.minimum(1., lam*normalized)
        for expected, observed in [(weight_by_cell[d], frame.donor_weight),
            (mean_weights[q], frame.neighborhood_weight_mean), (normalized, frame.normalized_donor_weight),
            (probability, frame.donation_probability), (lam, frame.sampled_lambda)]:
            if not np.allclose(expected, observed, atol=1e-12, rtol=0):
                raise AssertionError('Incorrect binomial probability or local donor weight')
        recovered += sparse.coo_matrix((count, (inverse[q], g)), shape=added.shape).tocsr()
        n_allocations += len(frame)
    difference = recovered-added
    difference.eliminate_zeros()
    if difference.nnz:
        raise AssertionError('Donor allocations do not reconstruct the added matrix')
    if native[recipients].multiply(added).nnz:
        raise AssertionError('Admixture was added to a natively expressed gene')
    return dict(donor_allocations_exact=True, binomial_count_bounds_valid=True,
        binomial_probabilities_valid=True, local_weight_mean_one=True,
        lambda_streams_valid=True, only_unexpressed_genes_added=True,
        source_recipient_ids_valid=True, selected_donor_graph_preserved=True,
        no_transcripts_discarded=True, additive_contamination=True,
        n_donor_recipient_gene_allocations=n_allocations,
        n_native_transcripts=int(native[recipients].sum()), n_added_transcripts=int(added.sum()))
