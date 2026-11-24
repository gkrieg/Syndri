#!/usr/bin/env python3
"""Calibrate donor weights, run a pilot, then optionally create all test slices."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '8')

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from diagnose_test_admixture import (read_metadata, read_counts, select_markers,
    expression_diagnostics, summarize, plot_results, REFERENCE, CLEAN, SIMULATED, REAL,
    DATASETS, METRICS, TEST_SLICES, _same_counts)
from donor_weighted_admixture import (neighbor_graph, marker_detection,
    marker_availability, exposed_marker_detection, donor_weights, sample_weighted, validate_allocations)

CALIBRATION_SLICES = ['Lung5_Rep1', 'Lung5_Rep2', 'Lung5_Rep3']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2,
        default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))+'\n')


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def graph_path(root, input_path, slice_name, config):
    key = hashlib.sha256(json.dumps([file_signature(input_path), slice_name,
        {name: config[name] for name in ['k_neighbors', 'min_source_counts',
        'use_same_cell_type_neighbors']}], sort_keys=True,
        default=lambda value: value.tolist()).encode()).hexdigest()[:16]
    return root/'cache'/f'{slice_name}_neighbors_{key}.npz'


def matched_real(path, simulated_obs, genes):
    obs, _, _ = read_metadata(path)
    indices = obs.index.get_indexer(simulated_obs.matched_st_id.astype(str))
    if (indices < 0).any() or len(np.unique(indices)) != len(indices):
        raise ValueError('Missing or duplicated matched real cells')
    selected = obs.iloc[indices].copy()
    if not np.array_equal(selected.cell_type_harmonized.astype(str), simulated_obs.cell_type.astype(str)):
        raise AssertionError('Matched real and simulated cell types differ')
    return read_counts(path, 'X', genes, indices), selected


def calibrate(args, genes, markers, config, ref_obs, ref_counts):
    folder = args.output_dir/'calibration'
    folder.mkdir(parents=True, exist_ok=True)
    input_paths = [args.training, args.reference, args.markers] + [
        args.spatial_dir/f'st_integrated_slice-{name}.h5ad' for name in args.calibration_slices]
    specification = dict(inputs=[file_signature(path) for path in input_paths],
        slices=args.calibration_slices, min_cells=args.min_calibration_cells,
        min_supported_slices=args.min_calibration_slices,
        exposure_graph='same selected native graph; real cells matched by matched_st_id',
        availability='conditional on donor presence, per marker and per selected donor',
        shrinkage=args.shrinkage, epsilon=args.epsilon, markers=markers.gene.tolist(),
        neighbor_config={key: config[key] for key in ['k_neighbors', 'min_source_counts',
            'use_same_cell_type_neighbors']},
        algorithm_sha256=hashlib.sha256(Path(__file__).with_name('donor_weighted_admixture.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Canonicalize numpy scalars before comparing to a saved JSON specification.
    specification = json.loads(json.dumps(specification, default=lambda x: x.tolist()))
    manifest = folder/'specification.json'
    weights_path = folder/'donor_weights.csv'
    if manifest.exists() and weights_path.exists() and json.loads(manifest.read_text()) == specification:
        print('Using calibrated donor weights from cache', flush=True)
        weights = pd.read_csv(weights_path)
        weights['neutral_reason'] = weights['neutral_reason'].fillna('')
        return weights
    types = sorted(ref_obs.cell_type.astype(str).unique())
    reference_profile, reference_n = marker_detection(ref_counts,
        ref_obs.cell_type.astype(str), types, genes, markers)
    train_obs, train_var, _ = read_metadata(args.training)
    if set(train_var.index) != set(genes):
        raise ValueError('Calibration must use the complete shared simulation panel')
    train_counts = read_counts(args.training, 'X', genes)
    strata = []
    for name in args.calibration_slices:
        print(f'Calibrating {name}...', flush=True)
        rows = np.flatnonzero(train_obs.spatial_slice.astype(str).eq(name) &
                             train_obs.split.astype(str).eq('train'))
        if len(rows) == 0:
            raise ValueError(f'{name} is not a populated training slice')
        obs, native = train_obs.iloc[rows].copy(), train_counts[rows]
        if set(obs.cell_type.astype(str)) - set(types):
            raise ValueError('Calibration cell types are absent from the reference')
        real, real_obs = matched_real(args.spatial_dir/f'st_integrated_slice-{name}.h5ad', obs, genes)
        graph = neighbor_graph(native, obs, config,
            graph_path(args.output_dir, args.training, name, config))
        availability, exposed_n, donor_counts = marker_availability(native, obs, graph, types, genes, markers)
        real_profile, real_exposed_n = exposed_marker_detection(real,
            real_obs.cell_type_harmonized, types, genes, markers, donor_counts)
        if not np.array_equal(exposed_n, real_exposed_n):
            raise AssertionError('Availability and real detection use different exposed recipient subsets')
        native_n = obs.cell_type.astype(str).value_counts().reindex(types, fill_value=0)
        for r, recipient in enumerate(types):
            for d, donor in enumerate(types):
                strata.append(dict(slice=name, recipient_type=recipient, donor_type=donor,
                    cosmx_detection=real_profile[r,d], reference_detection=reference_profile[r,d],
                    available_marker_counts=availability[r,d], n_spatial_cells=int(real_exposed_n[r,d]),
                    n_native_cells=int(exposed_n[r,d]), n_reference_cells=int(reference_n[r]),
                    n_recipient_cells_total=int(native_n.loc[recipient]),
                    n_selected_donors=int(donor_counts[obs.cell_type.astype(str).eq(recipient),d].sum()),
                    mean_selected_donors_when_exposed=(
                        float(donor_counts[obs.cell_type.astype(str).eq(recipient),d].sum()/exposed_n[r,d])
                        if exposed_n[r,d] else np.nan)))
        del native, real, graph
        gc.collect()
    weights, by_slice, eligible_strata = donor_weights(pd.DataFrame(strata),
        args.min_calibration_cells, args.shrinkage, args.epsilon, args.min_calibration_slices)
    weights.to_csv(weights_path, index=False)
    pd.DataFrame(strata).to_csv(folder/'all_marker_detection_strata.csv', index=False)
    eligible_strata.to_csv(folder/'eligible_calibration_strata.csv', index=False)
    by_slice.to_csv(folder/'slice_donor_estimates.csv', index=False)
    all_strata = pd.DataFrame(strata)
    support = all_strata.loc[all_strata.recipient_type.ne(all_strata.donor_type)].groupby(
        ['slice', 'donor_type'], as_index=False).agg(
            n_donor_exposed_neighborhoods=('n_native_cells', 'sum'),
            n_real_exposed_neighborhoods=('n_spatial_cells', 'sum'),
            n_selected_donors=('n_selected_donors', 'sum'))
    support.to_csv(folder/'donor_exposure_support_by_slice.csv', index=False)
    markers.to_csv(folder/'selected_markers.csv', index=False)
    write_json(manifest, specification)
    print(weights[['donor_type', 'excess_detection', 'available_marker_counts',
        'n_exposed_neighborhoods_total', 'n_exposed_neighborhoods_used',
        'n_calibration_slices', 'supported', 'weight']].to_string(index=False), flush=True)
    return weights


def pilot_rows(obs, n_per_type, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    groups = obs.groupby('cell_type', observed=True, sort=True).indices
    return np.sort(np.concatenate([rng.choice(rows, min(len(rows), n_per_type), replace=False)
                                    for rows in groups.values()]))


def write_slice(args, name, weights, markers, ref_obs, ref_counts, pilot=False,
                sampler=sample_weighted, allocation_validator=validate_allocations):
    input_path = args.test_dir/f'{name}.h5ad'
    obs, var, old_config = read_metadata(input_path)
    if not obs.split.astype(str).eq('test').all() or not obs.spatial_slice.astype(str).eq(name).all():
        raise ValueError('Input is not the requested held-out test slice')
    genes = var.index
    native = read_counts(input_path, 'layers/original_counts', genes)
    config = dict(old_config)
    target_mode = getattr(args, 'target_mode', 'final_fraction')
    config['admixture_target_mode'] = target_mode
    if target_mode in {'pool_fraction', 'binomial_transcripts'}:
        config.pop('tau_range', None)
        config['lambda_max'] = float(args.lambda_max)
    if target_mode == 'binomial_transcripts':
        # Independent donation has no target-count minimum or whole-gene cutoff.
        config['min_sampled_transcript_threshold'] = 0
        config['only_sample_unexpressed_genes'] = True
    seed = args.seed + sorted(TEST_SLICES).index(name)
    recipients = pilot_rows(obs, args.pilot_cells_per_type, args.seed) if pilot else np.arange(len(obs))
    folder = args.output_dir/'pilot' if pilot else args.output_dir/'slices'
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder/f'{name}.h5ad'
    if output_path.exists():
        raise FileExistsError(f'{output_path}: choose another output root to avoid overwriting results')
    graph = neighbor_graph(native, obs, config, graph_path(args.output_dir, input_path, name, config))
    print(f'{"Pilot" if pilot else "Test"} {name}: {len(recipients):,} recipients; full slice source pool {len(obs):,}', flush=True)
    allocations_path = folder/f'{name}_donor_allocations.parquet'
    weight_map = dict(zip(weights.donor_type, weights.weight))
    added, stats = sampler(native, obs, genes, graph, recipients, weight_map,
        config, seed, allocations_path=allocations_path, batch_size=args.batch_size)
    validation_kwargs = (dict(graph=graph, stats=stats, config=config, seed=seed)
                         if target_mode == 'binomial_transcripts' else {})
    validation = allocation_validator(allocations_path, native, obs, genes,
        recipients, added, weight_map, **validation_kwargs)
    clean = native[recipients]
    combined = clean+added
    if (added.data < 0).any() or not np.issubdtype(added.dtype, np.integer):
        raise AssertionError('Added counts must be nonnegative integers')
    selected_obs = obs.iloc[recipients].copy()
    for key in stats.columns:
        selected_obs[key] = stats[key].to_numpy()
    output = ad.AnnData(combined, obs=selected_obs, var=var.copy())
    output.layers['original_counts'] = clean
    output.layers['synthetic_admixture_counts'] = added
    output.layers['admixed_counts'] = combined.copy()
    with h5py.File(input_path, 'r') as handle:
        output.uns = ad.io.read_elem(handle['uns'])
    output.uns.pop('neighbor_dict', None)
    output.uns.pop('neighbor_dict_config', None)
    config.update(seed=seed, sampler='donor_type_weighted',
        integerization='floor(weight * source_count) + Bernoulli(fractional_part)',
        rng_scheme='SeedSequence([seed,input_row,401]) rounding; [seed,input_row,503] fraction/gene order',
        donor_type_weights=weight_map, calibration_slices=args.calibration_slices,
        min_calibration_cells=args.min_calibration_cells, weight_shrinkage=args.shrinkage,
        min_calibration_slices=args.min_calibration_slices,
        availability_conditioned_on_donor_presence=True, availability_per_donor_normalized=True,
        exposure_graph='same selected native graph for availability and matched real detection',
        donor_weight_support={row.donor_type: dict(
            n_exposed_total=int(row.n_exposed_neighborhoods_total),
            n_exposed_used=int(row.n_exposed_neighborhoods_used),
            n_supported_slices=int(row.n_calibration_slices),
            supported=bool(row.supported), neutral_reason=row.neutral_reason)
            for row in weights.itertuples()},
        weight_epsilon=args.epsilon, calibration_split='train',
        marker_definition='all positive DESeq2 hits with padj <= 0.05, unique to one type',
        source_input=str(input_path.resolve()), donor_allocations_file=str(allocations_path.resolve()),
        is_pilot=pilot)
    if target_mode == 'pool_fraction':
        config['sampler'] = 'donor_type_weighted_pool_fraction'
        config['target_added_counts_definition'] = 'round(sampled_lambda * eligible_weighted_pool_total)'
    elif target_mode == 'binomial_transcripts':
        config.update(sampler='donor_type_weighted_binomial_transcripts',
            integerization='none; binomial trials are original integer donor counts',
            rng_scheme='SeedSequence([seed,input_row,503]) lambda; [seed,input_row,607] binomial donation',
            donor_weight_normalization='divide frozen weights by their mean over all selected neighbors',
            donation_probability='min(1, sampled_lambda * normalized_donor_weight)',
            donation_distribution='Binomial(native_donor_count, donation_probability)',
            eligible_neighbor_transcripts_definition='sum of original unweighted donor counts at native query zeros',
            expected_added_counts_definition='sum over eligible donor/gene entries of original_count * donation_probability',
            additive_contamination=True, source_counts_depleted=False,
            whole_gene_selection=False, minimum_added_count_cutoff=False)
        config.pop('target_added_counts_definition', None)
    output.uns['synthetic_admixture_config'] = config
    output.write_h5ad(output_path, compression='gzip')
    # Check the saved layers, not just the in-memory arrays.
    with h5py.File(output_path, 'r') as handle:
        saved_clean = ad.io.read_elem(handle['layers/original_counts'])
        saved_added = ad.io.read_elem(handle['layers/synthetic_admixture_counts'])
        saved_combined = ad.io.read_elem(handle['layers/admixed_counts'])
        _same_counts(saved_clean, clean, 'Saved native counts changed')
        _same_counts(saved_added, added, 'Saved added counts changed')
        _same_counts(saved_combined, clean+added, 'Saved combined layer is inconsistent')
        _same_counts(ad.io.read_elem(handle['X']), saved_combined, 'Saved X differs from combined counts')
    validation.update(saved_native_counts_exact=True, saved_added_counts_exact=True,
        saved_combined_counts_exact=True, counts_nonnegative_integers=True,
        donor_weight_mean_one=bool(np.isclose(weights.weight.mean(), 1)),
        unsupported_weights_exactly_neutral=bool(weights.loc[~weights.supported, 'weight'].eq(1.).all()),
        donor_exposure_support_recorded=True,
        calibration_uses_training_only=not bool(set(args.calibration_slices) & set(TEST_SLICES)),
        n_cells=len(recipients), n_genes=len(genes), slice=name,
        mean_achieved_admixture_burden=float(stats.achieved_admixture_burden.mean()),
        mean_detected_genes=float(combined.getnnz(axis=1).mean()))
    if target_mode == 'pool_fraction':
        validation.update(pool_fraction_targets_exact=True, added_counts_within_available_pool=True,
            lambda_max=float(args.lambda_max), mean_achieved_pool_fraction=float(stats.achieved_pool_fraction.mean()))
    elif target_mode == 'binomial_transcripts':
        validation.update(added_counts_within_available_pool=True,
            lambda_max=float(args.lambda_max), mean_achieved_pool_fraction=float(stats.achieved_pool_fraction.mean()),
            mean_expected_added_counts=float(stats.expected_added_counts.mean()),
            n_saturated_query_donor_pairs=int(stats.n_saturated_neighbors.sum()))
    if not pilot and name == args.pilot_slice:
        pilot_path = args.output_dir/'pilot'/f'{name}.h5ad'
        pilot_obs, _, _ = read_metadata(pilot_path)
        pilot_added = read_counts(pilot_path, 'layers/synthetic_admixture_counts', genes)
        indices = obs.index.get_indexer(pilot_obs.index)
        _same_counts(added[indices], pilot_added, 'Pilot and full outputs differ for identical cells/seeds')
        validation['pilot_full_counts_identical'] = True
    stats.insert(0, 'cell_id', obs.index[recipients])
    stats.to_parquet(folder/f'{name}_cell_sampling.parquet', index=False)
    frames = []
    for counts, dataset in [(clean, CLEAN), (combined, SIMULATED)]:
        frame = expression_diagnostics(counts, selected_obs, dataset, name, genes, markers, 'cell_type')
        frame['eligible_genes_per_query'] = stats.eligible_genes_per_query.to_numpy()
        if dataset == SIMULATED:
            frame['achieved_admixture_fraction'] = stats.achieved_admixture_burden.to_numpy()
        frames.append(frame)
    real, real_obs = matched_real(args.spatial_dir/f'st_integrated_slice-{name}.h5ad', selected_obs, genes)
    frames.append(expression_diagnostics(real, real_obs, REAL, name, genes, markers, 'cell_type_harmonized'))
    if pilot:
        print('Pilot: sampling a unit-weight control with the same queries and fraction streams...', flush=True)
        control, control_stats = sampler(native, obs, genes, graph, recipients,
            {label: 1. for label in weight_map}, config, seed, batch_size=args.batch_size)
        fraction_column = 'sampled_tau' if target_mode == 'final_fraction' else 'sampled_lambda'
        if not np.array_equal(control_stats[fraction_column], stats[fraction_column]):
            raise AssertionError('Pilot/control sampled fraction streams differ')
        control_frame = expression_diagnostics(clean+control, selected_obs, 'Unit-weight control',
            name, genes, markers, 'cell_type')
        control_frame['achieved_admixture_fraction'] = control_stats.achieved_admixture_burden.to_numpy()
        control_frame['eligible_genes_per_query'] = control_stats.eligible_genes_per_query.to_numpy()
        control_frame.to_parquet(folder/'unit_weight_control_diagnostics.parquet', index=False)
        sparse.save_npz(folder/'unit_weight_control_added_counts.npz', control)
        validation['pilot_control_tau_identical' if target_mode == 'final_fraction' else 'pilot_control_lambda_identical'] = True
        validation['all_cell_types_represented'] = set(selected_obs.cell_type.astype(str)) == set(weight_map)
        validation['all_consistency_checks_passed'] = all(value for value in validation.values() if isinstance(value, bool))
        if not validation['all_consistency_checks_passed']:
            raise AssertionError('Pilot consistency checks failed; test generation is blocked')
        print(control_frame[['detected_genes', 'non_native_marker_genes', 'achieved_admixture_fraction']].mean().to_string(), flush=True)
    write_json(folder/f'{name}_validation.json', validation)
    per_cell = pd.concat(frames, ignore_index=True)
    per_cell.to_parquet(folder/f'{name}_diagnostics.parquet', index=False)
    print(f'{output_path}: mean achieved fraction={stats.achieved_admixture_burden.mean():.4f}, '
          f'mean detected genes={combined.getnnz(axis=1).mean():.2f}', flush=True)
    return per_cell, validation


def save_diagnostics(frames, output):
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.concat(frames, ignore_index=True)
    for key in ['dataset', 'slice', 'cell_type']:
        cells[key] = cells[key].astype('category')
    cells.to_parquet(output/'per_cell_diagnostics.parquet', index=False)
    summary = summarize(cells)
    summary.to_csv(output/'diagnostic_summary.csv', index=False)
    (cells.groupby(['dataset', 'cell_type'], observed=True)[METRICS[:2]].mean()
        .groupby('dataset', observed=True).mean().reindex(DATASETS)).to_csv(output/'cell_type_balanced_means.csv')
    plot_results(cells, output)
    print(summary.loc[summary.level.eq('dataset')].pivot(index='dataset', columns='metric', values='mean').to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['training', 'test-dir', 'reference', 'spatial-dir', 'markers', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--calibration-slices', nargs='+', default=CALIBRATION_SLICES)
    parser.add_argument('--pilot-slice', choices=TEST_SLICES, default='Lung12')
    parser.add_argument('--pilot-cells-per-type', type=int, default=100)
    parser.add_argument('--min-calibration-cells', type=int, default=50)
    parser.add_argument('--min-calibration-slices', type=int, default=2,
                        help='Minimum calibration slices with a qualifying exposed stratum per donor')
    parser.add_argument('--shrinkage', type=float, default=.25)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--target-mode', choices=['final_fraction', 'pool_fraction'], default='final_fraction')
    parser.add_argument('--lambda-max', type=float, default=None,
                        help='Maximum sampled donor-pool fraction; required for pool_fraction mode')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--run-all', action='store_true', help='Generate all four test slices after pilot checks pass')
    args = parser.parse_args()
    if set(args.calibration_slices) & set(TEST_SLICES) or len(set(args.calibration_slices)) != len(args.calibration_slices):
        parser.error('Calibration slices must be distinct training slices; test slices are forbidden')
    if min(args.batch_size, args.min_calibration_cells, args.min_calibration_slices, args.pilot_cells_per_type) < 1:
        parser.error('Cell counts and batch size must be positive')
    if args.min_calibration_slices > len(args.calibration_slices):
        parser.error('Minimum supported slices exceeds the number of calibration slices')
    if args.target_mode == 'pool_fraction' and (args.lambda_max is None or not 0 <= args.lambda_max <= 1):
        parser.error('pool_fraction requires --lambda-max in [0,1]')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    input_obs, input_var, config = read_metadata(args.test_dir/f'{args.pilot_slice}.h5ad')
    ref_obs, ref_var, _ = read_metadata(args.reference)
    if not ref_obs.index.is_unique:
        raise ValueError('Reference cells are duplicated')
    if set(ref_var.index) != set(input_var.index):
        parser.error('Reference and simulation panels must match')
    genes = input_var.index
    ref_counts = read_counts(args.reference, 'X', genes)
    types = sorted(ref_obs.cell_type.astype(str).unique())
    markers = select_markers(args.markers, genes, types, ref_var.gene_symbol.to_dict(),
        n_markers=None, max_padj=.05, unique_only=True)
    write_json(args.output_dir/'parameters.json', vars(args))
    weights = calibrate(args, genes, markers, config, ref_obs, ref_counts)
    if set(weights.donor_type) != set(types):
        raise ValueError('Calibrated weights omit a reference cell type')
    reference = expression_diagnostics(ref_counts, ref_obs, REFERENCE, 'Reference', genes, markers, 'cell_type')
    pilot, validation = write_slice(args, args.pilot_slice, weights, markers, ref_obs, ref_counts, pilot=True)
    save_diagnostics([reference, pilot], args.output_dir/'pilot'/'diagnostics')
    if args.run_all:
        if not validation['all_consistency_checks_passed']:
            raise AssertionError('Full test generation requires a successful pilot')
        print('Pilot passed. Generating all four test slices with frozen calibration weights.', flush=True)
        frames, validations = [reference], []
        for name in sorted(TEST_SLICES):
            frame, validation = write_slice(args, name, weights, markers, ref_obs, ref_counts)
            frames.append(frame)
            validations.append(validation)
            gc.collect()
        save_diagnostics(frames, args.output_dir/'diagnostics')
        write_json(args.output_dir/'validation.json', validations)
    write_json(args.output_dir/'run_info.json', dict(elapsed_seconds=time.perf_counter()-started,
        pilot_passed=True, full_test_generated=args.run_all, seed=args.seed))
    print(f'Completed in {time.perf_counter()-started:.1f} seconds. {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
