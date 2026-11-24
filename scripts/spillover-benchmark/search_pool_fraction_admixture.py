#!/usr/bin/env python3
"""Calibrate whole-gene donor-pool transfer on training slices, then make test data."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '8')

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from diagnose_test_admixture import (
    read_metadata, read_counts, expression_diagnostics, REFERENCE, TEST_SLICES,
)
from donor_weighted_admixture import neighbor_graph, sample_weighted
from prepare_weighted_test_admixture import (
    CALIBRATION_SLICES, file_signature, graph_path, matched_real, pilot_rows,
    write_json, write_slice, save_diagnostics,
)
from search_admixture_parameters import normalized_log_counts, rbf_bandwidth_squared

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_COLUMNS = ['detected_genes', 'non_native_marker_genes']


def squared_distances(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T, 0.)


def mmd_squared(x, metric):
    denominator = 2 * metric['bandwidth_squared']
    xx = np.exp(-squared_distances(x, x) / denominator).mean()
    xy = np.exp(-squared_distances(x, metric['target']) / denominator).mean()
    return float(max(0., xx + metric['yy_mean'] - 2*xy))


def codetection(counts, marker_indices):
    detected = (counts[:, marker_indices] > 0).toarray().astype(np.float64)
    return detected.T @ detected / len(detected)


def diagnostics(counts, obs, genes, markers, name):
    return expression_diagnostics(counts, obs, 'Simulated ST', name, genes, markers, 'cell_type')


def evaluate(counts, stats, obs, genes, markers, name, lam, seed, metrics, real_stats,
             real_codetection, cross_pairs, marker_indices):
    expression = normalized_log_counts(counts)
    cell_stats = diagnostics(counts, obs, genes, markers, name)
    rows = []
    for cell_type, metric in metrics.items():
        mask = metric['mask']
        row = dict(slice=name, cell_type=cell_type, lambda_max=lam, seed=seed,
                   is_baseline=lam == 0, n_query=int(mask.sum()),
                   mmd2=mmd_squared(expression[mask], metric),
                   mean_achieved_admixture_burden=float(stats.achieved_admixture_burden[mask].mean()))
        for column in DIAGNOSTIC_COLUMNS:
            simulated, real = cell_stats.loc[mask, column], real_stats.loc[mask, column]
            row[f'mean_{column}'] = float(simulated.mean())
            row[f'real_mean_{column}'] = float(real.mean())
            row[f'{column}_mean_error'] = float(simulated.mean() - real.mean())
            row[f'{column}_wasserstein'] = float(wasserstein_distance(simulated, real))
        rows.append(row)
    matrix = codetection(counts, marker_indices)
    simulated_marginal, real_marginal = np.diag(matrix), np.diag(real_codetection)
    covariance = matrix - np.outer(simulated_marginal, simulated_marginal)
    real_covariance = real_codetection - np.outer(real_marginal, real_marginal)
    joint = dict(slice=name, lambda_max=lam, seed=seed, is_baseline=lam == 0,
        n_query=len(obs), n_marker_pairs=int(cross_pairs.sum()),
        cross_type_codetection_mae=float(np.abs(matrix-real_codetection)[cross_pairs].mean()),
        cross_type_covariance_mae=float(np.abs(covariance-real_covariance)[cross_pairs].mean()),
        mean_cross_type_codetection=float(matrix[cross_pairs].mean()),
        real_mean_cross_type_codetection=float(real_codetection[cross_pairs].mean()),
        marker_marginal_mae=float(np.abs(simulated_marginal-real_marginal).mean()))
    return rows, joint, matrix


def freeze_weights(args):
    source = args.donor_weight_root/'calibration'
    specification = json.loads((source/'specification.json').read_text())
    if set(specification['slices']) != set(args.calibration_slices):
        raise ValueError('Frozen donor weights must use exactly the requested calibration slices')
    if set(specification['slices']) & set(TEST_SLICES):
        raise ValueError('Frozen weights must not use held-out test slices')
    expected_inputs = [args.training, args.reference, args.markers] + [
        args.spatial_dir/f'st_integrated_slice-{name}.h5ad' for name in args.calibration_slices]
    expected_signatures = {str(Path(path).resolve()): file_signature(path) for path in expected_inputs}
    for signature in specification['inputs']:
        if expected_signatures.get(signature[0]) != signature:
            raise ValueError(f'Frozen weight input/settings mismatch: {signature[0]}')
    if len(expected_signatures) != len(specification['inputs']):
        raise ValueError('Frozen weight input list is incomplete')
    destination = args.output_dir/'calibration'
    shutil.copytree(source, destination)
    weights = pd.read_csv(destination/'donor_weights.csv')
    weights['neutral_reason'] = weights.neutral_reason.fillna('')
    if not np.isfinite(weights.weight).all() or (weights.weight < 0).any() or not np.isclose(weights.weight.mean(), 1):
        raise ValueError('Frozen donor weights are invalid')
    markers = pd.read_csv(destination/'selected_markers.csv')
    if not markers.gene.is_unique:
        raise ValueError('Calibration markers must be unique to one cell type')
    args.min_calibration_cells = specification['min_cells']
    args.min_calibration_slices = specification['min_supported_slices']
    args.shrinkage = specification['shrinkage']
    args.epsilon = specification['epsilon']
    # Graph selection is unchanged; their keys include source file signatures.
    if (args.donor_weight_root/'cache').exists():
        shutil.copytree(args.donor_weight_root/'cache', args.output_dir/'cache', dirs_exist_ok=True)
    return weights, markers, specification


def search(args, weights, markers, ref_obs, genes, config, sampler=sample_weighted):
    folder = args.output_dir/'search'
    folder.mkdir()
    train_obs, train_var, _ = read_metadata(args.training)
    if set(train_var.index) != set(genes):
        raise ValueError('Training and reference must use the same complete gene panel')
    training = read_counts(args.training, 'X', genes)
    weight_map = dict(zip(weights.donor_type, weights.weight))
    marker_indices = genes.get_indexer(markers.gene)
    if (marker_indices < 0).any():
        raise ValueError('Calibration markers are absent from the gene panel')
    marker_types = markers.cell_type.to_numpy()
    cross_pairs = np.triu(marker_types[:, None] != marker_types[None, :], k=1)
    reference_support = ref_obs.cell_type.astype(str).value_counts()
    stratum_rows, joint_rows, kernel_rows, support_rows, query_frames = [], [], [], [], []
    # Include zero as an explicit diagnostic baseline; select among the positive grid.
    lambdas = [0.] + sorted(args.lambda_max_values)
    for name in args.calibration_slices:
        source_rows = np.flatnonzero(train_obs.spatial_slice.astype(str).eq(name) &
                                    train_obs.split.astype(str).eq('train'))
        if not len(source_rows):
            raise ValueError(f'No training-only mapped cells for calibration slice {name}')
        obs, native = train_obs.iloc[source_rows].copy(), training[source_rows]
        real, _ = matched_real(args.spatial_dir/f'st_integrated_slice-{name}.h5ad', obs, genes)
        graph = neighbor_graph(native, obs, config, graph_path(args.output_dir, args.training, name, config))
        available_types = obs.cell_type.astype(str).value_counts()
        supported = sorted(ct for ct, n in available_types.items()
                           if n >= args.min_evaluation_cells and reference_support.get(ct, 0) >= args.min_evaluation_cells)
        for ct, n in available_types.items():
            support_rows.append(dict(slice=name, cell_type=ct, n_mapped_real_pairs=int(n),
                n_reference_cells=int(reference_support.get(ct, 0)), supported=ct in supported))
        recipients = pilot_rows(obs, args.max_query_per_cell_type, args.query_seed)
        recipients = recipients[obs.cell_type.iloc[recipients].astype(str).isin(supported)]
        if not len(recipients):
            raise ValueError(f'No supported evaluation cell types in {name}')
        queries = obs.iloc[recipients].copy()
        clean, target = native[recipients], real[recipients]
        query_frame = queries[['cell_type', 'matched_st_id', 'source_sc_id']].copy()
        query_frame.insert(0, 'cell_id', queries.index.astype(str))
        query_frame.insert(0, 'slice', name)
        query_frames.append(query_frame.reset_index(drop=True))
        base_expression, target_expression = normalized_log_counts(clean), normalized_log_counts(target)
        metrics = {}
        for ct in supported:
            mask = queries.cell_type.astype(str).eq(ct).to_numpy()
            bandwidth = rbf_bandwidth_squared(base_expression[mask], target_expression[mask])
            yy = np.exp(-squared_distances(target_expression[mask], target_expression[mask]) / (2*bandwidth)).mean()
            metrics[ct] = dict(mask=mask, target=target_expression[mask], bandwidth_squared=bandwidth, yy_mean=yy)
            kernel_rows.append(dict(slice=name, cell_type=ct, n_query=int(mask.sum()),
                bandwidth_squared=bandwidth, kernel='RBF exp(-squared_distance / (2 * bandwidth_squared))',
                preprocessing='CP10K followed by log1p; all shared genes', n_genes=len(genes)))
        real_stats = diagnostics(target, queries, genes, markers, name)
        real_joint = codetection(target, marker_indices)
        print(f'{name}: {len(recipients):,} fixed queries; {len(supported)} supported cell types; fixed kernels', flush=True)
        for lam in lambdas:
            for seed in args.seeds:
                if lam == 0:
                    combined = clean
                    stats = pd.DataFrame({'achieved_admixture_burden': np.zeros(len(queries))})
                else:
                    candidate_config = dict(config,
                        admixture_target_mode=config.get('admixture_target_mode', 'pool_fraction'), lambda_max=lam)
                    candidate_config.pop('tau_range', None)
                    added, stats = sampler(native, obs, genes, graph, recipients, weight_map,
                        candidate_config, seed, batch_size=args.batch_size)
                    combined = clean + added
                    stats.insert(0, 'cell_id', obs.index[recipients])
                    stats.insert(0, 'cell_type', queries.cell_type.astype(str).to_numpy())
                    stats.to_parquet(folder/f'{name}_lambda_{lam:g}_seed_{seed}_sampling.parquet', index=False)
                rows, joint, matrix = evaluate(combined, stats, queries, genes, markers, name,
                    lam, seed, metrics, real_stats, real_joint, cross_pairs, marker_indices)
                stratum_rows.extend(rows)
                joint_rows.append(joint)
                np.savez_compressed(folder/f'{name}_lambda_{lam:g}_seed_{seed}_codetection.npz',
                    simulated=matrix, real=real_joint, gene_ids=markers.gene.to_numpy(dtype=str),
                    marker_types=marker_types.astype(str), cross_type_pairs=cross_pairs)
                print(f'  lambda_max={lam:g}, seed={seed}: mean MMD²={np.mean([r["mmd2"] for r in rows]):.6f}', flush=True)
            # Write partial results after each candidate so progress remains inspectable.
            pd.DataFrame(stratum_rows).to_csv(folder/'stratum_metrics.csv', index=False)
            pd.DataFrame(joint_rows).to_csv(folder/'codetection_metrics.csv', index=False)
        del native, real, graph, base_expression, target_expression, metrics
        gc.collect()
    pd.DataFrame(kernel_rows).to_csv(folder/'fixed_mmd_kernels.csv', index=False)
    pd.DataFrame(support_rows).to_csv(folder/'evaluation_support.csv', index=False)
    pd.concat(query_frames, ignore_index=True).to_parquet(folder/'fixed_query_cells.parquet', index=False)
    strata, joints = pd.DataFrame(stratum_rows), pd.DataFrame(joint_rows)
    # Explicit hierarchical averaging: types -> slices -> simulation seeds.
    slice_scores = strata.groupby(['lambda_max', 'seed', 'slice'], as_index=False).mmd2.mean()
    seed_scores = slice_scores.groupby(['lambda_max', 'seed'], as_index=False).mmd2.mean()
    summary = seed_scores.groupby('lambda_max').agg(mean_mmd2=('mmd2', 'mean'),
        sd_mmd2_across_seeds=('mmd2', 'std'), n_seeds=('seed', 'nunique')).reset_index()
    slice_scores.to_csv(folder/'mmd_slice_scores.csv', index=False)
    seed_scores.to_csv(folder/'mmd_seed_scores.csv', index=False)
    summary['is_baseline'] = summary.lambda_max.eq(0)
    summary['candidate_rank'] = summary.loc[~summary.is_baseline, 'mean_mmd2'].rank(method='min')
    summary.to_csv(folder/'lambda_search_summary.csv', index=False)
    diagnostic_columns = [c for c in strata if c.endswith('_wasserstein') or
        c.endswith('_mean_error') or c.startswith('mean_') or c.startswith('real_mean_')]
    diagnostic_columns = [c for c in diagnostic_columns if c != 'mean_mmd2']
    separate = strata.groupby(['lambda_max', 'seed', 'slice'])[diagnostic_columns].mean()
    separate = separate.groupby(['lambda_max', 'seed']).mean().groupby('lambda_max').mean().reset_index()
    joint_columns = ['cross_type_codetection_mae', 'cross_type_covariance_mae',
        'mean_cross_type_codetection', 'real_mean_cross_type_codetection', 'marker_marginal_mae']
    joint_summary = joints.groupby(['lambda_max', 'seed'])[joint_columns].mean().groupby('lambda_max').mean().reset_index()
    separate = separate.merge(joint_summary, on='lambda_max')
    separate.to_csv(folder/'separate_diagnostic_summary.csv', index=False)
    best = summary.loc[~summary.is_baseline].sort_values(['mean_mmd2', 'lambda_max']).iloc[0]
    baseline = float(summary.loc[summary.is_baseline, 'mean_mmd2'].iloc[0])
    selection = dict(lambda_max=float(best.lambda_max), mean_mmd2=float(best.mean_mmd2),
        baseline_mmd2=baseline, improves_over_no_admixture=bool(best.mean_mmd2 < baseline),
        selection_criterion='cell-type-conditional biased RBF MMD squared',
        averaging='equal supported cell types within slice, then equal slices, then equal seeds',
        candidate_lambda_max=sorted(args.lambda_max_values), no_admixture_baseline=0.,
        baseline_role='reported separately; lambda selected from the requested positive grid',
        calibration_slices=args.calibration_slices, seeds=args.seeds,
        preprocessing='CP10K then log1p; all shared genes',
        kernel='RBF with fixed per-slice/type bandwidth from clean+real counts, reused across all candidates/seeds',
        uses_held_out_test_for_selection=False)
    if config.get('admixture_target_mode') == 'binomial_transcripts':
        selection.update(sampler='binomial_transcripts',
            donor_weight_normalization='mean one over each query selected neighborhood',
            donation_probability='min(1, sampled_lambda * normalized_donor_weight)',
            minimum_added_count_cutoff=False)
    write_json(folder/'selected_parameters.json', selection)
    plot_search(summary, separate, folder)
    print(summary.sort_values('mean_mmd2').to_string(index=False), flush=True)
    print('Selected:', json.dumps(selection), flush=True)
    return selection


def plot_search(summary, diagnostics_table, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.titlesize': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.4), constrained_layout=True)
    data = summary.merge(diagnostics_table, on='lambda_max').sort_values('lambda_max')
    for ax, column, title in zip(axes,
        ['mean_mmd2', 'detected_genes_wasserstein', 'non_native_marker_genes_wasserstein', 'cross_type_codetection_mae'],
        ['Conditional MMD²', 'Gene detection', 'Non-native markers', 'Cross-type co-detection']):
        ax.plot(data.lambda_max, data[column], marker='o', markersize=3, linewidth=1)
        ax.set(title=title, xlabel=r'$\lambda_{\max}$', ylabel='MMD²' if column == 'mean_mmd2' else
               ('Mean absolute error' if column.endswith('_mae') else 'Wasserstein distance'))
        ax.set_xticks([0., .25, .5, .75, 1.])
        ax.spines[['top', 'right']].set_visible(False)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(folder/f'lambda_search_diagnostics.{extension}', dpi=400, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx'))
    parser.add_argument('--donor-weight-root', type=Path, default=ROOT/'outputs/SyndriSIM_donor_weighted_exposed_20261120')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/SyndriSIM_pool_fraction')
    parser.add_argument('--calibration-slices', nargs='+', default=CALIBRATION_SLICES)
    parser.add_argument('--lambda-max-values', nargs='+', type=float, default=[.1, .25, .5, .75, 1.])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--max-query-per-cell-type', type=int, default=500)
    parser.add_argument('--min-evaluation-cells', type=int, default=50)
    parser.add_argument('--query-seed', type=int, default=91827)
    parser.add_argument('--seed', type=int, default=20261120, help='Final pilot/test generation seed')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pilot-slice', choices=TEST_SLICES, default='Lung12')
    parser.add_argument('--pilot-cells-per-type', type=int, default=100)
    parser.add_argument('--run-all', action='store_true', help='Generate all four test slices after the pilot passes')
    args = parser.parse_args()
    if (not args.lambda_max_values or len(set(args.lambda_max_values)) != len(args.lambda_max_values)
            or any(not np.isfinite(x) or not 0 < x <= 1 for x in args.lambda_max_values)):
        parser.error('Supply distinct positive lambda_max values <=1; zero baseline is automatic')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Supply distinct nonnegative simulation seeds')
    if not args.calibration_slices or len(set(args.calibration_slices)) != len(args.calibration_slices) or set(args.calibration_slices) & set(TEST_SLICES):
        parser.error('Use distinct training/calibration slices only')
    if args.min_evaluation_cells < 2 or args.max_query_per_cell_type < args.min_evaluation_cells:
        parser.error('Query cap must be >= minimum evaluation cells, which must be >=2')
    if min(args.batch_size, args.pilot_cells_per_type) < 1:
        parser.error('Batch size and pilot size must be positive')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose another directory to preserve previous runs')
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    args.training = args.data_root/'spillover-benchmark/train_val.h5ad'
    args.test_dir = args.data_root/'spillover-benchmark/test_admixture/slices'
    args.reference = args.data_root/'spillover-benchmark/test_admixture/train_val_sc_reference_unique.h5ad'
    args.spatial_dir = args.data_root/'results/final'
    args.markers = args.data_root/'results/markers/pydeseq2/all_cell_type_vs_rest_deseq2.csv'
    args.target_mode = 'pool_fraction'
    weights, markers, donor_specification = freeze_weights(args)
    ref_obs, ref_var, _ = read_metadata(args.reference)
    genes = ref_var.index
    ref_counts = read_counts(args.reference, 'X', genes)
    if not set(ref_obs.cell_type.astype(str)).issubset(set(weights.donor_type)):
        raise ValueError('Missing weights for reference cell types')
    # Define search settings from training-calibrated metadata, without opening a test input.
    config = dict(donor_specification['neighbor_config'], only_sample_unexpressed_genes=True,
                  min_sampled_transcript_threshold=10)
    write_json(args.output_dir/'parameters.json', vars(args))
    write_json(args.output_dir/'provenance.json', dict(
        donor_weight_source=str(args.donor_weight_root.resolve()),
        frozen_weight_sha256=hashlib.sha256((args.output_dir/'calibration/donor_weights.csv').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampler_sha256=hashlib.sha256(Path(__file__).with_name('donor_weighted_admixture.py').read_bytes()).hexdigest()))
    selection = search(args, weights, markers, ref_obs, genes, config)
    args.lambda_max = selection['lambda_max']
    write_json(args.output_dir/'parameters.json', vars(args))
    # Selection is frozen before the first test slice is read.
    reference = expression_diagnostics(ref_counts, ref_obs, REFERENCE, 'Reference', genes, markers, 'cell_type')
    pilot, validation = write_slice(args, args.pilot_slice, weights, markers, ref_obs, ref_counts, pilot=True)
    save_diagnostics([reference, pilot], args.output_dir/'pilot/diagnostics')
    if args.run_all:
        if not validation['all_consistency_checks_passed']:
            raise AssertionError('A successful pilot is required before full test generation')
        frames, validations = [reference], []
        for name in sorted(TEST_SLICES):
            frame, report = write_slice(args, name, weights, markers, ref_obs, ref_counts)
            frames.append(frame)
            validations.append(report)
            gc.collect()
        save_diagnostics(frames, args.output_dir/'diagnostics')
        write_json(args.output_dir/'validation.json', validations)
    write_json(args.output_dir/'run_info.json', dict(elapsed_seconds=time.perf_counter()-started,
        selected_lambda_max=args.lambda_max, pilot_passed=True, full_test_generated=args.run_all,
        uses_test_for_parameter_selection=False))
    print(f'Completed in {time.perf_counter()-started:.1f}s: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
