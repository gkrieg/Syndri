#!/usr/bin/env python3
"""Calibrate and compare independent transcript donation without replacing data."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '8')

import numpy as np
import pandas as pd

from diagnose_test_admixture import read_metadata, read_counts, expression_diagnostics, REFERENCE, TEST_SLICES, _same_counts
from binomial_transcript_admixture import sample_binomial, validate_binomial_allocations
from prepare_weighted_test_admixture import (
    CALIBRATION_SLICES, write_json, write_slice, save_diagnostics, matched_real, pilot_rows,
)
from search_pool_fraction_admixture import (
    ROOT, freeze_weights, search, evaluate, diagnostics, codetection, squared_distances,
)
from search_admixture_parameters import normalized_log_counts, rbf_bandwidth_squared

BINOMIAL = 'Binomial transcripts (local weights)'
WHOLE_GENE = 'Whole genes (global weights)'


def check_matched_design(args):
    """Require identical queries, kernels, support, and frozen donor weights."""
    old, new = args.whole_gene_root, args.output_dir
    queries = pd.read_parquet(old/'search/fixed_query_cells.parquet')
    pd.testing.assert_frame_equal(queries, pd.read_parquet(new/'search/fixed_query_cells.parquet'))
    kernels = pd.read_csv(old/'search/fixed_mmd_kernels.csv')
    pd.testing.assert_frame_equal(kernels, pd.read_csv(new/'search/fixed_mmd_kernels.csv'),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(pd.read_csv(old/'search/evaluation_support.csv'),
                                  pd.read_csv(new/'search/evaluation_support.csv'))
    pd.testing.assert_frame_equal(pd.read_csv(old/'calibration/donor_weights.csv'),
                                  pd.read_csv(new/'calibration/donor_weights.csv'))
    before = pd.read_csv(old/'search/stratum_metrics.csv').query('lambda_max == 0')
    after = pd.read_csv(new/'search/stratum_metrics.csv').query('lambda_max == 0')
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    return dict(fixed_queries_identical=True, fixed_kernels_identical=True,
                support_identical=True, frozen_donor_weights_identical=True,
                no_admixture_baselines_identical=True)


def calibration_comparison(args):
    folder = args.output_dir/'comparison'
    folder.mkdir()
    report = check_matched_design(args)
    records, selected_rows = [], []
    for simulator, root in [(WHOLE_GENE, args.whole_gene_root), (BINOMIAL, args.output_dir)]:
        selection = json.loads((root/'search/selected_parameters.json').read_text())
        mmd = pd.read_csv(root/'search/lambda_search_summary.csv')
        other = pd.read_csv(root/'search/separate_diagnostic_summary.csv')
        data = mmd.merge(other, on='lambda_max', validate='one_to_one')
        data.insert(0, 'simulator', simulator)
        data['selected'] = np.isclose(data.lambda_max, selection['lambda_max'])
        records.append(data)
        selected_rows.append(data.loc[data.selected])
    table = pd.concat(records, ignore_index=True)
    selected = pd.concat(selected_rows, ignore_index=True)
    table.to_csv(folder/'calibration_all_candidates.csv', index=False)
    selected.to_csv(folder/'calibration_selected_comparison.csv', index=False)
    report.update(whole_gene_root=str(args.whole_gene_root.resolve()),
        weight_normalization_differs=True,
        comparison_note='Requested binomial sampler normalizes weights per neighborhood; existing whole-gene sampler uses global weights. Changes cannot be attributed to transcript sampling alone.',
        uses_held_out_test_for_selection=False)
    write_json(folder/'matched_design_validation.json', report)
    plot_comparison(table, folder/'calibration_comparison', search_grid=True)
    print('Selected calibration comparison:', flush=True)
    print(selected[['simulator', 'lambda_max', 'mean_mmd2', 'detected_genes_wasserstein',
        'non_native_marker_genes_wasserstein', 'cross_type_codetection_mae']].to_string(index=False), flush=True)
    return selected


def held_out_comparison(args, markers, ref_obs):
    """Post-selection comparison with the same fixed recipients and clean kernels."""
    folder = args.output_dir/'comparison'
    reference_support = ref_obs.cell_type.astype(str).value_counts()
    stratum_rows, joint_rows, kernel_rows = [], [], []
    for name in sorted(TEST_SLICES):
        new_path = args.output_dir/'slices'/f'{name}.h5ad'
        old_path = args.whole_gene_root/'slices'/f'{name}.h5ad'
        obs, var, new_config = read_metadata(new_path)
        old_obs, old_var, old_config = read_metadata(old_path)
        pd.testing.assert_frame_equal(obs.drop(columns='achieved_admixture_burden'),
                                      old_obs.drop(columns='achieved_admixture_burden'))
        pd.testing.assert_frame_equal(var, old_var)
        native = read_counts(new_path, 'layers/original_counts', var.index)
        old_native = read_counts(old_path, 'layers/original_counts', var.index)
        _same_counts(native, old_native, 'Binomial experiment changed the native matrices')
        del old_native
        recipients = pilot_rows(obs, args.max_query_per_cell_type, args.query_seed)
        available = obs.cell_type.astype(str).value_counts()
        supported = sorted(ct for ct, n in available.items() if n >= args.min_evaluation_cells
                           and reference_support.get(ct, 0) >= args.min_evaluation_cells)
        recipients = recipients[obs.cell_type.iloc[recipients].astype(str).isin(supported)]
        queries, clean = obs.iloc[recipients], native[recipients]
        target, _ = matched_real(args.spatial_dir/f'st_integrated_slice-{name}.h5ad', queries, var.index)
        x0, y = normalized_log_counts(clean), normalized_log_counts(target)
        metrics = {}
        for ct in supported:
            mask = queries.cell_type.astype(str).eq(ct).to_numpy()
            bandwidth = rbf_bandwidth_squared(x0[mask], y[mask])
            metrics[ct] = dict(mask=mask, target=y[mask], bandwidth_squared=bandwidth,
                yy_mean=np.exp(-squared_distances(y[mask], y[mask])/(2*bandwidth)).mean())
            kernel_rows.append(dict(slice=name, cell_type=ct, n_query=int(mask.sum()),
                                    bandwidth_squared=bandwidth))
        genes = var.index
        indices = genes.get_indexer(markers.gene)
        cross_pairs = np.triu(markers.cell_type.to_numpy()[:, None] != markers.cell_type.to_numpy()[None, :], k=1)
        real_stats = diagnostics(target, queries, genes, markers, name)
        real_joint = codetection(target, indices)
        for simulator, path, config in [('No admixture', None, new_config),
                (WHOLE_GENE, old_path, old_config), (BINOMIAL, new_path, new_config)]:
            combined = clean if path is None else read_counts(path, 'layers/admixed_counts', genes, recipients)
            stats = pd.DataFrame({'achieved_admixture_burden':
                np.zeros(len(queries)) if path is None else
                read_metadata(path)[0].achieved_admixture_burden.iloc[recipients].to_numpy()})
            lam = 0. if path is None else float(config['lambda_max'])
            rows, joint, matrix = evaluate(combined, stats, queries, genes, markers, name,
                lam, int(config['seed']), metrics, real_stats, real_joint, cross_pairs, indices)
            stratum_rows.extend(dict(row, simulator=simulator) for row in rows)
            joint_rows.append(dict(joint, simulator=simulator))
            np.savez_compressed(folder/f'{name}_{"baseline" if path is None else "binomial" if simulator == BINOMIAL else "whole_gene"}_codetection.npz',
                simulated=matrix, real=real_joint, gene_ids=markers.gene.to_numpy(dtype=str),
                marker_types=markers.cell_type.to_numpy(dtype=str), cross_type_pairs=cross_pairs)
        del native, target, x0, y, metrics, combined
        gc.collect()
        print(f'Held-out comparison: {name}; {len(recipients):,} matched queries', flush=True)
    strata, joints = pd.DataFrame(stratum_rows), pd.DataFrame(joint_rows)
    strata.to_csv(folder/'test_stratum_metrics.csv', index=False)
    joints.to_csv(folder/'test_codetection_metrics.csv', index=False)
    pd.DataFrame(kernel_rows).to_csv(folder/'test_fixed_mmd_kernels.csv', index=False)
    columns = ['mmd2', 'detected_genes_wasserstein', 'non_native_marker_genes_wasserstein',
        'detected_genes_mean_error', 'non_native_marker_genes_mean_error',
        'mean_detected_genes', 'real_mean_detected_genes', 'mean_non_native_marker_genes',
        'real_mean_non_native_marker_genes', 'mean_achieved_admixture_burden']
    by_slice = strata.groupby(['simulator', 'slice'], as_index=False)[columns].mean()
    joint_columns = ['cross_type_codetection_mae', 'cross_type_covariance_mae', 'marker_marginal_mae']
    by_slice = by_slice.merge(joints[['simulator', 'slice']+joint_columns], on=['simulator', 'slice'])
    by_slice.to_csv(folder/'test_slice_comparison.csv', index=False)
    result = by_slice.groupby('simulator', as_index=False).mean(numeric_only=True).rename(columns={'mmd2': 'mean_mmd2'})
    result.to_csv(folder/'test_selected_comparison.csv', index=False)
    plot_comparison(result, folder/'test_comparison', search_grid=False)
    print('Held-out test comparison (post-selection only):', flush=True)
    print(result[['simulator', 'mean_mmd2', 'detected_genes_wasserstein',
        'non_native_marker_genes_wasserstein', 'cross_type_codetection_mae']].to_string(index=False), flush=True)
    return result


def plot_comparison(data, path, search_grid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.titlesize': 10, 'svg.fonttype': 'none'})
    colors = {WHOLE_GENE: '#7b3294', BINOMIAL: '#008837', 'No admixture': '#999999'}
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.5))
    for ax, column, title in zip(axes,
        ['mean_mmd2', 'detected_genes_wasserstein', 'non_native_marker_genes_wasserstein', 'cross_type_codetection_mae'],
        ['Conditional MMD²', 'Gene detection', 'Non-native markers', 'Cross-type co-detection']):
        if search_grid:
            for simulator, group in data.groupby('simulator', sort=False):
                group = group.sort_values('lambda_max')
                ax.plot(group.lambda_max, group[column], marker='o', markersize=2.5,
                    linewidth=1., color=colors[simulator], label=simulator)
                selected = group.loc[group.selected]
                ax.scatter(selected.lambda_max, selected[column], marker='*', s=45,
                           color=colors[simulator], zorder=4)
            ax.set_xlabel(r'$\lambda_{\max}$')
            if data.loc[data.lambda_max.gt(0), 'lambda_max'].min() < .1:
                ax.set_xscale('symlog', linthresh=.0001)
                ax.set_xticks([0., .0001, .001, .01, .1, 1.], ['0', '10⁻⁴', '10⁻³', '.01', '.1', '1'])
            else:
                ax.set_xticks([0., .25, .5, .75, 1.])
        else:
            order = ['No admixture', WHOLE_GENE, BINOMIAL]
            values = data.set_index('simulator').loc[order, column]
            ax.bar(np.arange(3), values, color=[colors[key] for key in order], width=.65)
            ax.set_xticks(np.arange(3), ['Clean', 'Whole gene', 'Binomial'], rotation=35, ha='right')
        ax.set_title(title)
        ax.set_ylabel('MMD²' if column == 'mean_mmd2' else 'Mean absolute error' if column.endswith('_mae') else 'Wasserstein distance')
        ax.spines[['top', 'right']].set_visible(False)
    if search_grid:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
                   frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .15 if search_grid else 0, 1, 1))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.'+extension), dpi=400, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx'))
    parser.add_argument('--donor-weight-root', type=Path, default=ROOT/'outputs/SyndriSIM_donor_weighted_exposed_20261120')
    parser.add_argument('--whole-gene-root', type=Path, default=ROOT/'outputs/SyndriSIM_pool_fraction')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/SyndriSIM_binomial_transcripts')
    parser.add_argument('--calibration-slices', nargs='+', default=CALIBRATION_SLICES)
    parser.add_argument('--lambda-max-values', nargs='+', type=float,
        default=[.0001, .0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, .75, 1.])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--max-query-per-cell-type', type=int, default=500)
    parser.add_argument('--min-evaluation-cells', type=int, default=50)
    parser.add_argument('--query-seed', type=int, default=91827)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pilot-slice', choices=TEST_SLICES, default='Lung12')
    parser.add_argument('--pilot-cells-per-type', type=int, default=100)
    parser.add_argument('--run-all', action='store_true')
    args = parser.parse_args()
    if (not args.lambda_max_values or len(set(args.lambda_max_values)) != len(args.lambda_max_values)
            or any(not np.isfinite(x) or not 0 < x <= 1 for x in args.lambda_max_values)):
        parser.error('Supply distinct positive lambda_max values <=1; zero baseline is automatic')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Supply distinct nonnegative seeds')
    if (set(args.calibration_slices) & set(TEST_SLICES) or
            len(set(args.calibration_slices)) != len(args.calibration_slices)):
        parser.error('Only distinct training/calibration slices may select parameters')
    if min(args.batch_size, args.pilot_cells_per_type) < 1 or args.min_evaluation_cells < 2 or args.max_query_per_cell_type < args.min_evaluation_cells:
        parser.error('Invalid batch, pilot, or evaluation sizes')
    if args.output_dir.exists():
        parser.error('Output directory exists; choose a fresh directory')
    if not (args.whole_gene_root/'run_info.json').exists():
        parser.error('Whole-gene comparison must be a completed calibrated run')
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    args.training = args.data_root/'spillover-benchmark/train_val.h5ad'
    args.test_dir = args.data_root/'spillover-benchmark/test_admixture/slices'
    args.reference = args.data_root/'spillover-benchmark/test_admixture/train_val_sc_reference_unique.h5ad'
    args.spatial_dir = args.data_root/'results/final'
    args.markers = args.data_root/'results/markers/pydeseq2/all_cell_type_vs_rest_deseq2.csv'
    args.target_mode = 'binomial_transcripts'
    weights, markers, donor_specification = freeze_weights(args)
    ref_obs, ref_var, _ = read_metadata(args.reference)
    genes = ref_var.index
    ref_counts = read_counts(args.reference, 'X', genes)
    config = dict(donor_specification['neighbor_config'], only_sample_unexpressed_genes=True,
        admixture_target_mode='binomial_transcripts', min_sampled_transcript_threshold=0)
    if config['k_neighbors'] != 9:
        raise ValueError('This controlled comparison requires the existing nine-neighbor graph')
    write_json(args.output_dir/'parameters.json', vars(args))
    code_paths = [Path(__file__), Path(__file__).with_name('binomial_transcript_admixture.py'),
        Path(__file__).with_name('search_pool_fraction_admixture.py'),
        Path(__file__).with_name('prepare_weighted_test_admixture.py')]
    write_json(args.output_dir/'provenance.json', dict(
        donor_weight_source=str(args.donor_weight_root.resolve()),
        frozen_weight_sha256=hashlib.sha256((args.output_dir/'calibration/donor_weights.csv').read_bytes()).hexdigest(),
        scripts={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}))
    selection = search(args, weights, markers, ref_obs, genes, config, sampler=sample_binomial)
    args.lambda_max = selection['lambda_max']
    write_json(args.output_dir/'parameters.json', vars(args))
    calibration_comparison(args)
    # Test inputs are opened only after selection and fixed-design checks succeed.
    reference = expression_diagnostics(ref_counts, ref_obs, REFERENCE, 'Reference', genes, markers, 'cell_type')
    writer_args = dict(sampler=sample_binomial, allocation_validator=validate_binomial_allocations)
    pilot, validation = write_slice(args, args.pilot_slice, weights, markers, ref_obs,
                                    ref_counts, pilot=True, **writer_args)
    save_diagnostics([reference, pilot], args.output_dir/'pilot/diagnostics')
    if args.run_all:
        if not validation['all_consistency_checks_passed']:
            raise AssertionError('Pilot must pass before generating full test data')
        frames, validations = [reference], []
        for name in sorted(TEST_SLICES):
            frame, report = write_slice(args, name, weights, markers, ref_obs, ref_counts, **writer_args)
            frames.append(frame)
            validations.append(report)
            gc.collect()
        save_diagnostics(frames, args.output_dir/'diagnostics')
        write_json(args.output_dir/'validation.json', validations)
        held_out_comparison(args, markers, ref_obs)
    write_json(args.output_dir/'run_info.json', dict(elapsed_seconds=time.perf_counter()-started,
        selected_lambda_max=args.lambda_max, pilot_passed=True, full_test_generated=args.run_all,
        sampler='binomial_transcripts', original_whole_gene_outputs_unchanged=True,
        uses_test_for_parameter_selection=False))
    print(f'Completed in {time.perf_counter()-started:.1f}s: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
