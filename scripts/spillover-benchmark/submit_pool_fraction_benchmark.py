#!/usr/bin/env python3
"""Submit the original Figure 3 method recipes on whole-gene lambda=0.25 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT/'outputs/SyndriSIM_pool_fraction'
DATA = Path('/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx')
SLICES = ['Lung12', 'Lung13', 'Lung9_Rep1', 'Lung9_Rep2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['competitors', 'rctd-split', 'syndri', 'figures'], required=True)
    args = parser.parse_args()
    folder = BENCHMARK/'manifests'
    folder.mkdir(exist_ok=True)
    record_path = folder/'method_jobs.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {'jobs': {}}

    def save():
        temporary = record_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(record_path)

    def submit(key, script, exports, *options):
        if key in record['jobs']:
            raise ValueError(f'{key} was already submitted as job {record["jobs"][key]["job_id"]}')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in exports.items()})
        command = ['sbatch', '--parsable', '--export=ALL', *map(str, options), str(script)]
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise ValueError(f'Unexpected sbatch response: {result.stdout}')
        record['jobs'][key] = {'job_id': job, 'command': command,
            'explicit_environment': {key: str(value) for key, value in exports.items()}}
        save()
        print(f'{key}: {job}', flush=True)
        return job

    selected = json.loads((BENCHMARK/'search/selected_parameters.json').read_text())
    if selected['lambda_max'] != .25 or selected['uses_held_out_test_for_selection']:
        raise ValueError('Expected the frozen training-selected whole-gene lambda=0.25 simulation')
    reports = json.loads((BENCHMARK/'validation.json').read_text())
    if sorted(row['slice'] for row in reports) != SLICES:
        raise ValueError('Expected all four validated test slices')
    for report in reports:
        if not all(value for value in report.values() if isinstance(value, bool)):
            raise ValueError(f'Input consistency failure: {report["slice"]}')
    paths = [BENCHMARK/'slices'/f'{name}.h5ad' for name in SLICES]
    reference = DATA/'spillover-benchmark/test_admixture/train_val_sc_reference_unique.h5ad'
    train_val = DATA/'spillover-benchmark/train_val.h5ad'
    for path in paths+[reference, train_val]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = folder/'test_slices.txt'
    manifest.write_text(''.join(f'{path}\n' for path in paths))
    rctd_manifest = folder/'rctd_outputs.txt'
    rctd_manifest.write_text(''.join(f'{BENCHMARK / "rctd" / (name+".h5ad")}\n' for name in SLICES))
    record.update(benchmark_root=str(BENCHMARK), train_val=str(train_val),
        unique_train_val_reference=str(reference), test_manifest=str(manifest),
        input_simulator='donor_type_weighted_pool_fraction', lambda_max=.25,
        frozen_weights_sha256=hashlib.sha256((BENCHMARK/'calibration/donor_weights.csv').read_bytes()).hexdigest(),
        original_figure3_model_config=str(ROOT/'Heimdall/admixture_prediction_results/nsclc_cosmx_train_val_k9_tau0_0p95/config.txt'),
        syndri_checkpoint_milestone=29, syndri_retrained=True,
        resolvi={'epochs': 100, 'posterior_samples': 3, 'seed': 0, 'scvi_version': '1.3.3'},
        rctd={'mode': 'doublet', 'reference_min_umi': 10, 'reference_max_umi': 2000,
              'cell_min': 25, 'n_max_cells': 10000, 'umi_min_sigma': 100},
        split={'input_layer': 'X', 'reference_layer': 'X', 'reference_cell_type_col': 'cell_type',
               'chunk_size': 50000, 'keep_zero_weight_cells': False, 'fill_missing': 'original'})
    save()
    logs = ROOT/'logs'
    benchmark_scripts = ROOT/'scripts/spillover-benchmark'
    common_env = dict(OMP_NUM_THREADS=8, OPENBLAS_NUM_THREADS=8, MKL_NUM_THREADS=8, NUMBA_NUM_THREADS=8)
    if args.stage in {'competitors', 'rctd-split'}:
        for method in (['rctd', 'resolvi', 'split/merged_h5ad'] if args.stage == 'competitors'
                       else ['rctd', 'split/merged_h5ad']):
            for name in SLICES:
                path = BENCHMARK/method/f'{name}.h5ad'
                if path.exists():
                    raise FileExistsError(f'Preserving existing result: {path}')
        rctd = submit('rctd', benchmark_scripts/'run_pool_fraction_rctd.sbatch', dict(common_env,
            SPILLOVER_TEST_MANIFEST=manifest, SPILLOVER_SC_REFERENCE=reference,
            SPILLOVER_RCTD_OUTPUT_DIR=BENCHMARK/'rctd'), '--array=0-3%2',
            '--job-name=poolfraction-rctd', f'--output={logs}/poolfraction-rctd-%A_%a.out',
            f'--error={logs}/poolfraction-rctd-%A_%a.err')
        submit('split', benchmark_scripts/'run_pool_fraction_split.sbatch', dict(common_env,
            INPUT_LIST_FILE=rctd_manifest, INPUT_DIR=BENCHMARK/'rctd', INPUT_GLOB='*.h5ad',
            INPUT_LAYERS='X', REFERENCE_H5AD=reference, REFERENCE_LAYER='X',
            REFERENCE_CELL_TYPE_COL='cell_type', OUTPUT_ROOT=BENCHMARK/'split',
            INTERMEDIATE_ROOT=BENCHMARK/'split/intermediate', SPLIT_OUTPUT_ROOT=BENCHMARK/'split/split_output',
            MERGED_OUTPUT_DIR=BENCHMARK/'split/merged_h5ad', RESULT_LAYER_PREFIX='split_purified_counts',
            SPLIT_CHUNK_SIZE=50000, KEEP_ZERO_WEIGHT_CELLS=0, FILL_MISSING='original', MERGE_INPLACE=0),
            '--array=0-3%2', f'--dependency=aftercorr:{rctd}', '--job-name=poolfraction-split',
            f'--output={logs}/poolfraction-split-%A_%a.out', f'--error={logs}/poolfraction-split-%A_%a.err')
        if args.stage == 'competitors':
            submit('resolvi', benchmark_scripts/'run_pool_fraction_resolvi.sbatch', dict(common_env,
                SPILLOVER_TEST_MANIFEST=manifest, SPILLOVER_RESOLVI_OUTPUT_ROOT=BENCHMARK,
                RESOLVI_EPOCHS=100, RESOLVI_SAMPLES=3), '--array=0-3%2', '--job-name=poolfraction-resolvi',
                f'--output={logs}/poolfraction-resolvi-%A_%a.out', f'--error={logs}/poolfraction-resolvi-%A_%a.err')
    elif args.stage == 'syndri':
        train = submit('syndri_train', benchmark_scripts/'train_pool_fraction_syndri.sbatch',
                       common_env)
        submit('syndri', benchmark_scripts/'run_pool_fraction_syndri.sbatch', dict(common_env,
            SPILLOVER_TEST_MANIFEST=manifest, SPILLOVER_MODEL_REFERENCE=train_val,
            SPILLOVER_SYNDRI_OUTPUT_DIR=BENCHMARK/'syndri'), '--array=0-3%2', f'--dependency=afterok:{train}')
    else:
        required = ['syndri', 'resolvi', 'split']
        dependencies = ':'.join(record['jobs'][key]['job_id'] for key in required)
        submit('figures', benchmark_scripts/'render_pool_fraction_notebook.sbatch', common_env,
               f'--dependency=afterok:{dependencies}')
    print(f'Job manifest: {record_path}', flush=True)


if __name__ == '__main__':
    main()
