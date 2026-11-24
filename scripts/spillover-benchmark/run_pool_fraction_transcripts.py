#!/usr/bin/env python3
"""Generate locations for the existing whole-gene lambda=0.25 test counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'recomb_reproducibility'))
sys.path.insert(0, str(ROOT/'Heimdall'))

SLICES = ('Lung12', 'Lung13', 'Lung9_Rep1', 'Lung9_Rep2')


def require_pilot(output_root):
    path = output_root/'Lung12/fov_001_n100/validation.json'
    report = json.loads(path.read_text())
    if report['n_cells'] != 100 or len(report['checks']) != 7 or not all(report['checks'].values()):
        raise AssertionError(f'The 100-cell pilot must pass all seven checks before full slices: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'full'], required=True)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--benchmark-root', type=Path, default=ROOT/'outputs/SyndriSIM_pool_fraction')
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.task_id < len(SLICES):
        raise ValueError('Expected one of the four held-out slice task indices')
    root = args.benchmark_root.resolve()
    selected = json.loads((root/'search/selected_parameters.json').read_text())
    if selected['lambda_max'] != .25 or selected['uses_held_out_test_for_selection']:
        raise ValueError('Expected the frozen training-only whole-gene lambda=0.25 configuration')
    reports = json.loads((root/'validation.json').read_text())
    if {row['slice'] for row in reports} != set(SLICES):
        raise ValueError('Expected four validated test inputs')
    for report in reports:
        if not all(value for value in report.values() if isinstance(value, bool)):
            raise ValueError(f'Input validation failed: {report["slice"]}')
    data = Path(json.loads((root/'parameters.json').read_text())['data_root'])
    output = root/'transcript_coordinates'
    if args.stage == 'full':
        require_pilot(output)
    name = 'Lung12' if args.stage == 'pilot' else SLICES[args.task_id]
    print(f'Stage={args.stage}; slice={name}; saved whole-gene lambda=0.25 counts; decay=2 um; seed={args.seed}', flush=True)
    from syndri_repro.transcript_locations import simulate_transcripts
    _, report = simulate_transcripts(
        root/'slices'/f'{name}.h5ad',
        data/'results/final'/f'st_integrated_slice-{name}.h5ad',
        data/'raw/spatial'/name/f'{name}-Flat_files_and_images', output,
        fov=1 if args.stage == 'pilot' else None,
        max_cells=100 if args.stage == 'pilot' else None,
        seed=args.seed, decay_um=2., batch_cells=8 if args.stage == 'pilot' else 64, workers=args.workers,
        donor_allocations_path=root/'slices'/f'{name}_donor_allocations.parquet')
    if not all(report['checks'].values()):
        raise AssertionError('Coordinate validation did not pass')
    if args.stage == 'pilot':
        require_pilot(output)
    else:
        expected = next(row for row in reports if row['slice'] == name)
        for key, original in [('n_cells', 'n_cells'), ('n_native', 'n_native_transcripts'),
                              ('n_admixed', 'n_added_transcripts')]:
            if report[key] != expected[original]:
                raise AssertionError(f'Output transcript totals differ from frozen simulation: {key}')


if __name__ == '__main__':
    main()
