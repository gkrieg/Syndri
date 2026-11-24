#!/usr/bin/env python3
"""Run cellAdmix and ovrlpy on frozen lambda=0.25 test transcript locations."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT/'outputs/SyndriSIM_pool_fraction'
SLICES = ('Lung12', 'Lung13', 'Lung9_Rep1', 'Lung9_Rep2')
sys.path.insert(0, str(ROOT/'scripts/competitors'))
from common import save_new, validate_counts


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def inputs(name):
    folder = BENCHMARK/'transcript_coordinates'/name/'all_fovs'
    validation = json.loads((folder/'validation.json').read_text())
    if not all(validation['checks'].values()):
        raise ValueError(f'Invalid source transcripts: {name}')
    provenance = json.loads((folder/'simulation.json').read_text())
    return folder, validation, provenance


def normalized_folder(name):
    return BENCHMARK/'transcript_method_inputs'/name


def load_metadata(path):
    with h5py.File(path) as handle:
        return ad.io.read_elem(handle['obs']), ad.io.read_elem(handle['var'])


def add_entries(matrix, q, g):
    if len(q):
        keys, count = np.unique(q*matrix.shape[1]+g, return_counts=True)
        matrix.ravel()[keys] += count


def prepare(name):
    folder, validation, provenance = inputs(name)
    source = folder/f'{name}_tx_file.csv'
    output = normalized_folder(name)
    if output.exists():
        raise FileExistsError(f'Preserving existing normalized inputs: {output}')
    output.mkdir(parents=True)
    fov_folder = output/'fovs'
    fov_folder.mkdir()
    counts_path = folder/'simulated_counts.h5ad'
    a = ad.read_h5ad(counts_path)
    validate_counts(a)
    ids, genes = pd.Index(a.obs_names.astype(str)), pd.Index(a.var_names.astype(str))
    labels = a.obs.cell_type.astype(str).to_numpy()
    if a.obs.cell_type.isna().any():
        raise ValueError('Missing cell-type annotations')
    recovered = np.zeros(a.shape, dtype=np.int64)
    reader = pacsv.open_csv(source, read_options=pacsv.ReadOptions(block_size=32*1024**2),
        convert_options=pacsv.ConvertOptions(include_columns=[
            'assigned_cell','gene_id','fov','x_global_px','y_global_px','z']))
    writers, fov_counts = {}, {}
    total, header = 0, True
    try:
        with (output/'transcripts_celladmix.csv').open('xb') as csv_stream:
            for batch in reader:
                frame = batch.to_pandas()
                q, g = ids.get_indexer(frame.assigned_cell), genes.get_indexer(frame.gene_id)
                if (q < 0).any() or (g < 0).any():
                    raise AssertionError('Unknown assigned cell or gene; no rows may be discarded')
                fov = frame.fov.to_numpy(np.int64)
                if not np.array_equal(fov, a.obs.fov.to_numpy(np.int64)[q]):
                    raise AssertionError('FOV does not match assigned cell')
                add_entries(recovered, q, g)
                # Only observed/model covariates are exported. No native/added,
                # donor, source-cell, or admixture-truth columns enter either fit.
                table = pa.Table.from_pydict(dict(
                    mol_id=np.arange(total, total+len(frame), dtype=np.int64),
                    cell_id=ids.to_numpy()[q], cell_type=labels[q], gene=genes.to_numpy()[g],
                    x=frame.x_global_px.to_numpy(float)*float(provenance['pixel_size_um']),
                    y=frame.y_global_px.to_numpy(float)*float(provenance['pixel_size_um']),
                    z=frame.z.to_numpy(float)*.8, fov=fov))
                pacsv.write_csv(table.drop(['fov']), csv_stream,
                                write_options=pacsv.WriteOptions(include_header=header))
                header = False
                for number in np.unique(fov):
                    key = int(number)
                    if key not in writers:
                        writers[key] = pq.ParquetWriter(fov_folder/f'fov_{key:03d}.parquet',
                                                       table.schema, compression='zstd')
                    part = table.filter(pa.array(fov == number))
                    writers[key].write_table(part, row_group_size=250000)
                    fov_counts[key] = fov_counts.get(key, 0)+len(part)
                total += len(frame)
                if total // 5000000 != (total-len(frame)) // 5000000:
                    print(f'{name}: normalized {total:,} molecules', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    difference = sparse.csr_matrix(recovered)-sparse.csr_matrix(a.X)
    difference.eliminate_zeros()
    if difference.nnz or total != validation['n_transcripts']:
        raise AssertionError('Normalized molecules do not exactly reconstruct the simulated counts')
    write_json(output/'ready.json', dict(slice=name, source=signature(source),
        counts=signature(counts_path), n_transcripts=total, n_cells=a.n_obs,
        n_genes=a.n_vars, n_cell_types=int(a.obs.cell_type.nunique()),
        fov_counts=fov_counts, x_y_pixel_size_um=provenance['pixel_size_um'],
        z_plane_spacing_um=.8, z_conversion='original discrete planes multiplied by 0.8, matching the existing CosMx adapters',
        molecule_id='zero-based row number in the original simulated transcript CSV',
        admixture_ground_truth_used_for_fitting=False, count_reconstruction_exact=True))
    print(f'{name}: normalized {total:,} molecules; counts exactly reconstructed', flush=True)


def ready(name):
    folder, _, _ = inputs(name)
    staged = normalized_folder(name)
    record = json.loads((staged/'ready.json').read_text())
    if not record['count_reconstruction_exact'] or record['admixture_ground_truth_used_for_fitting']:
        raise ValueError('Unvalidated normalized molecules')
    if record['source'] != signature(folder/f'{name}_tx_file.csv'):
        raise ValueError('Source transcripts changed after preparation')
    return folder, staged, record


def export_celladmix(name, assignment_path, factor_map, target):
    folder, _, record = ready(name)
    a = ad.read_h5ad(folder/'simulated_counts.h5ad')
    ids, genes = pd.Index(a.obs_names.astype(str)), pd.Index(a.var_names.astype(str))
    mapping = pd.read_csv(factor_map)
    if list(mapping.columns) != ['cell_type','factor'] or mapping.duplicated().any():
        raise ValueError('Invalid cellAdmix factor map')
    to_remove = pd.MultiIndex.from_frame(mapping[['cell_type','factor']])
    removed = np.zeros(a.shape, dtype=np.int64)
    recovered = np.zeros(a.shape, dtype=np.int64)
    reader = pacsv.open_csv(assignment_path, read_options=pacsv.ReadOptions(block_size=32*1024**2),
        convert_options=pacsv.ConvertOptions(column_types={'mol_id':pa.int64(),'factor':pa.float64(),
            'cell_id':pa.string(),'gene':pa.string(),'cell_type':pa.string()}))
    total = 0
    for batch in reader:
        frame = batch.to_pandas()
        if not np.array_equal(frame.mol_id.to_numpy(), np.arange(total, total+len(frame))):
            raise AssertionError('cellAdmix molecule IDs are missing, duplicated, or reordered')
        q, g = ids.get_indexer(frame.cell_id), genes.get_indexer(frame.gene)
        if (q < 0).any() or (g < 0).any():
            raise AssertionError('Unknown cellAdmix assignment identifier')
        if not np.array_equal(frame.cell_type.to_numpy(), a.obs.cell_type.astype(str).to_numpy()[q]):
            raise AssertionError('cellAdmix cell types do not match the simulation')
        add_entries(recovered,q,g)
        keys = pd.MultiIndex.from_frame(frame[['cell_type','factor']])
        flag = keys.isin(to_remove)
        add_entries(removed,q[flag],g[flag])
        total += len(frame)
    raw = sparse.csr_matrix(a.X)
    delta = sparse.csr_matrix(recovered)-raw
    delta.eliminate_zeros()
    if delta.nnz or total != record['n_transcripts']:
        raise AssertionError('cellAdmix assignments do not reconstruct the original input')
    corrected = raw-sparse.csr_matrix(removed)
    if (corrected.data < 0).any():
        raise AssertionError('cellAdmix removed more molecules than were present')
    a.layers['celladmix_corrected_counts'] = corrected
    a.uns['celladmix_competitor'] = dict(factor_map=str(factor_map), assignments=str(assignment_path),
        factor_count=int(math.ceil(1.5*record['n_cell_types'])),
        annotation='existing bridge-test plus distant-cell false-positive check',
        admixture_ground_truth_used_for_fitting=False, counts_reconstructed_exactly=True)
    save_new(a,target)
    write_json(target.with_suffix('.json'), dict(n_cells=a.n_obs, n_transcripts=total,
        removed_transcripts=int(removed.sum()), n_factor_mappings=len(mapping),
        assignments_reconstruct_input=True, output=str(target)))


def celladmix(name, workers):
    _, staged, record = ready(name)
    target = BENCHMARK/'celladmix'/f'{name}.h5ad'
    if target.exists():
        raise FileExistsError(target)
    fit = BENCHMARK/'celladmix/factorization'/name
    annotation = BENCHMARK/'celladmix/annotation'/name
    fit.mkdir(parents=True,exist_ok=True)
    k = int(math.ceil(1.5*record['n_cell_types']))
    env = dict(os.environ,CELLADMIX_K=str(k),CELLADMIX_CORES=str(workers),
               CELLADMIX_SAMPLE_CELLS='1000',CELLADMIX_NMOL_DSAMP='2000')
    rscript = '/home/jwinchel/miniconda3/envs/r_env/bin/Rscript'
    fit_paths = [fit/f for f in ['assignments.csv','factor_summary.csv','factor_loadings.csv']]
    if not all(p.is_file() for p in fit_paths):
        if any(p.exists() for p in fit_paths):
            raise FileExistsError('Partial factorization requires inspection before retry')
        subprocess.run([rscript,str(ROOT/'scripts/spillover-benchmark/run_celladmix_batched.R'),
                        str(staged/'transcripts_celladmix.csv'), *map(str,fit_paths)],
                       env=env,check=True)
    map_path = annotation/'factor_map.csv'
    if not all((annotation/f).is_file() for f in ['bridge_raw.rds','bridge_res.rds','factor_map.csv']):
        backend = os.environ.get('CELLADMIX_ANNOTATION_BACKEND', 'bounded')
        if backend == 'original':
            subprocess.run([rscript,str(ROOT/'scripts/competitors/annotate_celladmix_k1p5ct.R'),
                str(staged/'transcripts_celladmix.csv'),str(fit_paths[0]),str(k),str(annotation),str(workers)],
                env=env,check=True)
        elif backend == 'bounded':
            from celladmix_annotation_bounded import run_annotation
            folder, _, _ = ready(name)
            obs, _ = load_metadata(folder/'simulated_counts.h5ad')
            annotation.mkdir(parents=True, exist_ok=True)
            metadata_path = annotation/'input_cells.csv'
            if not metadata_path.exists():
                pd.DataFrame(dict(cell=obs.index.astype(str),
                    celltype=obs.cell_type.astype(str).to_numpy())).to_csv(metadata_path,index=False)
            run_annotation(staged/'transcripts_celladmix.csv', fit_paths[0], metadata_path,
                           k, annotation, workers=workers)
        else:
            raise ValueError(f'Unknown annotation backend: {backend}')
    export_celladmix(name,fit_paths[0],map_path,target)
    print(f'cellAdmix output: {target}',flush=True)


def ovrlpy_run(name, workers):
    import polars as pl
    import ovrlpy
    from ovrlpy import Ovrlp, cell_integrity_from_transcripts
    folder, staged, record = ready(name)
    target = BENCHMARK/'ovrlpy'/f'{name}.h5ad'
    if target.exists():
        raise FileExistsError(target)
    a = ad.read_h5ad(folder/'simulated_counts.h5ad')
    ids, genes = pd.Index(a.obs_names.astype(str)),pd.Index(a.var_names.astype(str))
    removed = {threshold:np.zeros(a.shape,dtype=np.int64) for threshold in [.5,.7]}
    recovered = np.zeros(a.shape,dtype=np.int64)
    scores, total = [],0
    score_root = BENCHMARK/'ovrlpy/transcript_scores'/name
    score_root.mkdir(parents=True,exist_ok=True)
    for fov_path in sorted((staged/'fovs').glob('*.parquet')):
        print(f'{name}: ovrlpy {fov_path.stem}',flush=True)
        score_path = score_root/fov_path.name
        if score_path.exists():
            raise FileExistsError(f'Preserving existing per-FOV scores: {score_path}')
        tx = pl.from_arrow(pq.read_table(fov_path,columns=['mol_id','cell_id','gene','x','y','z']))
        # Preserve 3D distances while allowing the observed CosMx z=-1 plane.
        tx = tx.with_columns((pl.col('z')-pl.col('z').min()).alias('z'))
        ov = Ovrlp(tx,n_components=min(30,tx['gene'].n_unique()-1),n_workers=workers,random_state=0)
        ov.analyse(fit_umap=False)
        positioned = ov.transcripts
        pixel_x = positioned['x_pixel'].to_numpy().astype(np.int64)
        pixel_y = positioned['y_pixel'].to_numpy().astype(np.int64)
        vsi = np.asarray(ov.integrity_map[pixel_y,pixel_x],dtype=np.float32)
        table = positioned.select(['mol_id','cell_id','gene','x_pixel','y_pixel']).with_columns(pl.Series('vsi',vsi))
        q,g = ids.get_indexer(table['cell_id'].to_numpy()),genes.get_indexer(table['gene'].to_numpy())
        if (q < 0).any() or (g < 0).any() or len(table)!=len(tx):
            raise AssertionError('ovrlpy molecule IDs/counts changed')
        add_entries(recovered,q,g)
        for threshold in removed:
            flag = np.isfinite(vsi) & (vsi < threshold)
            add_entries(removed[threshold],q[flag],g[flag])
        per_cell = cell_integrity_from_transcripts(ov,cell_id='cell_id',unassigned=None).group_by('cell_id').agg(
            pl.col('vsi').mean().alias('ovrlpy_vsi'))
        scores.append(per_cell.to_pandas())
        pq.write_table(table.to_arrow(),score_path,compression='zstd')
        total += len(table)
        del ov,tx,positioned,table
        gc.collect()
    delta = sparse.csr_matrix(recovered)-sparse.csr_matrix(a.X)
    delta.eliminate_zeros()
    if delta.nnz or total != record['n_transcripts']:
        raise AssertionError('ovrlpy transcript scores do not reconstruct the input')
    score = pd.concat(scores).set_index('cell_id').ovrlpy_vsi
    if not score.index.is_unique:
        raise AssertionError('A cell occurs in multiple FOVs')
    a.obs['ovrlpy_vsi'] = score.reindex(ids).to_numpy(np.float32)
    for threshold,matrix in removed.items():
        corrected = sparse.csr_matrix(a.X)-sparse.csr_matrix(matrix)
        if (corrected.data < 0).any():
            raise AssertionError('Negative ovrlpy corrected counts')
        a.layers[f'ovrlpy_corrected_counts_vsi_lt_{str(threshold).replace(".","_")}'] = corrected
    a.uns['ovrlpy_competitor'] = dict(version=ovrlpy.__version__,workers=workers,random_state=0,
        fit='independent per FOV; analyse(fit_umap=False), n_components=min(30,n_genes-1)',
        cell_score='mean VSI over unique transcript-occupied pixels within cell',
        corrected_counts='remove transcripts whose pixel VSI is below threshold; retain nonfinite scores',
        transcript_scores=str(score_root),admixture_ground_truth_used_for_fitting=False,
        counts_reconstructed_exactly=True)
    save_new(a,target)
    write_json(target.with_suffix('.json'),dict(n_cells=a.n_obs,n_transcripts=total,
        removed_transcripts={str(t):int(m.sum()) for t,m in removed.items()},
        input_reconstructed_exactly=True,output=str(target)))
    print(f'ovrlpy output: {target}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','celladmix','ovrlpy'],required=True)
    parser.add_argument('--task-id',type=int,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    if args.workers<1 or not 0<=args.task_id<len(SLICES):
        raise ValueError('Invalid workers or array index')
    pa.set_cpu_count(args.workers)
    name=SLICES[args.task_id]
    print(f'{args.stage}: {name}; workers={args.workers}; whole-gene lambda_max=0.25',flush=True)
    if args.stage=='prepare':prepare(name)
    elif args.stage=='celladmix':celladmix(name,args.workers)
    else:ovrlpy_run(name,args.workers)


if __name__=='__main__':main()
