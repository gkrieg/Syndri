#!/usr/bin/env python
"""Import a user-supplied file specification with a configurable source root.

Only count matrices, Syndri probabilities, annotations, and simulation truth are
retained. No other method's layers, annotations, scripts, or result files are copied.
"""
import argparse
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import h5py
import numpy as np
from scipy import sparse
from omegaconf import OmegaConf
from syndri_repro import ROOT
from syndri_repro.io import read, write, resolve, save_json, csr
from syndri_repro.assets import sha256


def observed_probabilities(path, key, counts):
    blocks = []
    with h5py.File(path, 'r') as f:
        element = f[f'layers/{key}']
        if isinstance(element, h5py.Dataset):
            for start in range(0, counts.shape[0], 2048):
                blocks.append(sparse.csr_matrix(element[start:start + 2048]).multiply(counts[start:start + 2048] > 0))
            return sparse.vstack(blocks, format='csr')
        return csr(ad.io.read_elem(element)).multiply(counts > 0).tocsr()


def import_h5ad(item, target):
    original = Path(item['source'])
    obj = read(original, layers=item.get('layers', []), uns=('cell_type_dict',))
    if item.get('count_layer'):
        with h5py.File(original, 'r') as f:
            obj.X = csr(ad.io.read_elem(f['layers'][item['count_layer']]))
    obj.obs = obj.obs[[c for c in obj.obs if not (
        c.lower().startswith(('rctd_', 'split_', 'source_', 'sc_match_', 'st_match_'))
        or c.startswith('used_in_')
    ) or c in {'split', 'source_sc_id'}]].copy()
    if item.get('probability_layer'):
        p_source = Path(item.get('probability_source', original))
        with h5py.File(p_source, 'r') as f:
            obs = ad.io.read_elem(f['obs']); var = ad.io.read_elem(f['var'])
        rows = obs.index.get_indexer(obj.obs_names); cols = var.index.get_indexer(obj.var_names)
        if (rows < 0).any() or (cols < 0).any():
            raise ValueError(f'Prediction/input identifiers do not match: {original}')
        if np.array_equal(rows, np.arange(obj.n_obs)) and np.array_equal(cols, np.arange(obj.n_vars)):
            probs = observed_probabilities(p_source, item['probability_layer'], obj.X)
        else:
            with h5py.File(p_source, 'r') as f:
                probs = csr(ad.io.read_elem(f['layers'][item['probability_layer']]))[rows][:, cols]
            probs = probs.multiply(obj.X > 0).tocsr()
        obj.layers['admix_probs'] = probs
    if 'synthetic_admixture_counts' in obj.layers:
        residual = obj.X - obj.layers['original_counts'] - obj.layers['synthetic_admixture_counts']
        if residual.nnz and np.max(np.abs(residual.data)) > 1e-6:
            raise ValueError(f'Synthetic truth does not reconstruct mixed counts: {original}')
    write(obj, target)
    return {'shape': list(obj.shape), 'layers': list(obj.layers), 'obs_columns': list(obj.obs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    source_root = Path(sources.get('source_root', '.')).expanduser()
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    old_path = ROOT/'assets/manifest.json'
    old = {entry['path']: entry for entry in json.loads(old_path.read_text())['files']} if old_path.exists() else {}
    manifest = {'version': 1, 'description': 'Syndri-only paper assets', 'files': []}
    provenance = []
    for specification in sources['files']:
        item = dict(specification)
        for field in ['source', 'probability_source']:
            if item.get(field):
                source = Path(item[field]).expanduser()
                item[field] = str(source if source.is_absolute() else source_root / source)
        target = resolve(item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        details = {}
        if target.exists():
            print('Already imported:', item['path'], flush=True)
            details = {key: value for key, value in old.get(item['path'], {}).items()
                       if key not in {'path', 'bytes', 'sha256', 'url'}}
        elif item.get('kind') == 'h5ad':
            print('Import:', item['path'], flush=True)
            details = import_h5ad(item, target)
        elif item.get('kind') == 'config':
            cfg = OmegaConf.load(item['source']); OmegaConf.set_struct(cfg, False)
            name = item['dataset']
            cfg.data_path = 'assets/metadata'; cfg.ensembl_dir = '${data_path}'
            cfg.dataset.preprocess_args.data_path = f'assets/data/{name}/training.h5ad'
            cfg.fc.args.ensembl_dir = '${data_path}'
            cfg.fc.args.gene_metadata_filepath = '${data_path}/gene_metadata/species_chrom.csv'
            cfg.work_dir = f'outputs/models/{name}'
            cfg.run_name = f'syndri_recomb_{name}'
            if 'external_eval' in cfg.tasks.args:
                cfg.tasks.args.external_eval = {'enabled': False}
            if 'silhouette_delta' in cfg.tasks.args:
                cfg.tasks.args.silhouette_delta.enabled = False
            target.write_text(OmegaConf.to_yaml(cfg))
        else:
            shutil.copy2(item['source'], target)
        manifest['files'].append({'path': item['path'], 'bytes': target.stat().st_size,
                                  'sha256': sha256(target), 'url': 'FILL IN: public asset URL',
                                  **details, **({'mutable': True} if item['path'].startswith('configs/') else {})})
        provenance.append({'path': item['path'], 'source_name': Path(specification['source']).name, **details})
    # Preserve subsequently bundled context/source-catalog assets when repeating
    # the original data/model bootstrap import.
    imported = {entry['path'] for entry in manifest['files']}
    manifest['files'].extend(entry for path, entry in old.items() if path not in imported)
    previous_sources = ROOT/'provenance/asset_sources.json'
    if previous_sources.exists():
        provenance.extend(entry for entry in json.loads(previous_sources.read_text())
                          if entry['path'] not in imported)
    save_json(manifest, ROOT/'assets/manifest.json')
    save_json(provenance, ROOT/'provenance/asset_sources.json')
    print('Imported', len(manifest['files']), 'assets', flush=True)


if __name__ == '__main__':
    main()

