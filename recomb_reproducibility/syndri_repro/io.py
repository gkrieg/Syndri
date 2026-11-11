"""Paths and matrix I/O restricted to this reproduction directory."""
import json
from pathlib import Path
import h5py
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml
from . import ROOT
from .parameters import configuration_path


def config():
    return yaml.safe_load(configuration_path('paper').read_text())


def resolve(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Runtime paths must stay inside {ROOT}: {relative}')
    return path


def dataset(name):
    return config()['datasets'][name]


def data_dir(name):
    return resolve(dataset(name)['directory'])


def csr(matrix):
    result = sparse.csr_matrix(matrix)
    result.eliminate_zeros()
    return result


def read(path, layers=(), uns=('cell_type_dict',)):
    """Read only requested matrices; never load unrelated AnnData layers."""
    with h5py.File(path, 'r') as handle:
        x = handle['X']
        if isinstance(x, h5py.Dataset):
            blocks = [sparse.csr_matrix(x[i:i + 2048]) for i in range(0, len(x), 2048)]
            counts = sparse.vstack(blocks, format='csr')
        else:
            counts = csr(ad.io.read_elem(x))
        obj = ad.AnnData(counts, obs=ad.io.read_elem(handle['obs']), var=ad.io.read_elem(handle['var']))
        for key in layers:
            if f'layers/{key}' in handle:
                obj.layers[key] = csr(ad.io.read_elem(handle[f'layers/{key}']))
        for key in uns:
            if f'uns/{key}' in handle:
                obj.uns[key] = ad.io.read_elem(handle[f'uns/{key}'])
    return obj


def write(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.settings.allow_write_nullable_strings = True
    tmp = path.with_name(path.name + '.part')
    obj.write_h5ad(tmp, compression='gzip')
    tmp.replace(path)


def spatial_files(name):
    return sorted((data_dir(name) / ('test_slices' if name == 'SIM' else 'spatial')).glob('*.h5ad'))


def predicted_file(name, source):
    path = ROOT / 'outputs/predictions' / name / Path(source).name
    return path if path.exists() else source


def symbols(obj):
    ids = pd.Series(obj.var_names.astype(str), index=obj.var_names, dtype='string')
    key = next((key for key in ('feature_name', 'gene_symbol', 'name') if key in obj.var), None)
    values = obj.var[key].astype('string') if key else ids.copy()
    missing = values.isna() | values.str.lower().isin(['', 'n/a', 'na', 'nan', 'none', 'null'])
    return values.mask(missing, ids).astype(str).to_numpy()


def valid_labels(values):
    return values.notna().to_numpy() & ~values.astype(str).str.lower().isin(['nan', 'unannotated', 'unassigned', 'none', '']).to_numpy()


def qc(counts):
    settings = config()['qc']
    return ((np.asarray(counts.sum(1)).ravel() >= settings['min_counts']) &
            (np.asarray((counts > 0).sum(1)).ravel() >= settings['min_genes']))


def save_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')

