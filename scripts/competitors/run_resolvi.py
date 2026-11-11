"""Fit ResolVI to one final spatial AnnData and export posterior corrected counts."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from common import LABELS, dataset_of, output_path, save_new, validate_counts


def run(final: Path, root: Path | None, epochs: int, samples: int, model_dir: Path | None = None,
        dataset_override: str | None = None):
    import scvi

    if scvi.__version__ != '1.3.3':
        raise RuntimeError(f'Expected scvi-tools 1.3.3, found {scvi.__version__}')
    dataset = dataset_override or dataset_of(final)
    if dataset not in {'nsclc_cosmx', 'breast_xenium', 'agingbrain_merfish', 'brain_merfish'}:
        raise ValueError(f'Unsupported dataset: {dataset}')
    target = output_path(final, 'resolvi', root, dataset=dataset)
    if target.exists():
        raise FileExistsError(target)
    if model_dir is not None and model_dir.exists():
        raise FileExistsError(model_dir)
    a = ad.read_h5ad(final)
    validate_counts(a)
    key = LABELS[dataset]
    coord = {'nsclc_cosmx': ('CenterX_global_px', 'CenterY_global_px'),
             'breast_xenium': ('x_centroid', 'y_centroid'),
             'agingbrain_merfish': ('center_x', 'center_y'),
             'brain_merfish': ('x', 'y')}[dataset]
    xy = a.obs[list(coord)].to_numpy(dtype=float)
    if dataset == 'nsclc_cosmx':
        xy *= 0.12028  # CosMx global pixels to micrometers, matching transcript formatter
    nonzero = np.asarray(a.X.sum(axis=1)).ravel() > 0
    valid = np.isfinite(xy).all(axis=1) & a.obs[key].notna().to_numpy() & nonzero
    if valid.sum() < 20:
        raise ValueError('Too few annotated cells with spatial coordinates')
    work = a[valid].copy()
    work.obsm['X_spatial'] = xy[valid]
    work.layers['resolvi_input_counts'] = work.X.copy()
    scvi.settings.seed = 0
    scvi.external.RESOLVI.setup_anndata(work, labels_key=key, layer='resolvi_input_counts')
    model = scvi.external.RESOLVI(work, semisupervised=True)
    # scvi-tools 1.3.3 can pass numpy.float32 priors from dense AnnData X to
    # PyTorch LogNormal, which only accepts Python numbers or tensors here.
    for owner, field in ((model.module.guide, 'downsample_counts_mean'),
                         (model.module.guide, 'downsample_counts_std')):
        value = getattr(owner, field)
        if value is not None:
            setattr(owner, field, float(value))
    model.train(max_epochs=epochs)
    if model_dir is not None:
        if hasattr(ad.settings, 'allow_write_nullable_strings'):
            ad.settings.allow_write_nullable_strings = True
        model.save(str(model_dir), save_anndata=True, overwrite=False)
        print(f'Saved trained model and registered AnnData: {model_dir}', flush=True)
    posterior = model.sample_posterior(model=model.module.model_corrected,
                                       return_sites=['px_rate'],
                                       summary_fun={'median': np.median},
                                       num_samples=samples, summary_frequency=30)
    corrected = np.asarray(pd.DataFrame(posterior).T.loc['median', 'px_rate'], dtype=np.float32)
    if corrected.shape != work.shape or not np.isfinite(corrected).all() or (corrected < 0).any():
        raise ValueError(f'Unexpected corrected count shape or values: {corrected.shape}')
    full = a.X.toarray().astype(np.float32) if sparse.issparse(a.X) else np.asarray(a.X, dtype=np.float32).copy()
    full[valid, :] = corrected
    a.layers['resolvi_corrected_counts'] = full
    a.obs['resolvi_scored'] = valid
    a.uns['resolvi_competitor'] = {'scvi_version': scvi.__version__, 'max_epochs': epochs,
                                   'posterior_samples': samples, 'source_h5ad': str(final),
                                   'unscored_cells_retain_raw_counts': int((~valid).sum())}
    if model_dir is not None:
        a.uns['resolvi_competitor']['model_dir'] = str(model_dir)
    if final.parent.name == 'final':
        a.uns['resolvi_competitor']['source_final'] = str(final)
    save_new(a, target)
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('final_h5ad', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--model-dir', type=Path, help='Save trained model and registered AnnData without overwriting')
    p.add_argument('--dataset', choices=['nsclc_cosmx', 'breast_xenium', 'agingbrain_merfish', 'brain_merfish'],
                   help='Explicit dataset profile for inputs outside the standard dataset directories')
    a = p.parse_args()
    run(a.final_h5ad, a.output_root, a.epochs, a.samples, a.model_dir, a.dataset)
