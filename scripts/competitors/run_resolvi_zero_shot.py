"""Fit ResolVI on the other slices in a dataset and score one held-out slice."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import scvi
from scipy import sparse

from common import LABELS, dataset_of, output_path, save_new, validate_counts


COORDS = {
    'nsclc_cosmx': ('CenterX_global_px', 'CenterY_global_px'),
    'breast_xenium': ('x_centroid', 'y_centroid'),
    'agingbrain_merfish': ('center_x', 'center_y'),
    'brain_merfish': ('x', 'y'),
}


def slim_slice(final: Path, dataset: str, labels: list[str]):
    original = ad.read_h5ad(final)
    validate_counts(original)
    key = LABELS[dataset]
    xy = original.obs[list(COORDS[dataset])].to_numpy(dtype=float)
    if dataset == 'nsclc_cosmx':
        xy *= 0.12028
    valid = np.isfinite(xy).all(axis=1) & original.obs[key].notna().to_numpy()
    if valid.sum() < 20:
        raise ValueError(f'Too few annotated cells with spatial coordinates: {final}')
    work = ad.AnnData(
        X=original.X[valid].copy(),
        obs=pd.DataFrame(index=original.obs_names[valid].copy()),
        var=pd.DataFrame(index=original.var_names.copy()),
    )
    work.obs[key] = pd.Categorical(original.obs.loc[valid, key].astype(str), categories=labels)
    work.obsm['X_spatial'] = xy[valid]
    return original, work, valid


def transfer_without_fitting(source, query, pyro_state):
    """Transfer shared learned parameters, resetting the source's per-cell map."""
    state = {name: value.detach().clone() for name, value in source.module.state_dict().items()}
    target_state = query.module.state_dict()
    local_buffer = '_guide.per_neighbor_diffusion_init'
    shared = {
        name: value for name, value in state.items()
        if name != local_buffer and name in target_state and value.shape == target_state[name].shape
    }
    unexpected = set(state) - set(shared) - {local_buffer}
    if unexpected:
        raise ValueError(f'Cannot transfer model parameters: {sorted(unexpected)}')
    query.module.load_state_dict(shared, strict=False)
    for name in ('downsample_counts_mean', 'downsample_counts_std'):
        value = getattr(source.module.guide, name)
        setattr(query.module.guide, name, None if value is None else float(value))
    # The per-neighbor diffusion map has one row per training cell. It must
    # start at its neutral value for the held-out cells, with no query fitting.
    local_param = 'per_neighbor_diffusion_map'
    for field in ('params', 'constraints'):
        pyro_state[field].pop(local_param, None)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(pyro_state)
    query.module.eval()


def run(final: Path, epochs: int, samples: int, output_root: Path | None = None):
    if scvi.__version__ != '1.3.3':
        raise RuntimeError(f'Expected scvi-tools 1.3.3, found {scvi.__version__}')
    dataset = dataset_of(final)
    target = output_path(final, 'resolvi_zero_shot', output_root)
    if target.exists():
        raise FileExistsError(target)
    all_finals = sorted((final.parent).glob('*.h5ad'))
    sources = [path for path in all_finals if path != final]
    if not sources:
        raise ValueError(f'No training slices remain after holding out {final}')

    # A shared category vocabulary only defines the annotation schema. No
    # held-out expression or coordinates enter the training AnnData.
    key = LABELS[dataset]
    labels = set()
    for path in all_finals:
        backed = ad.read_h5ad(path, backed='r')
        try:
            labels.update(backed.obs[key].dropna().astype(str).unique())
        finally:
            backed.file.close()
    labels = sorted(labels)

    train_parts = []
    reference_vars = None
    for path in sources:
        original, work, _ = slim_slice(path, dataset, labels)
        if reference_vars is None:
            reference_vars = work.var_names.copy()
        elif not work.var_names.equals(reference_vars):
            raise ValueError(f'Gene order differs in {path}')
        train_parts.append(work)
        del original
    train = ad.concat(
        {path.stem: work for path, work in zip(sources, train_parts)},
        label='_source_slice', index_unique=':', join='inner',
    )
    del train_parts
    print(f'Training: {dataset}, {len(sources)} source slices, {train.n_obs} cells', flush=True)
    scvi.settings.seed = 0
    scvi.external.RESOLVI._prepare_data(train, slice_key='_source_slice')
    scvi.external.RESOLVI.setup_anndata(train, labels_key=key, prepare_data=False)
    model = scvi.external.RESOLVI(train, semisupervised=True)
    for name in ('downsample_counts_mean', 'downsample_counts_std'):
        value = getattr(model.module.guide, name)
        if value is not None:
            setattr(model.module.guide, name, float(value))
    model.train(max_epochs=epochs)
    source_pyro_state = deepcopy(pyro.get_param_store().get_state())

    original, query, valid = slim_slice(final, dataset, labels)
    if not query.var_names.equals(reference_vars):
        raise ValueError(f'Gene order differs in held-out slice {final}')
    scvi.external.RESOLVI.setup_anndata(query, labels_key=key)
    query_model = scvi.external.RESOLVI(query, semisupervised=True)
    transfer_without_fitting(model, query_model, source_pyro_state)
    posterior = query_model.sample_posterior(
        model=query_model.module.model_corrected,
        return_sites=['px_rate'], summary_fun={'median': np.median},
        num_samples=samples, summary_frequency=30,
    )
    corrected = np.asarray(pd.DataFrame(posterior).T.loc['median', 'px_rate'], dtype=np.float32)
    if corrected.shape != query.shape or not np.isfinite(corrected).all() or (corrected < 0).any():
        raise ValueError(f'Unexpected corrected count shape or values: {corrected.shape}')
    full = original.X.toarray().astype(np.float32) if sparse.issparse(original.X) else np.asarray(original.X, dtype=np.float32).copy()
    full[valid, :] = corrected
    original.layers['resolvi_corrected_counts'] = full
    original.obs['resolvi_scored'] = valid
    original.uns['resolvi_competitor'] = {
        'fit_mode': 'leave_one_slice_out_zero_shot',
        'scvi_version': scvi.__version__,
        'max_epochs': epochs,
        'posterior_samples': samples,
        'source_finals': [str(path) for path in sources],
        'held_out_final': str(final),
        'training_cells': train.n_obs,
        'unscored_cells_retain_raw_counts': int((~valid).sum()),
    }
    save_new(original, target)
    print(target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('final_h5ad', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    run(args.final_h5ad, args.epochs, args.samples, args.output_root)
