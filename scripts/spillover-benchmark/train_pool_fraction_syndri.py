#!/usr/bin/env python3
"""Retrain the Figure 3 model with the frozen whole-gene pool-fraction sampler.

Uses the original train.py entry point, architecture, optimizer, split, and
learning-rate horizon. Only this process substitutes the epoch-store sampler;
the shared Heimdall implementation and original checkpoints remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '8')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Heimdall'))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

from Heimdall.admixture_epoch_store import AdmixtureEpochStore
from Heimdall.trainer import HeimdallTrainer
import Heimdall.cell_representations as representations
from donor_weighted_admixture import sample_weighted


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist())+'\n')


class PoolFractionEpochStore(AdmixtureEpochStore):
    """Keep lazy epoch refresh and fixed validation exactly as in the original."""

    def __init__(self, adata, admixture_args, *args, **kwargs):
        if admixture_args.get('sampler') != 'donor_type_weighted_pool_fraction':
            raise ValueError('Expected the calibrated whole-gene donor-pool sampler')
        if float(admixture_args['lambda_max']) != .25:
            raise ValueError('This run is fixed to lambda_max=0.25')
        self._pool_obs = adata.obs.copy()
        self._pool_native = sparse.csr_matrix(adata.X)
        if not np.isfinite(self._pool_native.data).all() or (self._pool_native.data < 0).any() or not np.array_equal(
                self._pool_native.data, np.rint(self._pool_native.data)):
            raise ValueError('Training data must contain unchanged native integer counts')
        self._pool_native = self._pool_native.astype(np.int64)
        ids = pd.Index(adata.obs_names.astype(str))
        neighbors = adata.uns['neighbor_dict']
        sizes = np.array([len(neighbors.get(key, [])) for key in ids], dtype=np.int64)
        sources = ids.get_indexer([source for key in ids for source in neighbors.get(key, [])])
        if (sources < 0).any() or (sizes > 9).any():
            raise AssertionError('Invalid existing nine-neighbor graph')
        queries = np.repeat(np.arange(len(ids)), sizes)
        labels = adata.obs.cell_type.astype(str).to_numpy()
        slices = adata.obs.spatial_slice.astype(str).to_numpy()
        if (labels[queries] == labels[sources]).any() or (slices[queries] != slices[sources]).any():
            raise AssertionError('Neighbor graph crosses cell-type eligibility or slice boundaries')
        self._pool_graph = sparse.csr_matrix((np.ones(len(sources), dtype=np.int64),
            sources, np.r_[0, np.cumsum(sizes)]), shape=(len(ids), len(ids)))
        self._pool_weights = {str(key): float(value) for key, value in admixture_args['donor_type_weights'].items()}
        if set(labels) - set(self._pool_weights):
            raise ValueError('Missing frozen donor-type weights for training cells')
        super().__init__(adata, admixture_args, *args, **kwargs)
        write_json(self.cache_dir/'training_split_manifest.json', dict(
            n_cells=adata.n_obs, n_genes=adata.n_vars,
            split_counts=adata.obs['split'].astype(str).value_counts().to_dict(),
            slice_split_counts=adata.obs.groupby(['spatial_slice', 'split'], observed=True).size().reset_index(name='n_cells').to_dict(orient='records'),
            obs_names_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            var_names_sha256=hashlib.sha256('\n'.join(self.var_names).encode()).hexdigest(),
            fixed_validation_seed=self.rng_seed, training_seed_definition='seed + epoch + 1',
            graph_preserved=True, donors_remain_within_slice=True))

    def _sample(self, seed):
        config = dict(k_neighbors=9, admixture_target_mode='pool_fraction', lambda_max=.25,
            only_sample_unexpressed_genes=True,
            min_sampled_transcript_threshold=int(self.admixture_args['min_sampled_transcript_threshold']))
        added, stats = sample_weighted(self._pool_native, self._pool_obs, self.var_names,
            self._pool_graph, np.arange(len(self._pool_obs)), self._pool_weights, config,
            seed=int(seed), batch_size=256)
        if self._pool_native.multiply(added).nnz:
            raise AssertionError('Training admixture touched native-positive genes')
        write_json(self.cache_dir/f'sampling_seed_{seed}.json', dict(seed=int(seed),
            lambda_max=.25, native_transcripts=int(self._pool_native.sum()),
            added_transcripts=int(added.sum()), n_cells=added.shape[0], n_genes=added.shape[1],
            mean_achieved_admixture_burden=float(stats.achieved_admixture_burden.mean()),
            only_unexpressed_genes_added=True, whole_gene_pool_targets_valid=True))
        return added


class MilestoneReached(Exception):
    pass


class MilestoneTrainer(HeimdallTrainer):
    """Finish after the comparison checkpoint without changing its LR schedule."""
    last_instance = None
    stop_after_milestone = 29

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_completed_training_epoch = None
        type(self).last_instance = self

    def train_epoch(self, epoch):
        result = super().train_epoch(epoch)
        self._last_completed_training_epoch = epoch
        return result

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)
        # fit also saves before train_epoch when validation improves. Stop only
        # after the regular save following the completed requested training epoch.
        if epoch == self.stop_after_milestone and self._last_completed_training_epoch == epoch:
            write_json(Path(self.cfg.work_dir)/'training_complete.json', dict(
                milestone=epoch, completed_training_epochs=epoch+1,
                original_scheduler_horizon_epochs=int(self.cfg.tasks.args.epochs),
                sampler='donor_type_weighted_pool_fraction', lambda_max=.25,
                inference_ready=True))
            raise MilestoneReached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, default=ROOT/'outputs/SyndriSIM_pool_fraction')
    parser.add_argument('--baseline-config', type=Path, default=ROOT/'Heimdall/admixture_prediction_results/nsclc_cosmx_train_val_k9_tau0_0p95/config.txt')
    parser.add_argument('--milestone', type=int, default=29)
    parser.add_argument('--prepare-config-only', action='store_true',
                        help='Write the frozen training configuration and recipe without starting training')
    args = parser.parse_args()
    if args.milestone != 29:
        raise ValueError('Use milestone 29 to match the original Figure 3 evaluation')
    root = args.benchmark_root.resolve()
    selection = json.loads((root/'search/selected_parameters.json').read_text())
    if selection['lambda_max'] != .25 or selection['uses_held_out_test_for_selection']:
        raise ValueError('Expected the frozen training-only lambda=0.25 calibration')
    weights_path = root/'calibration/donor_weights.csv'
    weights = pd.read_csv(weights_path)
    config = OmegaConf.load(args.baseline_config)
    old_config = OmegaConf.to_container(config, resolve=True)
    original_work_dir = config.work_dir
    config.run_name = 'nsclc_cosmx_train_val_k9_pool_fraction_lambda0p25'
    config.work_dir = str(root/'syndri_model')
    config.dataset.dataset_name = 'nsclc_cosmx_spillover_train_val_pool_fraction_lambda0p25'
    sampling = config.dataset.preprocess_args.admixture_args
    sampling.pop('tau_range', None)
    sampling.sampler = 'donor_type_weighted_pool_fraction'
    sampling.admixture_target_mode = 'pool_fraction'
    sampling.lambda_max = .25
    sampling.donor_type_weights = dict(zip(weights.donor_type, weights.weight.astype(float)))
    sampling.frozen_weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    sampling.sampler_sha256 = hashlib.sha256(Path(__file__).with_name('donor_weighted_admixture.py').read_bytes()).hexdigest()
    config_path = root/'manifests/syndri_training_config.yaml'
    config_path.parent.mkdir(exist_ok=True)
    OmegaConf.save(config, config_path)
    changed_config = OmegaConf.to_container(config, resolve=True)
    for key in ['model', 'tasks', 'scheduler', 'trainer', 'optimizer', 'fc', 'fe', 'fg', 'loss', 'seed', 'float_dtype']:
        if old_config[key] != changed_config[key]:
            raise AssertionError(f'Original training hyperparameters changed: {key}')
    if config.dataset.preprocess_args.data_path != old_config['dataset']['preprocess_args']['data_path']:
        raise AssertionError('Training input and split must remain unchanged')
    write_json(root/'manifests/syndri_training_recipe.json', dict(
        baseline_config=str(args.baseline_config.resolve()), original_work_dir=original_work_dir,
        output_checkpoint_dir=config.work_dir, milestone=29, nominal_epochs=int(config.tasks.args.epochs),
        planned_training_epochs=30, same_training_hyperparameters=True, original_train_val_input_unchanged=True,
        fresh_training=True, validation_admixture_fixed=True, training_admixture_resampled_each_epoch=True,
        lambda_max=.25, donor_weights_file=str(weights_path),
        donor_weights_sha256=sampling.frozen_weights_sha256,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if args.prepare_config_only:
        print(f'Prepared matched training recipe: {config_path}', flush=True)
        return
    representations.AdmixtureEpochStore = PoolFractionEpochStore
    import train as training_entry
    training_entry.HeimdallTrainer = MilestoneTrainer
    print(f'Original training recipe; pool-fraction sampler; train through milestone {args.milestone}', flush=True)
    try:
        training_entry.main.__wrapped__(config)
    except MilestoneReached:
        trainer = MilestoneTrainer.last_instance
        if trainer.run_wandb:
            trainer.accelerator.end_training()
        print(f'Completed matched training budget; checkpoint ready: {config.work_dir}/model-29.pt', flush=True)


if __name__ == '__main__':
    main()
