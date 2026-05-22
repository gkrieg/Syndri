"""Lazy per-epoch storage for sampled admixture matrices."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from Heimdall.admixture_sampling import sample_admixture_from_neighbor_dict


class AdmixtureEpochStore:
    """Sample, cache, and retrieve one admixture CSR matrix per epoch.

    Matrices are indexed by AnnData ``obs_names``/``var_names`` so callers can
    request matrices aligned to a filtered or reordered AnnData object.
    """

    def __init__(
        self,
        adata,
        admixture_args,
        cache_dir,
        split_col="split",
        train_key="train",
        rng_seed=0,
        fixed_epoch=None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.admixture_args = admixture_args
        self.rng_seed = int(rng_seed)
        self.split_col = split_col
        self.train_key = train_key
        self.fixed_epoch = None if fixed_epoch is None else int(fixed_epoch)

        self.obs_names = pd.Index(adata.obs_names.astype(str))
        self.var_names = pd.Index(adata.var_names.astype(str))
        if self.obs_names.has_duplicates:
            raise ValueError("AdmixtureEpochStore requires unique adata.obs_names.")
        if self.var_names.has_duplicates:
            raise ValueError("AdmixtureEpochStore requires unique adata.var_names.")

        obs = adata.obs[[split_col]].copy() if split_col in adata.obs else adata.obs.iloc[:, 0:0].copy()
        var = pd.DataFrame(index=self.var_names.copy())
        self._sampling_adata = ad.AnnData(X=csr_matrix(adata.X), obs=obs, var=var)
        self._sampling_adata.uns["neighbor_dict"] = dict(adata.uns["neighbor_dict"])

        if split_col in adata.obs:
            self._val_test_mask = (adata.obs[split_col].to_numpy() != train_key)
        else:
            self._val_test_mask = np.zeros(adata.n_obs, dtype=bool)

        self._current_epoch = None
        self._current_matrix = None
        self._fixed_matrix = None

    def get(self, epoch, obs_names=None, var_names=None):
        matrix = self._get_full_epoch(epoch)
        return self._align(matrix, obs_names=obs_names, var_names=var_names)

    def get_binary(self, epoch, obs_names=None, var_names=None):
        matrix = self.get(epoch, obs_names=obs_names, var_names=var_names)
        return matrix.astype(bool).astype(np.float32)

    def _get_full_epoch(self, epoch):
        epoch = int(epoch)
        if self.fixed_epoch is not None:
            epoch = self.fixed_epoch
        if self._current_epoch == epoch and self._current_matrix is not None:
            return self._current_matrix

        path = self.cache_dir / f"admixed_epoch_{epoch}.npz"
        if path.exists():
            matrix = load_npz(path).tocsr()
        else:
            matrix = self._sample_epoch(epoch)
            save_npz(path, matrix)
        matrix.sum_duplicates()
        matrix.sort_indices()

        self._current_epoch = epoch
        self._current_matrix = matrix
        return matrix

    def _get_fixed_matrix(self):
        if self._fixed_matrix is not None:
            return self._fixed_matrix

        path = self.cache_dir / "admixed_fixed_val_test.npz"
        if path.exists():
            matrix = load_npz(path).tocsr()
        else:
            matrix = self._sample(seed=self.rng_seed)
            save_npz(path, matrix)
        matrix.sum_duplicates()
        matrix.sort_indices()

        self._fixed_matrix = matrix
        return matrix

    def _sample_epoch(self, epoch):
        matrix = self._sample(seed=self.rng_seed + int(epoch) + 1)
        if self._val_test_mask.any():
            fixed = self._get_fixed_matrix()
            matrix = matrix.tolil()
            matrix[self._val_test_mask, :] = fixed[self._val_test_mask, :]
            matrix = matrix.tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def _sample(self, seed):
        matrix, _ = sample_admixture_from_neighbor_dict(
            self._sampling_adata,
            min_sampled_transcript_threshold=self.admixture_args["min_sampled_transcript_threshold"],
            mask_out_random_genes=self.admixture_args["mask_out_random_genes"],
            mask_out_frac=self.admixture_args["mask_out_frac"],
            tau_range=tuple(self.admixture_args["tau_range"]),
            rng=np.random.default_rng(seed),
            return_sparse=True,
        )
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def _align(self, matrix, obs_names=None, var_names=None):
        obs_indexer = self._indexer(self.obs_names, obs_names)
        var_indexer = self._indexer(self.var_names, var_names)

        if obs_indexer is not None:
            matrix = matrix[obs_indexer, :]
        if var_indexer is not None:
            matrix = matrix[:, var_indexer]
        return matrix.tocsr()

    @staticmethod
    def _indexer(source_names, requested_names):
        if requested_names is None:
            return None

        requested = pd.Index(requested_names.astype(str) if hasattr(requested_names, "astype") else requested_names)
        if requested.equals(source_names):
            return None

        indexer = source_names.get_indexer(requested)
        if (indexer < 0).any():
            missing = requested[indexer < 0].tolist()[:5]
            raise KeyError(f"Names are missing from AdmixtureEpochStore: {missing}")
        return indexer
