import argparse
from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from split_sparsity import cached_profiles, run, sparsity_stats
from rctd_annotation import build_reference, build_rctd_config, ensure_common_genes


class SplitSparsityTest(unittest.TestCase):
    def test_reference_bounds_inclusive_and_independent_of_sigma(self):
        for matrix in [np.array([[0., 9.], [10., 0.], [0., 2001.], [0., 2000.]]),
                       sparse.csr_matrix([[0., 9.], [10., 0.], [0., 2001.], [0., 2000.]])]:
            reference = ad.AnnData(
                matrix, obs=pd.DataFrame({"cell_type": ["A", "A", "A", "B"]},
                                         index=["low", "min", "high", "max"]),
                var=pd.DataFrame(index=["g1", "g2"]))
            obj = build_reference(reference, cell_min=1)
            np.testing.assert_array_equal(obj.profiles.T, [[1., 0.], [0., 1.]])
        self.assertEqual(build_rctd_config().UMI_min_sigma, 100)

    def test_low_panel_counts_retained_and_zero_cells_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = ad.AnnData(
                sparse.csr_matrix([[2., 0.], [0., 0.], [0., 3.]]),
                obs=pd.DataFrame({"cell_type": ["A", "A", "B"]},
                                 index=["a", "zero", "b"]),
                var=pd.DataFrame(index=["g1", "g2"]),
            )
            path = root / "ref.h5ad"
            reference.write_h5ad(path)
            profiles, _ = cached_profiles(
                path, reference.var_names, root / "cache", cell_type_col="cell_type",
                cell_min=1, n_max_cells=10000, min_UMI=1)
            np.testing.assert_array_equal(profiles.X, [[1., 0.], [0., 1.]])
            self.assertTrue(np.isfinite(profiles.X).all())

    def test_sparse_explicit_zeros_and_empty_type(self):
        matrix = sparse.csr_matrix(([0., 2.], ([0, 1], [0, 1])), shape=(2, 3))
        stats = sparsity_stats(matrix)
        self.assertEqual(stats["n_nonzero"], 1)
        self.assertEqual(stats["zero_fraction"], 5 / 6)
        self.assertEqual(stats["mean_profile_zero_fraction"], 2 / 3)
        self.assertTrue(np.isnan(sparsity_stats(matrix[:0])["zero_fraction"]))

    def test_annotation_parity_cache_and_paired_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The off-panel gene must not contribute to UMI normalization.
            reference = ad.AnnData(
                sparse.csr_matrix([[100., 0., 900.], [50., 50., 900.],
                                   [0., 100., 900.], [0., 100., 900.]]),
                obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]},
                                 index=["r1", "r2", "r3", "r4"]),
                var=pd.DataFrame(index=["g1", "g2", "off_panel"]),
            )
            ref_path = root / "reference.h5ad"
            reference.write_h5ad(ref_path)
            spatial = ad.AnnData(
                sparse.csr_matrix([[2., 1.], [0., 3.], [9., 9.]]),
                obs=pd.DataFrame(index=["s1", "s2", "excluded"]),
                var=pd.DataFrame(index=["g2", "g1"]),
            )
            spatial.layers["split_purified_counts_X"] = sparse.csr_matrix(
                [[0., 1.], [0., 0.], [9., 9.]])
            spatial.obsm["rctd_weights_X"] = np.array([[1., 0.], [0., 1.], [0., 0.]])
            spatial.uns["rctd_cell_type_names_X"] = ["A", "B"]
            spatial_path = root / "spatial.h5ad"
            spatial.write_h5ad(spatial_path)
            settings = dict(cell_type_col="cell_type", cell_min=1, n_max_cells=10000, min_UMI=100)
            profiles, path = cached_profiles(ref_path, spatial.var_names, root / "cache", **settings)
            _, reference_common = ensure_common_genes(spatial, reference)
            expected = build_reference(reference_common, **settings)
            np.testing.assert_array_equal(profiles.X, expected.profiles.T)
            np.testing.assert_allclose(profiles.X[0], [.25, .75])
            mtime = path.stat().st_mtime_ns
            _, reused = cached_profiles(ref_path, spatial.var_names, root / "cache", **settings)
            self.assertEqual(path, reused)
            self.assertEqual(mtime, reused.stat().st_mtime_ns)
            _, changed = cached_profiles(ref_path, spatial.var_names[::-1], root / "cache", **settings)
            self.assertNotEqual(path, changed)
            run(argparse.Namespace(
                spatial=str(spatial_path), reference=str(ref_path), layers="X",
                result_layer_prefix="split_purified_counts", cell_type_col="cell_type",
                spatial_cell_type_col=None, cell_min=1, n_max_cells=10000,
                reference_min_umi=100, reference_max_umi=2000,
                cache_dir=root / "cache", output_dir=root / "out"))
            result = pd.read_csv(next((root / "out").glob("*.csv")))
            self.assertEqual(len(result), 6)
            self.assertTrue((result.n_excluded_cells == 1).all())
            fractions = result.set_index(["cell_type", "stage"]).zero_fraction
            self.assertEqual(fractions["A", "spatial_before"], 0)
            self.assertEqual(fractions["A", "spatial_after"], .5)
            self.assertEqual(fractions["B", "spatial_before"], .5)
            self.assertEqual(fractions["B", "spatial_after"], 1)


if __name__ == "__main__":
    unittest.main()
