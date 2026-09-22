from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from split_h5ad_io import merge_layer, sample_id_for


class MergeLayerDtypeTest(unittest.TestCase):
    def write_split_output(self, root, h5ad_path):
        sample_dir = root / sample_id_for(h5ad_path, "X")
        sample_dir.mkdir(parents=True)
        # On disk SPLIT is genes x cells. Both values would become zero if
        # assigned into the source int32 matrix.
        mmwrite(sample_dir / "split_purified_counts.mtx",
                sparse.csr_matrix([[0.25], [0.75]], dtype=np.float64))
        (sample_dir / "split_purified_counts_genes.txt").write_text("g1\ng2\n")
        (sample_dir / "split_purified_counts_cells.txt").write_text("c1\n")

    def make_adata(self):
        return ad.AnnData(
            X=sparse.csr_matrix(np.array([[2, 3], [4, 5]], dtype=np.int32)),
            obs=pd.DataFrame(index=["c1", "c2"]),
            var=pd.DataFrame(index=["g1", "g2"]),
        )

    def test_fractional_counts_survive_with_original_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5ad_path = root / "input.h5ad"
            self.write_split_output(root, h5ad_path)
            adata = self.make_adata()
            result = merge_layer(adata, h5ad_path, "X", root, "split", "original", False)
            output = adata.layers["split_X"]
            self.assertTrue(np.issubdtype(output.dtype, np.floating))
            np.testing.assert_array_equal(output.toarray(), [[0.25, 0.75], [4., 5.]])
            self.assertEqual(result["output_dtype"], "float64")

    def test_fractional_counts_survive_with_zero_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5ad_path = root / "input.h5ad"
            self.write_split_output(root, h5ad_path)
            adata = self.make_adata()
            merge_layer(adata, h5ad_path, "X", root, "split", "zero", False)
            output = adata.layers["split_X"]
            np.testing.assert_array_equal(output.toarray(), [[0.25, 0.75], [0., 0.]])


if __name__ == "__main__":
    unittest.main()
