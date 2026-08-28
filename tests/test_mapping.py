import sys
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from mapping import map_sc_reference_to_unique_xenium_cells


def make_adata(n_obs, embedding, obs=None):
    adata = ad.AnnData(X=np.ones((n_obs, 2)))
    if obs is not None:
        for col, values in obs.items():
            adata.obs[col] = values
    adata.obsm["X_pca_harmony"] = np.asarray(embedding, dtype=float)
    return adata


class MapScReferenceToUniqueXeniumCellsTest(unittest.TestCase):
    def test_uses_every_sc_reference_row_when_spatial_has_more_cells(self):
        sc_data = make_adata(
            3,
            embedding=[
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
            ],
        )
        sc_data.obs_names = ["sc0", "sc1", "sc2"]

        st_data = make_adata(
            5,
            embedding=[
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
            ],
            obs={
                "x_centroid": [0, 1, 2, 3, 4],
                "y_centroid": [0, 0, 0, 0, 0],
            },
        )

        mapped = map_sc_reference_to_unique_xenium_cells(
            sc_data,
            st_data,
            n_matches=1,
            reuse_penalty=0.0,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(set(mapped.obs["source_sc_id"].astype(str)), {"sc0", "sc1", "sc2"})
        self.assertEqual(mapped.uns["sc_reference_to_spatial_mapping"]["sc_coverage"], 1.0)
        self.assertGreater(
            mapped.uns["sc_reference_to_spatial_mapping"]["n_forced_sc_coverage_assignments"],
            0,
        )

    def test_can_restrict_mapping_to_matching_cell_types(self):
        sc_data = make_adata(
            2,
            embedding=[
                [100.0, 0.0],
                [0.0, 0.0],
            ],
            obs={"cell_type": ["A", "B"]},
        )
        sc_data.obs_names = ["sc_A", "sc_B"]

        st_data = make_adata(
            2,
            embedding=[
                [0.0, 0.0],
                [100.0, 0.0],
            ],
            obs={
                "x_centroid": [0, 1],
                "y_centroid": [0, 0],
                "cell_type": ["A", "B"],
            },
        )

        mapped = map_sc_reference_to_unique_xenium_cells(
            sc_data,
            st_data,
            n_matches=1,
            reuse_penalty=0.0,
            match_cell_type=True,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(mapped.obs["source_sc_id"].astype(str).tolist(), ["sc_A", "sc_B"])
        self.assertTrue((mapped.obs["cell_type"].astype(str).to_numpy() == ["A", "B"]).all())

    def test_cell_type_restriction_raises_when_spatial_type_has_no_sc_candidate(self):
        sc_data = make_adata(
            1,
            embedding=[[0.0, 0.0]],
            obs={"cell_type": ["A"]},
        )
        st_data = make_adata(
            1,
            embedding=[[0.0, 0.0]],
            obs={
                "x_centroid": [0],
                "y_centroid": [0],
                "cell_type": ["B"],
            },
        )

        with self.assertRaisesRegex(ValueError, "No SC reference cells found"):
            map_sc_reference_to_unique_xenium_cells(
                sc_data,
                st_data,
                n_matches=1,
                match_cell_type=True,
            )


if __name__ == "__main__":
    unittest.main()
