import sys
import types
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

rctd_stub = types.ModuleType("rctd")


class DummyRctdConfig:
    pass


class DummyReference:
    def __init__(self, *args, **kwargs):
        pass


def run_rctd(*args, **kwargs):
    raise NotImplementedError


rctd_stub.RCTDConfig = DummyRctdConfig
rctd_stub.Reference = DummyReference
rctd_stub.run_rctd = run_rctd
sys.modules.setdefault("rctd", rctd_stub)

from rctd_annotation import find_matching_layers, pad_result_array


class FindMatchingLayersTest(unittest.TestCase):
    def test_find_matching_layers_matches_regex_pattern(self) -> None:
        adata = ad.AnnData(X=np.ones((3, 3)))
        adata.layers["trial_0"] = np.ones((3, 3))
        adata.layers["trial_1"] = np.ones((3, 3))
        adata.layers["trial_2"] = np.ones((3, 3))

        self.assertEqual(
            find_matching_layers(adata, r"^(X|trial_(0|1|2))$"),
            ["X", "trial_0", "trial_1", "trial_2"],
        )

    def test_pad_result_array_uses_mask_and_defaults(self) -> None:
        values = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=float)
        mask = np.array([True, False, True, False])

        padded = pad_result_array(values, mask, n_obs=4)

        self.assertEqual(padded.shape, (4, 2))
        self.assertTrue(np.allclose(padded[0], values[0]))
        self.assertTrue(np.allclose(padded[2], values[1]))
        self.assertTrue(np.allclose(padded[1], np.zeros(2)))
        self.assertTrue(np.allclose(padded[3], np.zeros(2)))


if __name__ == "__main__":
    unittest.main()
