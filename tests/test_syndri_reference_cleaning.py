import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from syndri_reference_cleaning import load_profiles, prepare_reference_cleaned


class ReferenceCleaningTests(unittest.TestCase):
    def test_only_additional_counts_skips_baseline_and_sweep(self):
        import metrics_eval as metrics

        adata = ad.AnnData(
            X=np.array([[5., 1.], [1., 4.]]),
            obs=pd.DataFrame({"ct": ["A", "B"]}, index=["c1", "c2"]),
            var=pd.DataFrame(index=["g1", "g2"]),
        )
        adata.layers["probs"] = np.zeros(adata.shape)
        adata.layers["split"] = np.array([[2., .5], [.25, 3.]])
        args = metrics.build_parser().parse_args([
            "--label-key", "ct", "--admix-probs-layer", "probs",
            "--additional-counts-layer", "split", "--additional-counts-case-label", "SPLIT",
            "--target-removed-fractions", ".1,.2", "--only-additional-counts"])

        def score(tissue, source, case, matrix, labels, options, **kwargs):
            self.assertEqual(case, "SPLIT")
            np.testing.assert_array_equal(matrix, adata.layers["split"])
            return {"case": case, "error": ""}

        with patch.object(metrics, "score_case", side_effect=score) as mocked:
            rows = metrics.run_cell_type_separation(adata, "t", "f", args)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual([row["case"] for row in rows], ["SPLIT"])

    def test_cache_identity_gene_alignment_and_saved_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.h5ad"
            ad.AnnData(np.ones((1, 2)), var=pd.DataFrame(index=["a", "b"])).write_h5ad(reference_path)
            cache = root / "cache"
            cache.mkdir()
            profiles = ad.AnnData(np.array([[1., 0.]]),
                                 obs=pd.DataFrame(index=["T"]), var=pd.DataFrame(index=["b", "a"]))
            stat = reference_path.stat()
            profiles.uns["cache_metadata_json"] = json.dumps(dict(
                reference_file=str(reference_path), genes=["a", "b"],
                size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                settings=dict(min_UMI=10, max_UMI=2000, cell_min=25,
                              n_max_cells=10000, cell_type_col="cell_type")))
            profile_path = cache / "profiles.h5ad"
            profiles.write_h5ad(profile_path)
            spatial = ad.AnnData(sparse.csr_matrix([[10., 7., 1.], [1., 2., 3.]]),
                                obs=pd.DataFrame({"ct": ["T", "unknown"]}, index=["s1", "s2"]),
                                var=pd.DataFrame(index=["a", "unshared", "b"]))
            spatial.layers["probs"] = np.array([[.9, 1., .1], [.9, .9, .9]])
            spatial.uns["rctd_run_metadata_X"] = {"reference_file": str(reference_path)}
            _, selected = load_profiles(spatial, cache)
            self.assertEqual(selected, profile_path)
            args = argparse.Namespace(syndri_reference_profiles=str(cache), reference_data="",
                                      syndri_cell_type_key="ct", label_key="ct", admix_probs_layer="probs",
                                      syndri_cosine_window=1, output_dir=str(root / "out"))
            cleaned, achieved, _ = prepare_reference_cleaned(spatial, args, "source.h5ad")
            np.testing.assert_array_equal(cleaned.toarray(), [[0., 7., 1.], [1., 2., 3.]])
            self.assertAlmostEqual(achieved, 10 / 24)
            diagnostics = pd.read_csv(next((root / "out/syndri_thresholds").glob("*.csv")))
            self.assertEqual(diagnostics.cell_id.tolist(), ["s1", "s2"])
            self.assertEqual(diagnostics.status.tolist(), ["ok", "missing_reference"])
            # Two cache configurations must not be selected arbitrarily.
            profiles.write_h5ad(cache / "other.h5ad")
            with self.assertRaisesRegex(ValueError, "2 matching"):
                load_profiles(spatial, cache)
            load_profiles(spatial, profile_path)

    def test_metric_families_receive_same_cleaned_counts(self):
        import metrics_eval as metrics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profiles.h5ad"
            ad.AnnData(np.array([[0., 1.]]), obs=pd.DataFrame(index=["T"]),
                       var=pd.DataFrame(index=["a", "b"])).write_h5ad(profile_path)
            spatial = ad.AnnData(np.array([[10., 1.]]),
                                obs=pd.DataFrame({"ct": ["T"]}, index=["s1"]),
                                var=pd.DataFrame(index=["a", "b"]))
            spatial.layers["probs"] = np.array([[.9, .1]])
            args = metrics.build_parser().parse_args([
                "--syndri-reference-cosine", "--syndri-reference-profiles", str(profile_path),
                "--syndri-cosine-window", "1", "--label-key", "ct",
                "--admix-probs-layer", "probs", "--output-dir", str(root)])

            def separation(work, tissue, source, options):
                np.testing.assert_array_equal(work.X.toarray(), [[0., 1.]])
                self.assertEqual(options.target_removed_fractions, "")
                self.assertEqual(options.additional_counts_layer, "")
                return [{"case": "admixed", "error": ""}]

            def tcell(work, tissue, source, options, **kwargs):
                np.testing.assert_array_equal(work.X.toarray(), [[0., 1.]])
                self.assertEqual(options.cleaning_mode, "zero")
                return tuple(pd.DataFrame({"condition": ["placeholder"], "error": [""]}) for _ in range(4))

            with patch.object(metrics, "run_cell_type_separation", side_effect=separation), \
                    patch.object(metrics, "run_tcell_contamination", side_effect=tcell):
                rows, *frames = metrics.run_reference_cosine_evaluation(spatial, "tissue", "source", args)
            self.assertEqual(rows[0]["case"], "syndri_reference_cosine")
            self.assertEqual(rows[0]["cleaning_mode"], "zero")
            for frame in frames:
                self.assertEqual(frame.condition.iloc[0], "syndri_reference_cosine")
                self.assertEqual(frame.cosine_window.iloc[0], 1)
            np.testing.assert_array_equal(spatial.X, [[10., 1.]])


if __name__ == "__main__":
    unittest.main()
