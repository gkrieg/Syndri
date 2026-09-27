from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from syndri_thresholds import select_cell_threshold, clean_cells_by_reference


class ThresholdTests(unittest.TestCase):
    def test_all_candidates_match_brute_force(self):
        rng = np.random.default_rng(13)
        for window in [1, 3, 100]:
            for _ in range(20):
                x = rng.integers(0, 20, 37).astype(float)
                p = rng.integers(0, 8, len(x)) / 7
                r = rng.random(len(x))
                result = select_cell_threshold(x, p, r, window=window, return_trace=True)
                thresholds = np.r_[np.inf, np.sort(np.unique(p[x > 0]))[::-1]]
                scores = []
                for threshold in thresholds:
                    cleaned = np.where(p >= threshold, 0, x)
                    denominator = np.linalg.norm(cleaned) * np.linalg.norm(r)
                    scores.append(cleaned @ r / denominator if denominator else np.nan)
                rolling = pd.Series(scores).rolling(window, min_periods=1).mean().to_numpy()
                rolling[~np.isfinite(scores)] = np.nan
                expected = np.flatnonzero(np.isclose(rolling, np.nanmax(rolling),
                                                   rtol=1e-12, atol=1e-14))[0]
                np.testing.assert_allclose(result.trace["cosine"], scores, atol=1e-14)
                np.testing.assert_allclose(result.trace["rolling_cosine"], rolling, atol=1e-14)
                self.assertEqual(result.threshold, thresholds[expected])
                self.assertAlmostEqual(result.cosine_after, scores[expected])

    def test_ties_are_removed_together_and_full_scan_recovers_after_dip(self):
        x = np.array([1., 10., 1.])
        p = np.array([.9, .8, .1])
        r = np.array([1., 0., 10.])
        result = select_cell_threshold(x, p, r, window=1, return_trace=True)
        self.assertLess(result.trace["cosine"][1], result.cosine_before)
        self.assertEqual(result.threshold, .8)
        tied = select_cell_threshold([10., 10., 1.], [.9, .9, .1], [0., 0., 1.], window=1)
        self.assertEqual(tied.n_removed_genes, 2)
        self.assertEqual(tied.threshold, .9)

    def test_sparse_duplicates_explicit_zeros_and_stability(self):
        x = sparse.csr_matrix(([1., 2., 0., 4.], [0, 0, 1, 2], [0, 4]), shape=(1, 3))
        a = select_cell_threshold(x, [.9, .8, .1], [0., 0., 1.], window=1)
        b = select_cell_threshold([3., 0., 4.], [.9, .8, .1], [0., 0., 1.], window=1)
        self.assertEqual(a, b)
        stable = select_cell_threshold([1e12, 1.], [.9, .1], [0., 1.], window=1)
        self.assertEqual(stable.cosine_after, 1.)

    def test_zero_cases_and_invalid_input(self):
        self.assertEqual(select_cell_threshold([0., 0.], [.9, .1], [1., 0.]).status, "zero_cell")
        self.assertEqual(select_cell_threshold([1., 0.], [.9, .1], [0., 0.]).status, "zero_reference")
        with self.assertRaises(ValueError):
            select_cell_threshold([1.], [1.1], [1.])
        with self.assertRaises(ValueError):
            select_cell_threshold([-1.], [.1], [1.])
        with self.assertRaises(ValueError):
            select_cell_threshold([1.], [.1], [1.], window=0)

    def test_batch_sparse_dense_parity_and_missing_profile(self):
        x = np.array([[10., 1., 0.], [1., 8., 2.], [1., 2., 3.], [0., 0., 0.]])
        p = np.array([[.9, .1, 0.], [.8, .9, .1], [.9, .8, .1], [0., 0., 0.]])
        profiles = pd.DataFrame([[0., 1., 0.], [1., 0., 1.]], index=["A", "B"])
        labels = ["A", "B", "unknown", "A"]
        a, da = clean_cells_by_reference(x, p, labels, profiles, window=1)
        b, db = clean_cells_by_reference(sparse.csr_matrix(x), sparse.csr_matrix(p), labels, profiles, window=1)
        np.testing.assert_array_equal(a.toarray(), b.toarray())
        pd.testing.assert_frame_equal(da, db)
        np.testing.assert_array_equal(a.toarray()[2], x[2])
        self.assertEqual(da.status.tolist(), ["ok", "ok", "missing_reference", "zero_cell"])
        self.assertTrue(da.loc[:1, "selected_k"].notna().all())
        self.assertTrue(da.loc[:1, "threshold"].isna().all())

    def test_cell_type_expressed_mean_ordering_and_shared_k(self):
        x = np.array([[10., 1.], [10., 1.]])
        probabilities = np.array([[.9, .1], [.2, .8]])
        profiles = pd.DataFrame([[0., 1.]], index=["A"])

        grouped, diagnostics, curves = clean_cells_by_reference(
            x, probabilities, ["A", "A"], profiles, return_curves=True)

        # Mean probabilities are [.55, .45], so both cells rank gene 0 first.
        np.testing.assert_array_equal(grouped.toarray(), [[0., 1.], [0., 1.]])
        self.assertTrue((diagnostics.probability_ordering == "cell_type_expressed_mean").all())
        self.assertTrue((diagnostics.n_expressed_genes_before == 2).all())
        self.assertTrue((diagnostics.selected_k == 1).all())
        self.assertEqual(curves.loc[curves.k.eq(1), "removed_gene"].iloc[0], 0)
        self.assertAlmostEqual(curves.loc[curves.k.eq(1), "removed_gene_score"].iloc[0], .55)

    def test_probabilities_are_averaged_only_where_gene_is_expressed(self):
        x = np.array([[10., 1., 0.], [10., 1., 1.]])
        probabilities = np.array([[.9, .1, 0.], [.2, .8, .99]])
        profiles = pd.DataFrame([[0., 1., 1.]], index=["A"])

        _, diagnostics, curves = clean_cells_by_reference(
            x, probabilities, ["A", "A"], profiles, return_curves=True)

        # Gene 2's score is .99, not .495: cell 0 does not express it and is
        # excluded from that gene's probability mean.
        self.assertEqual(curves.loc[curves.k.eq(1), "removed_gene"].iloc[0], 2)
        self.assertAlmostEqual(curves.loc[curves.k.eq(1), "removed_gene_score"].iloc[0], .99)
        self.assertEqual(diagnostics.selected_k.nunique(), 1)

    def test_cell_type_curve_and_one_se_selection_match_brute_force(self):
        rng = np.random.default_rng(91)
        x = rng.integers(0, 8, size=(7, 9)).astype(float)
        p = rng.random(x.shape)
        reference = rng.random(9)
        profiles = pd.DataFrame([reference], index=["A"])

        cleaned, diagnostics, curves = clean_cells_by_reference(
            sparse.csr_matrix(x), sparse.csr_matrix(p), ["A"] * len(x),
            profiles, one_se_multiplier=1.0, return_curves=True)

        scores = np.array([
            p[x[:, gene] > 0, gene].mean() if np.any(x[:, gene] > 0) else -np.inf
            for gene in range(x.shape[1])
        ])
        ranking = np.argsort(-scores, kind="stable")
        baseline = np.array([
            row @ reference / (np.linalg.norm(row) * np.linalg.norm(reference))
            for row in x
        ])
        deltas = []
        for k in range(x.shape[1] + 1):
            candidate = x.copy()
            candidate[:, ranking[:k]] = 0
            cosine = np.array([
                row @ reference / (np.linalg.norm(row) * np.linalg.norm(reference))
                if np.linalg.norm(row) else 0.0
                for row in candidate
            ])
            deltas.append(cosine - baseline)
        deltas = np.asarray(deltas)
        mean = deltas.mean(axis=1)
        se = deltas.std(axis=1, ddof=0) / np.sqrt(len(x))
        k_max = int(np.argmax(mean))
        selected = int(np.flatnonzero(mean >= mean[k_max] - se[k_max] - 1e-14)[0])

        np.testing.assert_allclose(curves.mean_cosine_improvement, mean, atol=1e-14)
        np.testing.assert_allclose(curves.bootstrap_se, se, atol=1e-14)
        self.assertEqual(diagnostics.selected_k.iloc[0], selected)
        expected = x.copy()
        expected[:, ranking[:selected]] = 0
        np.testing.assert_array_equal(cleaned.toarray(), expected)


if __name__ == "__main__":
    unittest.main()
