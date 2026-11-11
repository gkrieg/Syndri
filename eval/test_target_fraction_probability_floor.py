"""Regression checks for the target-fraction eligibility floor."""
import unittest

import numpy as np
from scipy import sparse

import metrics_eval as metrics
import score_admixture_slice as scoring


class TargetFractionProbabilityFloorTests(unittest.TestCase):
    def setUp(self):
        self.counts = sparse.csr_matrix([[5., 5., 5.]])
        self.probs = np.array([[.9, .5, .49]])

    def test_zeroing_stops_at_half_probability(self):
        args = metrics.build_parser().parse_args([])
        for counts in (self.counts, self.counts.toarray()):
            cleaned, threshold, achieved, _ = metrics.cleaned_counts_for_removed_fraction(
                counts, self.probs, 1.0, args)
            self.assertEqual(threshold, .5)
            self.assertAlmostEqual(achieved, 2 / 3)
            np.testing.assert_array_equal(
                cleaned.toarray() if sparse.issparse(cleaned) else cleaned,
                [[0., 0., 5.]],
            )
        self.assertEqual(
            scoring.threshold_for_removed_fraction(self.counts, self.probs, 1.0),
            (.5, 2 / 3),
        )

    def test_probability_scaling_never_touches_ineligible_gene(self):
        args = metrics.build_parser().parse_args(["--cleaning-mode", "probability_scale"])
        for counts in (self.counts, self.counts.toarray()):
            cleaned, _, achieved, scale = metrics.cleaned_counts_for_removed_fraction(
                counts, self.probs, 1.0, args)
            self.assertTrue(np.isinf(scale))
            self.assertAlmostEqual(achieved, 2 / 3)
            np.testing.assert_array_equal(
                cleaned.toarray() if sparse.issparse(cleaned) else cleaned,
                [[0., 0., 5.]],
            )

    def test_override_and_no_eligible_genes(self):
        args = metrics.build_parser().parse_args([
            "--cleaning-mode", "probability_scale", "--clean-prob-threshold", "0"])
        cleaned, _, achieved, _ = metrics.cleaned_counts_for_removed_fraction(
            self.counts, self.probs, 1.0, args)
        self.assertEqual(achieved, 1.0)
        np.testing.assert_array_equal(cleaned.toarray(), [[0., 0., 0.]])
        low = np.array([[.1, .2, .3]])
        cleaned, _, achieved, _ = metrics.cleaned_counts_for_removed_fraction(
            self.counts, low, .5,
            metrics.build_parser().parse_args(["--cleaning-mode", "probability_scale"]))
        self.assertEqual(achieved, 0.0)
        np.testing.assert_array_equal(cleaned.toarray(), self.counts.toarray())
        self.assertEqual(metrics.threshold_for_removed_fraction(low, self.counts, .5),
                         (np.inf, 0.0))

    def test_cell_type_specific_target_fraction(self):
        counts = sparse.csr_matrix([[4., 6.], [3., 7.]])
        probs = np.array([[.9, .2], [.4, .5]])
        args = metrics.build_parser().parse_args(["--cell-type-specific-removal"])
        cleaned, _, achieved, *_ = metrics.cleaned_counts_for_removed_fraction_maybe_grouped(
            counts, probs, 1.0, args, labels=["A", "B"])
        np.testing.assert_array_equal(cleaned.toarray(), [[0., 6.], [3., 0.]])
        self.assertAlmostEqual(achieved, 11 / 20)


if __name__ == "__main__":
    unittest.main()
