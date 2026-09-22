"""Regression checks for the count-weighted cellAdmix adaptation.

Run: python -m unittest discover -s eval -p 'test_score_admixture_slice.py'
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import score_admixture_slice as cli
from admix_scores import mask_out_genes, estimate_cell_adjacency
from sc_score_from_st import get_scrna_admixture_scores_from_spatial_K as score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.ref = ad.AnnData(
            sparse.csr_matrix([[1., 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]]),
            obs=pd.DataFrame({'cell_type': ['A', 'A', 'B', 'B']}, index=['r1', 'r2', 'r3', 'r4']),
            var=pd.DataFrame(index=['g1', 'g2', 'absent']),
        )
        self.st = ad.AnnData(
            sparse.csr_matrix([[1., 9, 0], [2, 8, 0], [0, 0, 0]]),
            obs=pd.DataFrame({'cell_type': ['A', 'B', 'A']}, index=['s1', 's2', 'empty']),
            var=self.ref.var.copy(),
        )
        self.K = pd.DataFrame([[1., 1], [1, 1]], index=['A', 'B'], columns=['A', 'B'])

    def test_count_weighting_and_background(self):
        # Prior=.25*.5=.125; q(g2,A)=2/9, q(g1,B)=1.
        # Reference null(A)=(.5*2/9)/1.5=2/27; null(B)=0.
        for dense in [False, True]:
            st = self.st.copy()
            if dense:
                st.X = st.X.toarray()
            for adjust in [False, True]:
                result = score(self.ref, st, self.K, signal_thres=.1, adjust=adjust).set_index('cell')
                expected = [.2 - (2/27 if adjust else 0), .2]
                np.testing.assert_allclose(result.loc[['s1', 's2'], 'scores'], expected)
                self.assertTrue(np.isnan(result.loc['empty', 'scores']))

    def test_gene_mask_rows_and_unchanged_denominator(self):
        result, sets = score(self.ref, self.st, self.K, signal_thres=.1,
                             adjust=False, exclude_genes=['g2'], return_gene_sets=True)
        self.assertTrue((sets['cont_probs_per_type'].loc['g2'] == 0).all())
        np.testing.assert_allclose(result.set_index('cell').loc[['s1', 's2'], 'scores'], [0, .2])
        cp = pd.DataFrame([[.2, .7], [.4, .8]], index=['g1', 'g2'], columns=['A', 'B'])
        masked = mask_out_genes(cp, {'A': ['g1'], 'all': ['g2']})
        np.testing.assert_allclose(masked, [[0, .7], [0, 0]])
        self.assertEqual(cp.loc['g1', 'A'], .2)

    def test_gene_order_and_missing_reference_labels(self):
        ref = self.ref.copy()
        ref.obs.loc['r3', 'cell_type'] = None
        result = score(ref, self.st[:, ['g2', 'absent', 'g1']], self.K,
                       signal_thres=.1, adjust=False).set_index('cell')
        np.testing.assert_allclose(result.loc[['s1', 's2'], 'scores'], [.2, .2])

    def test_normalized_correlation_matches_dense_calculation(self):
        rng = np.random.default_rng(13)
        a = rng.poisson(3, (25, 7)).astype(float)
        b = rng.poisson(4, (31, 7)).astype(float)
        a[:, 2] = b[:, 2] = 0
        aa = ad.AnnData(sparse.csr_matrix(a), var=pd.DataFrame(index=list('abcdefg')))
        bb = ad.AnnData(sparse.csr_matrix(b), var=aa.var.copy())
        args = SimpleNamespace(skip_correlation_filter=False, correlation_threshold=.2)
        excluded, table = cli.correlation_exclusions(aa, bb, args)
        ca = np.corrcoef(np.log1p(a / a.sum(axis=1)[:, None] * 1e4)[:, [0,1,3,4,5,6]].T)
        cb = np.corrcoef(np.log1p(b / b.sum(axis=1)[:, None] * 1e4)[:, [0,1,3,4,5,6]].T)
        expected = [np.corrcoef(np.delete(ca[:, i], i), np.delete(cb[:, i], i))[0, 1]
                    for i in range(6)]
        np.testing.assert_allclose(table.set_index('gene').loc[list('abdefg'), 'correlation'], expected, atol=1e-12)
        self.assertIn('c', excluded)
        self.assertEqual(table.set_index('gene').loc['c', 'reason'], 'not_expressed_in_both')

    def test_deterministic_adjacency(self):
        rng = np.random.default_rng(0)
        xy = rng.random((40, 2))
        df = pd.DataFrame({'x':xy[:,0], 'y':xy[:,1], 'cell':[str(i) for i in range(40)],
                           'cell_type':['A', 'B']*20})
        pd.testing.assert_frame_equal(estimate_cell_adjacency(df, random_state=42),
                                      estimate_cell_adjacency(df, random_state=42))

    def test_normalization_preserves_original_panel_totals(self):
        X = np.array([[1., 9, 90], [5, 5, 0]])
        normalized_full = cli.log_normalize_counts(X).toarray()[:, :2]
        normalized_subset = cli.log_normalize_counts(X[:, :2], totals=X.sum(axis=1)).toarray()
        np.testing.assert_allclose(normalized_subset, normalized_full)

    def test_cli_raw_split_sweep_and_metadata(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refpath, stpath = root/'ref.h5ad', root/'slice.h5ad'
            self.ref.write_h5ad(refpath)
            st = ad.AnnData(sparse.csr_matrix(rng.integers(0, 10, (30, 3))),
                obs=pd.DataFrame({'cell_type':['A', 'B']*15, 'x':rng.random(30), 'y':rng.random(30)},
                                 index=[f's{i}' for i in range(30)]), var=self.ref.var.copy())
            st.layers['syndri_admix_probs_X'] = rng.random(st.shape)
            st.layers['split_purified_counts_X'] = st.X * .5
            st.uns['rctd_run_metadata_X'] = {'reference_file': str(refpath)}
            st.write_h5ad(stpath)
            argv = ['score_admixture_slice.py', '--spatial-h5ad', str(stpath), '--output-dir', str(root/'out'),
                    '--spatial-cell-type-key', 'cell_type', '--reference-cell-type-key', 'cell_type',
                    '--coord-x', 'x', '--coord-y', 'y', '--target-removed-fractions', '.1', '--skip-correlation-filter']
            with patch.object(sys, 'argv', argv):
                cli.main()
            result = pd.read_csv(root/'out/admixture_scores.csv')
            self.assertEqual(set(result.setting), {'raw', 'SPLIT', 'syndri_remove_0.1'})
            self.assertEqual(len(result), 90)
            pivot = result.pivot(index='cell', columns='setting', values='scores')
            np.testing.assert_allclose(pivot['raw'], pivot['SPLIT'])
            self.assertTrue((result.score_version == '2_count_weighted').all())


if __name__ == '__main__':
    unittest.main()
