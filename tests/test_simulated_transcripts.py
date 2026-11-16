"""Small synthetic-mask tests; the real-data pilot is restricted to 100 cells."""
import hashlib
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'recomb_reproducibility'))
sys.path.insert(0, str(ROOT/'Heimdall'))

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from syndri_repro.transcript_locations import (MaskGeometry, donor_allocations, gene_targets,
    integer_counts, load_inputs, real_transcript_profile, simulate_transcripts, validate_transcripts)


class TranscriptLocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root/'raw'
        (self.raw/'CellLabels').mkdir(parents=True)
        self.labels = {}
        first = np.zeros((16,16),dtype=np.uint16)
        first[2:7,6:13] = 1
        first[1:4,1:4] = 2
        second = np.zeros_like(first)
        second[2:7,1:5] = 1
        self.labels = {1:first,2:second}
        for fov, mask in self.labels.items():
            Image.fromarray(mask).save(self.raw/f'CellLabels/CellLabels_F{fov:03d}.tif')
        positions = pd.DataFrame({'fov':[1,2], 'x_global_px':[100.,116.], 'y_global_px':[-200.,-200.]})
        positions.to_csv(self.raw/'Lung12_fov_positions_file.csv',index=False)
        metadata = []
        for fov, cid in [(1,1),(2,1),(1,2)]:
            rr, cc = np.where(self.labels[fov]==cid)
            x, y = cc.mean()+.5, 16-rr.mean()-.5
            origin = positions.set_index('fov').loc[fov]
            metadata.append(dict(fov=fov,cell_ID=cid,CenterX_local_px=x,CenterY_local_px=y,
                                 CenterX_global_px=x+origin.x_global_px,CenterY_global_px=y+origin.y_global_px))
        self.cells = pd.DataFrame(metadata,index=['Lung12:1:1','Lung12:2:1','Lung12:1:2'])
        self.var = pd.DataFrame({'gene_symbol':['A','B']},index=['gene-A','gene-B'])
        spatial = ad.AnnData(sparse.csr_matrix((3,2)),obs=self.cells.copy(),var=self.var.copy())
        self.spatial = self.root/'spatial.h5ad'
        spatial.write_h5ad(self.spatial)
        obs = self.cells[['CenterX_global_px','CenterY_global_px']].copy()
        obs['matched_st_id'] = self.cells.index
        obs.index = ['Lung12:'+x for x in obs.index]
        obs['source_sc_id'] = ['sc-query','sc-donor1','sc-donor2']
        obs['cell_type'] = ['query','donor','donor']
        obs['spatial_slice'] = 'Lung12'
        obs['split'] = 'test'
        native = sparse.csr_matrix([[1000,0],[0,12],[0,8]],dtype=np.int64)
        added = sparse.csr_matrix([[0,20],[0,0],[0,0]],dtype=np.int64)
        simulated = ad.AnnData(native+added,obs=obs,var=self.var.copy())
        simulated.layers['original_counts'] = native
        simulated.layers['synthetic_admixture_counts'] = added
        simulated.uns['neighbor_dict'] = {str(obs.index[0]):obs.index[1:].to_numpy()}
        self.input = self.root/'simulated.h5ad'
        simulated.write_h5ad(self.input)
        examples=[]
        for fov in [1,2]:
            rr, cc = np.where(self.labels[fov]==1)
            x, y = cc[:100]+.5, 16-rr[:100]-.5
            origin = positions.set_index('fov').loc[fov]
            examples.append(pd.DataFrame(dict(fov=fov,cell_ID=1,x_global_px=x+origin.x_global_px,
                y_global_px=y+origin.y_global_px,x_local_px=x,y_local_px=y,
                z=np.resize([2,4,6],len(x)),target=np.resize(['A','B'],len(x)),CellComp='Cytoplasm')))
        pd.concat(examples).to_csv(self.raw/'Lung12_tx_file.csv',index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def run_fixture(self, name='outputs'):
        return simulate_transcripts(self.input,self.spatial,self.raw,self.root/name,fov=1,seed=31)

    def geometry(self):
        geom = MaskGeometry(self.raw,'Lung12',self.root/'geometry')
        geom.flip_y = {1:True,2:True}
        return geom

    def test_end_to_end_seven_invariants_and_cross_fov_ids(self):
        path, report = self.run_fixture()
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(report['n_transcripts'],1028)
        self.assertEqual(report['n_native'],1008)
        self.assertEqual(report['n_admixed'],20)
        self.assertEqual(report['n_cross_fov_admixed'],12)
        frame = pd.read_csv(path,keep_default_na=False)
        self.assertTrue(frame.CellComp.eq('').all())
        self.assertEqual(set(frame.z),{2,4,6})
        self.assertEqual(set(frame.fov),{1})
        self.assertEqual(set(frame.loc[frame.is_admixed,'source_fov']),{1,2})

    def test_reproducible_csv(self):
        first, _ = self.run_fixture('first')
        second, _ = self.run_fixture('second')
        self.assertEqual(hashlib.sha256(first.read_bytes()).digest(),hashlib.sha256(second.read_bytes()).digest())

    def test_gene_specific_proportions_preserve_exact_pool(self):
        obj,native,added,cells,var,name=load_inputs(self.input,self.spatial)
        allocations, origin = donor_allocations(obj,native,added,np.array([0]),7)
        self.assertEqual(allocations['count'].tolist(),[12,8])
        np.testing.assert_allclose(allocations.gene_specific_source_proportion,[.6,.4])

    def test_recover_original_neighbor_pool_and_reject_mismatch(self):
        obj,native,added,cells,var,name=load_inputs(self.input,self.spatial)
        del obj.uns['neighbor_dict']
        obj.uns['synthetic_admixture_config'] = dict(k_neighbors=2,min_source_counts=0,
            use_same_cell_type_neighbors=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            allocations, origin = donor_allocations(obj,native,added,np.array([0]),7)
        self.assertTrue(origin.startswith('reconstructed'))
        self.assertEqual(dict(zip(allocations.source_cell,allocations['count'])),
                         {str(obj.obs_names[1]):12,str(obj.obs_names[2]):8})
        altered = added.copy()
        altered[0,1] = 19
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with self.assertRaisesRegex(ValueError,'exactly explain'):
                donor_allocations(obj,native,altered,np.array([0]),7)

    def test_no_added_counts(self):
        obj = ad.read_h5ad(self.input)
        obj.layers['synthetic_admixture_counts'] = sparse.csr_matrix(obj.shape,dtype=np.int64)
        obj.X = obj.layers['original_counts'].copy()
        obj.write_h5ad(self.input)
        path, report = self.run_fixture()
        self.assertEqual(report['n_transcripts'],1008)
        self.assertEqual(report['n_admixed'],0)
        self.assertTrue(all(report['checks'].values()))

    def test_exact_region_distance_and_weighting(self):
        geom = self.geometry()
        donor = geom.region(2,1)
        # Donor rectangle in global pixels: x=[117,121], y=[-191,-186].
        points = np.array([[110,-192],[118,-190],[125,-185],[101,-205]],dtype=float)
        outside = np.maximum(np.maximum([117,-191]-points,points-[121,-186]),0.)
        expected = np.linalg.norm(outside,axis=1)*geom.scale
        np.testing.assert_allclose(geom.distance(points*geom.scale,donor),expected,atol=1e-12)
        recipient = geom.region(1,1)
        uniform, _ = geom.sample(recipient,20000,np.random.default_rng(4))
        biased, _ = geom.sample(recipient,20000,np.random.default_rng(4),donor,2.)
        self.assertTrue(geom.contains(1,1,uniform).all())
        self.assertTrue(geom.contains(1,1,biased).all())
        self.assertGreater(biased[:,0].mean(),uniform[:,0].mean()+.1)

    def test_reject_training_and_fractional_counts(self):
        obj = ad.read_h5ad(self.input)
        obj.obs['split'] = 'train'
        obj.write_h5ad(self.root/'training.h5ad')
        with self.assertRaisesRegex(ValueError,'held-out'):
            load_inputs(self.root/'training.h5ad',self.spatial)
        with self.assertRaisesRegex(ValueError,'integer counts'):
            integer_counts(sparse.csr_matrix([[.5]]))

    def test_validation_detects_corruption(self):
        path, _ = self.run_fixture()
        obj,native,added,cells,var,name=load_inputs(self.input,self.spatial)
        profile,examples=real_transcript_profile(self.raw/'Lung12_tx_file.csv',self.root/'profile')
        symbols=gene_targets(obj,var,profile['targets'])
        allocations=pd.read_parquet(path.parent/'donor_allocations.parquet')
        original=pd.read_csv(path,keep_default_na=False)
        changes=[('z',999),('x_global_px',-99999.),('source_fov',999),('gene_id','unknown-gene')]
        for key,value in changes:
            with self.subTest(key=key):
                frame=original.copy();frame.loc[0,key]=value
                damaged=self.root/f'bad_{key}.csv';frame.to_csv(damaged,index=False)
                with self.assertRaises(AssertionError):
                    validate_transcripts(damaged,obj,native,added,np.array([0,2]),cells,
                        allocations,self.geometry(),profile,symbols)


if __name__=='__main__':
    unittest.main()
