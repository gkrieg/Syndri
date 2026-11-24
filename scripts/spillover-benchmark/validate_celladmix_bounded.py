#!/usr/bin/env python3
"""Compare bounded annotation with the untouched installed implementation."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from celladmix_annotation_bounded import stage, global_pairs, atomic_json, HERE


def fixture(root):
    rng = np.random.default_rng(5418)
    n_cells, n_tx, factors = 120, 70, 3
    cell = np.repeat(np.arange(n_cells),n_tx)
    types = np.array(['A','B','C'])[np.arange(n_cells)%3]
    centers = np.c_[np.arange(n_cells)%12, np.arange(n_cells)//12,
                    np.zeros(n_cells)]
    xyz = centers[cell]+rng.normal(0,.45,(len(cell),3))
    # Give each type a dominant factor and retain genuine neighboring overlap.
    f = rng.integers(1,4,len(cell))
    native = rng.random(len(cell))<.65
    f[native] = cell[native]%3+1
    ids = np.array([f'cell_{i:04d}' for i in range(n_cells)])
    tx = pd.DataFrame(dict(mol_id=np.arange(len(cell)),cell_id=ids[cell],
        cell_type=types[cell],gene='G',x=xyz[:,0],y=xyz[:,1],z=xyz[:,2]))
    ass = tx[['mol_id','cell_id','gene','cell_type']].copy();ass['factor']=f
    tx.to_csv(root/'transcripts.csv',index=False,float_format='%.17g')
    ass.to_csv(root/'assignments.csv',index=False)
    pd.DataFrame(dict(cell=ids,celltype=types)).to_csv(root/'metadata.csv',index=False)
    atomic_json(root/'ready.json',dict(n_transcripts=len(cell)))
    return factors


def actual_subset(root,source,maximum=60):
    # Preserve all molecules for the first complete cells. Original simulator
    # emits each recipient contiguously; the extra cell proves the boundary.
    chunks=[];assigned=[];n_cells=0
    a=pd.read_csv(source/'factorization/Lung12/assignments.csv',chunksize=250000)
    for tx in pd.read_csv(source.parent/'transcript_method_inputs/Lung12/transcripts_celladmix.csv',chunksize=250000):
        chunks.append(tx);assigned.append(next(a));n_cells=pd.concat(chunks).cell_id.nunique()
        if n_cells>maximum:break
    tx=pd.concat(chunks,ignore_index=True);ass=pd.concat(assigned,ignore_index=True)
    ids=tx.cell_id.drop_duplicates().iloc[:maximum]
    keep=tx.cell_id.isin(ids)
    tx=tx.loc[keep].copy();ass=ass.loc[keep].copy()
    tx['mol_id']=np.arange(len(tx));ass['mol_id']=tx.mol_id.to_numpy()
    k=int(ass.factor.max())
    tx.to_csv(root/'transcripts.csv',index=False,float_format='%.17g')
    ass.to_csv(root/'assignments.csv',index=False)
    tx[['cell_id','cell_type']].drop_duplicates().rename(columns={'cell_id':'cell','cell_type':'celltype'}).to_csv(root/'metadata.csv',index=False)
    atomic_json(root/'ready.json',dict(n_transcripts=len(tx)))
    return k


def validate(root,kind,source):
    root.mkdir(parents=True,exist_ok=False)
    k=fixture(root) if kind=='fixture' else actual_subset(root,source)
    work=root/'cache'
    stage(root/'transcripts.csv',root/'assignments.csv',root/'metadata.csv',work,k,chunk_rows=997)
    global_pairs(work,chunk_rows=701)
    report=root/'equivalence.json'
    subprocess.run(['/home/jwinchel/miniconda3/envs/r_env/bin/Rscript',
        str(HERE/'validate_celladmix_bounded.R'),str(HERE),str(work),str(k),str(report)],check=True)
    n=json.loads((work/'staged.json').read_text())['n_transcripts']
    xyz=np.memmap(work/'xyz.dat',mode='r',dtype='<f8',shape=(n,3))
    indices=cKDTree(xyz).query(xyz,k=11)[1]
    mask=indices!=np.arange(n)[:,None]
    indices=indices[mask&(np.cumsum(mask,axis=1)<=10)].reshape(n,10)
    reference=np.fromfile(work/'reference_neighbors.dat',dtype='<i4').reshape(n,10)
    exact=np.all(np.sort(indices,axis=1)==np.sort(reference,axis=1),axis=1)
    result=json.loads(report.read_text())
    result['global_neighbor_sets_exact']=bool(exact.all())
    result['global_neighbor_agreement_fraction']=float(exact.mean())
    if not exact.all():raise AssertionError(f'Global neighbor disagreement: {result}')
    atomic_json(report,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['fixture','actual'],default='fixture')
    parser.add_argument('--source',type=Path,default=HERE.parents[1]/'outputs/SyndriSIM_pool_fraction/celladmix')
    args=parser.parse_args();validate(args.output,args.kind,args.source)
