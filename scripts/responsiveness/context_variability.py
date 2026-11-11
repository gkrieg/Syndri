"""Natural malignant-neighbor variation in the full eligible cell population."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy import sparse
from all_types_panel import PROJECT,TYPE_FAMILY


def variation(run,index):
 import anndata as ad
 sys.path.insert(0,str(PROJECT/'Heimdall'))
 from Heimdall.admixture_sampling import build_neighbor_dict,build_neighbor_celltype_dict
 run=Path(run);design=json.loads((run/'design.json').read_text());entry=design['slices'][index]
 out=run/'context_variability'/entry['slice'];out.mkdir(parents=True,exist_ok=False)
 source=ad.read_h5ad(entry['input'])
 coords=source.obs[['x_centroid','y_centroid']].apply(pd.to_numeric,errors='coerce')
 valid=source.obs.harmony_cell_type.notna().to_numpy() & np.isfinite(coords.to_numpy()).all(1)
 a=ad.AnnData(X=sparse.csr_matrix(source.X,dtype=np.float32)[valid],obs=source.obs.loc[valid].copy(),var=source.var.copy())
 del source
 count=np.asarray(a.X.sum(1)).ravel();genes=np.asarray((a.X>0).sum(1)).ravel()
 types=a.obs.harmony_cell_type.astype(str).to_numpy();ids=a.obs_names.astype(str).to_numpy()
 values={};neighbor_counts={}
 for name,allow_same in [('model_context',False),('physical_context',True)]:
  a=build_neighbor_dict(a,k_neighbors=5,coord_cols=('x_centroid','y_centroid'),tissue_col=None,
                       cell_type_col='harmony_cell_type',min_source_counts=10,use_same_cell_type_neighbors=allow_same)
  a=build_neighbor_celltype_dict(a,cell_type_col='harmony_cell_type')
  contexts=a.uns['neighbor_celltype_dict']
  n=np.array([len(contexts[c]) for c in ids]);m=np.array([sum(str(t)=='malignant cell' for t in contexts[c]) for c in ids])
  values[name]=np.divide(m,n,out=np.zeros(len(n),dtype=float),where=n>0);neighbor_counts[name]=n
 eligible=(count>=10)&(genes>=5)&(neighbor_counts['model_context']==5)&(neighbor_counts['physical_context']==5)
 summaries=[];distributions=[]
 for t in TYPE_FAMILY:
  mask=eligible&(types==t);n=int(mask.sum())
  for definition in values:
   v=values[definition][mask]
   hist=np.bincount(np.rint(v*5).astype(int),minlength=6)
   for j,c in enumerate(hist):distributions.append(dict(slice=entry['slice'],cell_type=t,definition=definition,malignant_neighbor_proportion=j/5,n_cells=int(c),fraction=c/n if n else np.nan))
   qs=np.quantile(v,[0,.1,.25,.5,.75,.9,1]) if n else np.full(7,np.nan)
   positive=hist[hist>0]/n if n else np.array([])
   summaries.append(dict(slice=entry['slice'],cell_type=t,definition=definition,n_cells=n,
    tested=n>=design['min_cells_per_type'],mean=float(v.mean()) if n else np.nan,
    sd=float(v.std(ddof=0)) if n else np.nan,minimum=qs[0],q10=qs[1],q25=qs[2],median=qs[3],q75=qs[4],q90=qs[5],maximum=qs[6],
    iqr=qs[4]-qs[2],q90_minus_q10=qs[5]-qs[1],n_distinct_proportions=int((hist>0).sum()),
    fraction_zero=hist[0]/n if n else np.nan,fraction_one=hist[5]/n if n else np.nan,
    entropy_bits=float(-(positive*np.log2(positive)).sum()),
    sufficient_family_variation=bool(n and (hist>0).sum()>=3 and qs[6]-qs[0]>=.2)))
 pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
 pd.DataFrame(distributions).to_csv(out/'distribution.csv',index=False)
 (out/'complete.json').write_text(json.dumps(dict(slice=entry['slice'],n_eligible=int(eligible.sum()),
    model_context='Five nearest eligible neighbors excluding the query exact cell type',
    physical_context='Five nearest eligible neighbors allowing the query exact cell type',source_counts='>10',
    note='Both definitions evaluated on same eligible query cells; not all unfiltered cells'),indent=2)+'\n')
 print('Context variation:',entry['slice'],int(eligible.sum()),flush=True)

if __name__=='__main__':variation(Path(sys.argv[1]),int(sys.argv[2]))
