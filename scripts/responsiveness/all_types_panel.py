"""Balanced all-type context swaps and figure summaries; no model fitting."""
from pathlib import Path
import argparse,json,sys,subprocess
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT=Path('/home/jwinchel/Syndri')
FAMILIES={
 'Lymphoid':['CD4-positive, alpha-beta memory T cell','cytotoxic T cell','natural T-regulatory cell','natural killer cell','plasma cell'],
 'Myeloid / innate':['classical monocyte','macrophage','mast cell','plasmacytoid dendritic cell'],
 'Stromal / vascular':['capillary endothelial cell','fibroblast of breast','fibroblast of lung','pericyte','smooth muscle cell'],
 'Malignant':['malignant cell']}
TYPE_FAMILY={t:f for f,ts in FAMILIES.items() for t in ts}
MARKERS={
 'Lymphoid':['CD3D','CD3E','TRAC','CD247','NKG7','GNLY','MS4A1','CD79A','JCHAIN','MZB1'],
 'Myeloid / innate':['LST1','TYROBP','FCER1G','AIF1','CD68','CSF1R','TPSAB1','TPSB2','CLEC4C'],
 'Stromal / vascular':['COL1A1','COL1A2','COL3A1','DCN','LUM','PECAM1','VWF','RGS5','ACTA2','TAGLN'],
 'Malignant':['KRT8','KRT18','KRT19','EPCAM','CEACAM6','KRT17']}
EXAMPLES=[('cytotoxic T cell','Malignant'),('fibroblast of breast','Malignant')]

def dense(x):return x.toarray() if sparse.issparse(x) else np.asarray(x)
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def task_entry(run,index):return json.loads((run/'design.json').read_text())['slices'][index]

def prepare(run,index):
 import anndata as ad
 sys.path.insert(0,str(PROJECT/'Heimdall'))
 from Heimdall.admixture_sampling import build_neighbor_dict,build_neighbor_celltype_dict
 sys.path.insert(0,str(PROJECT/'Heimdall/scripts'))
 from prepare_responsiveness_contexts import _farthest_point_sample
 design=json.loads((run/'design.json').read_text());e=task_entry(run,index)
 out=Path(e['contexts']);out.parent.mkdir(exist_ok=True,parents=True)
 if out.exists():raise FileExistsError(out)
 source=ad.read_h5ad(e['input'])
 X=sparse.csr_matrix(source.X,dtype=np.float32)
 totals=np.asarray(X.sum(1)).ravel();genes=np.asarray((X>0).sum(1)).ravel()
 labels=source.obs.harmony_cell_type
 coords=source.obs[['x_centroid','y_centroid']].apply(pd.to_numeric,errors='coerce')
 valid=labels.notna().to_numpy() & np.isfinite(coords.to_numpy()).all(1)
 unknown=set(labels[valid].astype(str))-set(TYPE_FAMILY)
 if unknown:raise ValueError(f'Unmapped types: {unknown}')
 source=ad.AnnData(X=X[valid],obs=source.obs.loc[valid].copy(),var=source.var.copy())
 source.uns['cell_type_dict']={t:-(i+1) for i,t in enumerate(sorted(TYPE_FAMILY))}
 source=build_neighbor_dict(source,k_neighbors=5,coord_cols=('x_centroid','y_centroid'),tissue_col=None,
                           cell_type_col='harmony_cell_type',min_source_counts=10,use_same_cell_type_neighbors=False)
 source=build_neighbor_celltype_dict(source,cell_type_col='harmony_cell_type')
 types=list(TYPE_FAMILY);lookup={t:i for i,t in enumerate(types)}
 ids=source.obs_names.astype(str).to_numpy();contexts=source.uns['neighbor_celltype_dict']
 composition=np.zeros((source.n_obs,len(types)),np.float32)
 for i,cid in enumerate(ids):
  for t in contexts[cid]:composition[i,lookup[str(t)]]+=1
 denom=composition.sum(1);composition=np.divide(composition,denom[:,None],out=np.zeros_like(composition),where=denom[:,None]>0)
 total=np.asarray(source.X.sum(1)).ravel();detected=np.asarray((source.X>0).sum(1)).ravel()
 rng=np.random.default_rng(design['seed']+index)
 rows=[];positions=[];neighbor_lists=[];coverage=[]
 for ti,t in enumerate(types):
  candidates=np.flatnonzero((source.obs.harmony_cell_type.astype(str).to_numpy()==t)&(total>=10)&(detected>=5)&(denom==5))
  if len(candidates)<design['min_cells_per_type']:
   coverage.append(dict(cell_type=t,n_eligible=len(candidates),status='insufficient_cells',n_queries=0,n_contexts=0));continue
  queries=np.sort(rng.choice(candidates,min(design['n_queries'],len(candidates)),replace=False))
  _,unique=np.unique(composition[candidates],axis=0,return_index=True)
  pool=candidates[unique]
  chosen=pool[_farthest_point_sample(composition[pool],min(design['n_contexts'],len(pool)),rng)]
  coverage.append(dict(cell_type=t,n_eligible=len(candidates),status='ok',n_queries=len(queries),n_contexts=len(chosen)))
  for q in queries:
   for c,kind in [(q,'original'),(q,'unchanged_control')]+[(c,'representative') for c in chosen if c!=q]:
    row=len(rows);positions.append(q);neighbor_lists.append(list(contexts[ids[c]]))
    rows.append(dict(focal_cell_id=ids[q],context_source_cell_id=ids[c],query_cell_type=t,
                     query_family=TYPE_FAMILY[t],context_kind=kind,is_original=kind=='original',
                     is_representative_context=kind=='representative',sampling_repeat=0,context_source_position=c))
 if not rows:raise ValueError('No supported query types')
 positions=np.asarray(positions);obs=source.obs.iloc[positions].copy();obs.index=[f'panel_{index}_{i:07d}' for i in range(len(rows))]
 for k in rows[0]:obs[k]=[r[k] for r in rows]
 obs['split']='val';obs['panel_slice']=e['slice']
 result=ad.AnnData(X=source.X[positions].copy(),obs=obs,var=source.var.copy())
 result.uns['cell_type_dict']=source.uns['cell_type_dict']
 result.uns['neighbor_celltype_dict']={rid:ls for rid,ls in zip(obs.index,neighbor_lists)}
 result.uns['responsiveness_panel_design_json']=json.dumps(design)
 cpositions=np.array([r['context_source_position'] for r in rows])
 result.obsm['neighborhood_proportions']=pd.DataFrame(composition[cpositions],index=obs.index,columns=types)
 result.write_h5ad(out,compression='gzip')
 pd.DataFrame(coverage).assign(slice=e['slice']).to_csv(run/'contexts'/f'{e["slice"]}_coverage.csv',index=False)
 write_json(out.with_suffix('.complete.json'),dict(n_rows=result.n_obs,n_queries=obs.focal_cell_id.nunique()))
 print('Prepared',out,result.shape,flush=True)

def infer(run,index):
 e=task_entry(run,index);d=json.loads((run/'design.json').read_text());out=Path(e['predictions'])
 if out.exists():raise FileExistsError(out)
 out.parent.mkdir(exist_ok=True,parents=True)
 cmd=[sys.executable,str(PROJECT/'Heimdall/scripts/run_admixture_inference.py'),
      '--st-data',e['contexts'],'--reference-data',d['reference'],'--checkpoint-dir',d['checkpoint'],
      '--milestone',str(d['milestone']),'--output-dir',str(out.parent),
      '--prepared-name',e['slice']+'_prepared.h5ad','--output-name',out.name,
      '--cell-type-col','harmony_cell_type','--coord-cols','x_centroid,y_centroid',
      '--batch-size','16','--use-precomputed-neighbor','--preserve-split-column',
      '--extra-overrides','fc.args.query_cell_type_col=harmony_cell_type']
 subprocess.run(cmd,check=True)
 write_json(out.with_suffix('.complete.json'),dict(input=e['contexts'],checkpoint=d['checkpoint'],milestone=d['milestone']))

def summarize(run, response_scale="probability"):
 if response_scale not in {"probability", "logit"}: raise ValueError(response_scale)
 import anndata as ad
 dest=run/('summaries_logits' if response_scale=='logit' else 'summaries');dest.mkdir(exist_ok=False)
 design=json.loads((run/'design.json').read_text());responses=[];slopes=[];details=[];marker_support=[];scale_audit=[]
 for e in design['slices']:
  p=Path(e['predictions'])
  if not p.with_suffix('.complete.json').exists():raise FileNotFoundError(f'Incomplete predictions: {p}')
  a=ad.read_h5ad(p);obs=a.obs.reset_index(drop=True)
  if 'admix_probs' not in a.layers:raise KeyError('Missing probabilities')
  probs=dense(a.layers['admix_probs']).astype(np.float32)
  if not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():raise ValueError('Invalid probabilities')
  clipped = int(((probs < 1e-6) | (probs > 1-1e-6)).sum())
  scale_audit.append(dict(slice=e['slice'],response_scale=response_scale,
                         n_values=probs.size,n_values_clipped=clipped if response_scale=='logit' else 0))
  if response_scale=='logit':
   # Transform each gene prediction before differences or averaging.
   probs=np.clip(probs.astype(np.float64),1e-6,1-1e-6)
   probs=np.log(probs)-np.log1p(-probs)

  raw=sparse.csr_matrix(a.X);names=a.var.get('feature_name',pd.Series(a.var_names,index=a.var_names)).fillna('').astype(str).to_numpy()
  names=np.where(names=='',a.var_names.astype(str),names)
  comp=a.obsm['neighborhood_proportions']
  family_comp={f:comp.reindex(columns=ts,fill_value=0).sum(axis=1).to_numpy() for f,ts in FAMILIES.items()}
  marker_idx={f:np.flatnonzero(np.isin(names,gs)) for f,gs in MARKERS.items()}
  for f,ix in marker_idx.items():marker_support.append(dict(slice=e['slice'],family=f,n_panel_markers=len(ix),markers=';'.join(names[ix])))
  for q,group in obs.groupby('focal_cell_id',sort=False,observed=True):
   ix=group.index.to_numpy();base=group.index[group.context_kind.eq('original')].to_numpy()
   if len(base)!=1:raise ValueError('Expected one original per query')
   b=base[0];expressed=raw[b].toarray().ravel()>0
   if not expressed.any():continue
   meta=dict(slice=e['slice'],query_cell_id=q,query_cell_type=obs.loc[b,'query_cell_type'],query_family=obs.loc[b,'query_family'])
   delta=probs[ix]-probs[b]
   for kind in ['representative','unchanged_control']:
    selected=group.context_kind.eq(kind).to_numpy()
    values=np.abs(delta[selected][:,expressed])
    if values.size:responses.append({**meta,'comparison':kind,f'mean_absolute_delta_{response_scale}':float(values.mean()),'n_contexts':int(selected.sum()),'n_expressed_genes':int(expressed.sum())})
   rep=group.index[group.context_kind.eq('representative')].to_numpy()
   for f,marker_columns in marker_idx.items():
    marker_columns=marker_columns[expressed[marker_columns]]
    x=family_comp[f][rep]
    status='ok'
    if len(rep)<10 or len(np.unique(x))<3 or np.ptp(x)<.2:status='insufficient_context_variation'
    if len(marker_columns)==0:status='no_expressed_markers'
    beta=np.nan
    if status=='ok':
     y=(probs[rep][:,marker_columns]-probs[b,marker_columns]).mean(axis=1)
     xc=x-x.mean();beta=float(.1*np.dot(xc,y-y.mean())/np.dot(xc,xc))
    slopes.append({**meta,'neighbor_family':f,f'delta_{response_scale}_per_10pct':beta,'status':status,
                   'n_contexts':len(rep),'n_expressed_markers':len(marker_columns),'context_range':float(np.ptp(x)) if len(x) else 0})
    if (meta['query_cell_type'],f) in EXAMPLES and len(marker_columns):
     y=(probs[rep][:,marker_columns]-probs[b,marker_columns]).mean(axis=1)
     details.extend({**meta,'neighbor_family':f,'neighbor_proportion':float(xx),f'delta_marker_{response_scale}':float(yy),'markers':';'.join(names[marker_columns])} for xx,yy in zip(x,y))
  del a,probs,raw
 response=pd.DataFrame(responses);slope=pd.DataFrame(slopes)
 response.to_csv(dest/'query_responses.csv.gz',index=False,compression='gzip')
 slope.to_csv(dest/'query_family_slopes.csv.gz',index=False,compression='gzip')
 pd.DataFrame(details).to_csv(dest/'example_context_responses.csv.gz',index=False,compression='gzip')
 pd.DataFrame(marker_support).to_csv(dest/'marker_support.csv',index=False)
 response.groupby(['slice','query_cell_type','comparison'],observed=True).agg(
   **{f'median_absolute_delta_{response_scale}':(f'mean_absolute_delta_{response_scale}','median'), 'n_queries':('query_cell_id','nunique')}).reset_index().to_csv(dest/'slice_responses.csv',index=False)
 supported=slope[slope.status.eq('ok')]
 supported.groupby(['slice','query_cell_type','neighbor_family'],observed=True).agg(
   median_slope=(f'delta_{response_scale}_per_10pct','median'),n_queries=('query_cell_id','nunique')).reset_index().to_csv(dest/'slice_family_slopes.csv',index=False)
 slope.groupby(['slice','query_cell_type','neighbor_family','status'],observed=True).size().rename('n_queries').reset_index().to_csv(dest/'support.csv',index=False)
 pd.DataFrame(scale_audit).to_csv(dest/'response_scale_audit.csv',index=False)
 write_json(dest/'complete.json',dict(n_slices=len(design['slices']),n_query_responses=len(response),response_scale=response_scale,logit_epsilon=1e-6 if response_scale=='logit' else None))
 print('Summaries saved:',dest,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','infer','summarize']);parser.add_argument('run',type=Path);parser.add_argument('--index',type=int,default=0);parser.add_argument('--response-scale',choices=['probability','logit'],default='probability');args=parser.parse_args()
 if args.mode=='summarize':summarize(args.run,response_scale=args.response_scale)
 else:globals()[args.mode](args.run,args.index)
