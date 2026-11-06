"""Nearest annotated malignant-centroid distance and responsiveness diagnostics."""
from pathlib import Path
import json,sys
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata,spearmanr
from all_types_panel import MARKERS,EXAMPLES,dense


def summarize_distance(run):
 run=Path(run);out=run/'summaries_distance';out.mkdir(exist_ok=False)
 design=json.loads((run/'design.json').read_text());baseline_rows=[];context_rows=[];audits=[]
 for entry in design['slices']:
  source=ad.read_h5ad(entry['input'],backed='r')
  try:obs=source.obs.copy()
  finally:source.file.close()
  if not obs.index.is_unique:raise ValueError('Duplicate source identifiers')
  coords=obs[['x_centroid','y_centroid']].apply(pd.to_numeric,errors='coerce').to_numpy()
  finite=np.isfinite(coords).all(axis=1)
  tumor=finite&obs.harmony_cell_type.astype(str).eq('malignant cell').to_numpy()
  if not tumor.any():raise ValueError(f'No malignant cells in {entry["slice"]}')
  tree=cKDTree(coords[tumor]);distance=np.full(len(obs),np.nan)
  distance[finite]=tree.query(coords[finite],k=1)[0]
  distances=pd.Series(distance,index=obs.index.astype(str))
  a=ad.read_h5ad(entry['predictions']);o=a.obs.reset_index(drop=True)
  probabilities=dense(a.layers['admix_probs']);logits=np.log(np.clip(probabilities.astype(float),1e-6,1-1e-6));logits-=np.log1p(-np.clip(probabilities.astype(float),1e-6,1-1e-6))
  raw=sparse.csr_matrix(a.X)
  names=a.var.get('feature_name',pd.Series(a.var_names,index=a.var_names)).fillna('').astype(str).to_numpy()
  names=np.where(names=='',a.var_names.astype(str),names)
  focal_dist=o.focal_cell_id.astype(str).map(distances).to_numpy()
  donor_dist=o.context_source_cell_id.astype(str).map(distances).to_numpy()
  if not np.isfinite(focal_dist).all() or not np.isfinite(donor_dist).all():raise ValueError('Unmatched/missing coordinates')
  if not np.allclose(focal_dist[o.context_kind.eq('original')],donor_dist[o.context_kind.eq('original')]):raise ValueError('Baseline distance mismatch')
  for q,group in o.groupby('focal_cell_id',observed=True,sort=False):
   ix=group.index.to_numpy();base=group.index[group.context_kind.eq('original')].to_numpy()
   if len(base)!=1:raise ValueError('Expected one baseline')
   b=base[0];expressed=np.flatnonzero(raw[b].toarray().ravel()>0)
   marker_indices=np.flatnonzero(np.isin(names[expressed],MARKERS['Malignant']))
   meta=dict(slice=entry['slice'],query_cell_id=q,query_cell_type=o.loc[b,'query_cell_type'],
             query_distance_um=float(focal_dist[b]),n_expressed_genes=len(expressed),n_expressed_malignant_markers=len(marker_indices))
   score=float(logits[b,expressed].mean()) if len(expressed) else np.nan
   marker_score=float(logits[b,expressed[marker_indices]].mean()) if len(marker_indices) else np.nan
   rank_valid=len(expressed)>=10
   ranks=rankdata(probabilities[ix][:,expressed],axis=1,method='average') if rank_valid else None
   percentiles=(ranks-1)/(len(expressed)-1) if rank_valid else None
   local=int(np.flatnonzero(ix==b)[0])
   baseline_marker_rank=float(percentiles[local,marker_indices].mean()) if rank_valid and len(marker_indices) else np.nan
   baseline_rows.append({**meta,'mean_expressed_gene_logit':score,'malignant_marker_logit':marker_score,
                         'malignant_marker_rank_percentile':baseline_marker_rank})
   if (meta['query_cell_type'],'Malignant') in EXAMPLES and len(marker_indices):
    selected_genes=expressed[marker_indices]
    for j,row in enumerate(ix):
     kind=o.loc[row,'context_kind']
     context_rows.append({**meta,'context_kind':kind,'context_source_cell_id':o.loc[row,'context_source_cell_id'],
          'context_source_distance_um':float(donor_dist[row]),'delta_context_source_distance_um':float(donor_dist[row]-focal_dist[b]),
          'delta_marker_logit':float((logits[row,selected_genes]-logits[b,selected_genes]).mean()),
          'delta_marker_rank_percentile':float((percentiles[j,marker_indices]-percentiles[local,marker_indices]).mean()) if rank_valid else np.nan})
  audits.append(dict(slice=entry['slice'],n_annotated_malignant_cells=int(tumor.sum()),n_finite_coordinate_cells=int(finite.sum()),n_queries=int(o.focal_cell_id.nunique())))
  del a,raw,probabilities,logits
 baseline=pd.DataFrame(baseline_rows);contexts=pd.DataFrame(context_rows)
 baseline.to_csv(out/'query_original_distance_scores.csv',index=False)
 contexts.to_csv(out/'example_distance_responses.csv.gz',index=False,compression='gzip')
 pd.DataFrame(audits).to_csv(out/'distance_audit.csv',index=False)
 correlations=[]
 for (slice_id,cell_type),frame in baseline.groupby(['slice','query_cell_type']):
  if cell_type=='malignant cell':continue
  for metric in ['mean_expressed_gene_logit','malignant_marker_logit','malignant_marker_rank_percentile']:
   good=frame[['query_distance_um',metric]].dropna();status='ok'
   if len(good)<10:status='insufficient_queries'
   elif good.query_distance_um.nunique()<3 or good[metric].nunique()<3:status='insufficient_variation'
   rho=float(spearmanr(good.query_distance_um,good[metric]).statistic) if status=='ok' else np.nan
   correlations.append(dict(slice=slice_id,query_cell_type=cell_type,metric=metric,n_queries=len(good),spearman_rho=rho,status=status))
 pd.DataFrame(correlations).to_csv(out/'original_distance_correlations.csv',index=False)
 # Within-query slopes for donor-context distance hold expression and query type fixed.
 slopes=[]
 for (slice_id,q),frame in contexts[contexts.context_kind.eq('representative')].groupby(['slice','query_cell_id']):
  for metric in ['delta_marker_logit','delta_marker_rank_percentile']:
   good=frame[['context_source_distance_um',metric]].dropna();status='ok'
   if len(good)<10 or good.context_source_distance_um.nunique()<3:status='insufficient_variation'
   rho=float(spearmanr(good.context_source_distance_um,good[metric]).statistic) if status=='ok' and good[metric].nunique()>1 else np.nan
   slopes.append(dict(slice=slice_id,query_cell_id=q,query_cell_type=frame.query_cell_type.iloc[0],metric=metric,
                      spearman_rho=rho,n_contexts=len(good),status=status))
 pd.DataFrame(slopes).to_csv(out/'query_context_distance_correlations.csv',index=False)
 controls=contexts[contexts.context_kind.eq('unchanged_control')]
 assert np.allclose(controls.delta_context_source_distance_um,0)
 assert np.allclose(controls.delta_marker_logit,0)
 assert np.allclose(controls.delta_marker_rank_percentile.dropna(),0)
 (out/'complete.json').write_text(json.dumps(dict(n_slices=len(design['slices']),n_queries=len(baseline),units='micrometers',
   distance='2D Euclidean centroid distance to nearest annotated malignant cell in same slice; all finite-coordinate malignant cells, no source-count filter',
   malignant_queries='Self included: distance zero; omitted from proximity-response plots',
   context_donor_distance='Distance at the donor location, not physical relocation of the query; raw distance and delta relative to original query both exported'),indent=2)+'\n')
 print('Distance summaries saved:',out,flush=True)

if __name__=='__main__':summarize_distance(Path(sys.argv[1]))
