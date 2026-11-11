"""Responsiveness versus actual query proximity, using matched context panels."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import rankdata,spearmanr
from all_types_panel import MARKERS,dense


def shared_panel_spread(probability,expressed,malignant_marker_mask,malignant_proportions):
    """Context spread within one query; no original-context subtraction."""
    p=probability[:,expressed].astype(float)
    clipped=np.clip(p,1e-6,1-1e-6);logits=np.log(clipped)-np.log1p(-clipped)
    result=dict(n_panel_contexts=len(p),n_expressed_genes=len(expressed),
                n_expressed_malignant_markers=int(malignant_marker_mask.sum()),
                mean_gene_logit_sd=float(logits.std(axis=0,ddof=0).mean()),
                mean_gene_logit_iqr=float((np.quantile(logits,.75,axis=0)-np.quantile(logits,.25,axis=0)).mean()))
    ranks=rankdata(p,axis=1,method='average') if p.shape[1]>=10 else None
    if ranks is not None:
        centered=ranks-ranks.mean(axis=1,keepdims=True);norm=np.linalg.norm(centered,axis=1)
        denominator=norm[:,None]*norm[None,:]
        correlation=np.divide(centered@centered.T,denominator,out=np.full((len(p),len(p)),np.nan),where=denominator>0)
        pairs=(1-np.clip(correlation,-1,1))[np.triu_indices(len(p),k=1)]
        result['mean_pairwise_rank_distance']=float(np.nanmean(pairs)) if np.isfinite(pairs).any() else np.nan
        result['n_valid_context_pairs']=int(np.isfinite(pairs).sum())
    else:
        result['mean_pairwise_rank_distance']=np.nan;result['n_valid_context_pairs']=0
    result.update(malignant_marker_score_sd=np.nan,mean_malignant_gene_logit_sd=np.nan,
                  malignant_marker_rank_score_sd=np.nan,malignant_marker_slope_per_10pct=np.nan)
    if malignant_marker_mask.any():
        y=logits[:,malignant_marker_mask].mean(axis=1)
        result['malignant_marker_score_sd']=float(y.std(ddof=0))
        result['mean_malignant_gene_logit_sd']=float(logits[:,malignant_marker_mask].std(axis=0,ddof=0).mean())
        if ranks is not None:
            score=((ranks-1)/(p.shape[1]-1))[:,malignant_marker_mask].mean(axis=1)
            result['malignant_marker_rank_score_sd']=float(score.std(ddof=0))
        x=np.asarray(malignant_proportions)
        if len(x)>=10 and np.unique(x).size>=3 and np.ptp(x)>=.2:
            xc=x-x.mean();result['malignant_marker_slope_per_10pct']=float(.1*np.dot(xc,y-y.mean())/np.dot(xc,xc))
    return result


def summarize(run, marker_genes=None, output_subdir="summaries_query_distance_responsiveness"):
    marker_genes = MARKERS["Malignant"] if marker_genes is None else list(marker_genes)
    run=Path(run);out=run/output_subdir;out.mkdir(exist_ok=False)
    design=json.loads((run/'design.json').read_text());rows=[];panels=[]
    for entry in design['slices']:
        a=ad.read_h5ad(entry['predictions']);obs=a.obs.reset_index(drop=True)
        probability=dense(a.layers['admix_probs']);raw=sparse.csr_matrix(a.X)
        names=a.var.get('feature_name',pd.Series(a.var_names,index=a.var_names)).fillna('').astype(str).to_numpy()
        names=np.where(names=='',a.var_names.astype(str),names)
        comp=a.obsm['neighborhood_proportions'].reindex(columns=['malignant cell'],fill_value=0).sum(axis=1).to_numpy()
        for cell_type,type_rows in obs.groupby('query_cell_type',observed=True):
            # The union recovers the prespecified panel; a donor coinciding with
            # a query is represented by that query's original-context row.
            donors=sorted(type_rows.loc[type_rows.context_kind.eq('representative'),'context_source_cell_id'].astype(str).unique())
            if len(donors)<10:raise ValueError(f'Insufficient common panel: {entry["slice"]}, {cell_type}')
            verified=0
            for q,query_rows in type_rows.groupby('focal_cell_id',observed=True,sort=False):
                original=query_rows.index[query_rows.context_kind.eq('original')].to_numpy()
                if len(original)!=1:raise ValueError('Expected one original')
                b=original[0]
                current=query_rows.loc[~query_rows.context_kind.eq('unchanged_control')&query_rows.context_source_cell_id.astype(str).isin(donors)].copy()
                if current.context_source_cell_id.duplicated().any():raise ValueError('Duplicate context donor')
                current=current.assign(donor=current.context_source_cell_id.astype(str)).set_index('donor').reindex(donors)
                if current.context_source_cell_id.isna().any():raise ValueError('Context panel differs between queries')
                # Recover row indices explicitly, retaining donor order.
                lookup=dict(zip(query_rows.loc[~query_rows.context_kind.eq('unchanged_control'),'context_source_cell_id'].astype(str),query_rows.loc[~query_rows.context_kind.eq('unchanged_control')].index))
                ix=np.array([lookup[d] for d in donors]);verified+=1
                genes=np.flatnonzero(raw[b].toarray().ravel()>0)
                if not len(genes):raise ValueError('No expressed genes')
                marker_mask=np.isin(names[genes],marker_genes)
                metrics=shared_panel_spread(probability[ix],genes,marker_mask,comp[ix])
                rows.append(dict(slice=entry['slice'],query_cell_id=q,query_cell_type=cell_type,**metrics))
            panels.append(dict(slice=entry['slice'],query_cell_type=cell_type,n_panel_contexts=len(donors),n_queries=verified,donor_ids_json=json.dumps(donors)))
        del a,probability,raw
    frame=pd.DataFrame(rows)
    original=pd.read_csv(run/'summaries_distance/query_original_distance_scores.csv')
    frame=frame.merge(original[['slice','query_cell_id','query_distance_um']],on=['slice','query_cell_id'],how='left',validate='one_to_one')
    if frame.query_distance_um.isna().any():raise ValueError('Unmatched actual query distances')
    # Keep the prior original-relative measurements as a sensitivity analysis.
    baseline=pd.read_csv(run/'summaries_logits/query_responses.csv.gz')
    baseline=baseline.loc[baseline.comparison.eq('representative'),['slice','query_cell_id','mean_absolute_delta_logit']]
    frame=frame.merge(baseline,on=['slice','query_cell_id'],how='left',validate='one_to_one')
    rank=pd.read_csv(run/'summaries_ranks/query_rank_changes.csv.gz')
    rank=rank.loc[rank.comparison.eq('representative'),['slice','query_cell_id','mean_rank_distance','mean_top20_overlap']]
    frame=frame.merge(rank,on=['slice','query_cell_id'],how='left',validate='one_to_one')
    metrics=['mean_gene_logit_sd','mean_gene_logit_iqr','malignant_marker_score_sd','mean_malignant_gene_logit_sd',
             'malignant_marker_rank_score_sd','mean_pairwise_rank_distance','malignant_marker_slope_per_10pct',
             'mean_absolute_delta_logit','mean_rank_distance']
    correlations=[]
    for (slice_id,cell_type),part in frame.groupby(['slice','query_cell_type']):
        for metric in metrics:
            good=part[['query_distance_um',metric]].dropna();status='ok'
            if cell_type=='malignant cell':status='zero_distance_by_definition'
            elif len(good)<10:status='insufficient_queries'
            elif good.query_distance_um.nunique()<3 or good[metric].nunique()<3:status='insufficient_variation'
            rho=float(spearmanr(good.query_distance_um,good[metric]).statistic) if status=='ok' else np.nan
            correlations.append(dict(slice=slice_id,query_cell_type=cell_type,metric=metric,n_queries=len(good),spearman_rho=rho,status=status))
    frame.to_csv(out/'query_distance_responsiveness.csv',index=False)
    pd.DataFrame(panels).to_csv(out/'matched_context_panels.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'distance_responsiveness_correlations.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(n_slices=len(design['slices']),n_queries=len(frame),
        marker_genes=marker_genes,
        context_panels='Identical donor IDs within each slice and query type, including original rows only when that donor belongs to the shared panel',
        primary='Baseline-independent SD across contexts and mean pairwise rank distance',
        sensitivity='Existing original-context-relative logit/rank changes',logit_epsilon=1e-6,
        distance='Actual query centroid distance in micrometers, same annotated malignant reference as prior section'),indent=2)+'\n')
    print('Query-distance responsiveness saved:',out,flush=True)

if __name__=='__main__':summarize(Path(sys.argv[1]))
