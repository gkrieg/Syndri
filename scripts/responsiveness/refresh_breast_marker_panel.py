"""Refresh all marker-dependent legacy tables from saved predictions in one pass."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata,spearmanr
from encoded_context_rank import _column,_matrix
from all_types_panel import EXAMPLES
from context_baselines import attach_delta_neighbor_proportion
from physical20_context_counts import ensure_physical20_context_counts

def refresh_marker_panel(run,destination,markers):
    run,destination=Path(run),Path(destination)
    if (destination/'complete.json').exists():
        if json.loads((destination/'complete.json').read_text())['marker_genes']!=list(markers):raise ValueError('Marker cache differs')
        return
    destination.mkdir(exist_ok=True)
    for name in ['summaries_logits','summaries_ranks','summaries_distance','summaries_delta_context']:
        target=destination/name
        if not target.exists():shutil.copytree(run/name,target)
        if (target/'complete.json').exists():(target/'complete.json').unlink()
    baseline=pd.read_csv(run/'summaries_distance/query_original_distance_scores.csv')
    old_context=pd.read_csv(run/'summaries_distance/example_distance_responses.csv.gz')
    distance_maps={}
    for section,rows in baseline.groupby('slice'):
        pairs=pd.concat([rows[['query_cell_id','query_distance_um']].rename(columns={'query_cell_id':'cell_id','query_distance_um':'distance'}),
            old_context.loc[old_context['slice'].eq(section),['context_source_cell_id','context_source_distance_um']].rename(columns={'context_source_cell_id':'cell_id','context_source_distance_um':'distance'})])
        distance_maps[section]=pairs.drop_duplicates('cell_id').set_index('cell_id').distance
    count_paths=ensure_physical20_context_counts(run)
    slope_rows=[];rank_rows=[];examples=[];rank_examples=[];original_updates=[];contexts=[];marker_support=[]
    for entry in json.loads((run/'design.json').read_text())['slices']:
        section=entry['slice'];counts=pd.read_csv(count_paths[section]).set_index('cell_id').malignant_neighbors_k20/20
        distances=distance_maps[section]
        with h5py.File(entry['predictions']) as f:
            obs=pd.DataFrame({k:_column(f['obs/'+k]) for k in ['focal_cell_id','context_source_cell_id','context_kind','query_cell_type','query_family']})
            names=_column(f['var/feature_name']);marker_mask=np.isin(names,markers)
            marker_support.append(dict(slice=section,family='Malignant',n_panel_markers=int(marker_mask.sum()),markers=';'.join(names[marker_mask])))
            raw=_matrix(f['X']);prob=_matrix(f['layers/admix_probs']);prob=prob.toarray() if hasattr(prob,'toarray') else np.asarray(prob)
            p=np.clip(prob[:,marker_mask].astype(float),1e-6,1-1e-6)
            logit=np.log(p)-np.log1p(-p)
            raw_mark=raw[:,marker_mask];raw_mark=raw_mark.toarray() if hasattr(raw_mark,'toarray') else np.asarray(raw_mark)
            expressed_mark=raw_mark>0;num=expressed_mark.sum(axis=1)
            score=np.divide((logit*expressed_mark).sum(axis=1),num,out=np.full(len(obs),np.nan),where=num>0)
            malignant_prop=f['obsm/neighborhood_proportions/malignant cell'][:]
            for q,group in obs.groupby('focal_cell_id',sort=False,observed=True):
                ix=group.index.to_numpy();bix=group.index[group.context_kind.eq('original')].to_numpy()
                if len(bix)!=1:raise ValueError('Expected one original')
                b=int(bix[0]);rep=group.index[group.context_kind.eq('representative')].to_numpy();x=malignant_prop[rep]
                rr=raw[b];expressed=(rr.toarray().ravel() if hasattr(rr,'toarray') else np.asarray(rr).ravel())>0
                ng=int(expressed.sum());nm=int(num[b]);ct=str(obs.loc[b,'query_cell_type'])
                meta=dict(slice=section,query_cell_id=q,query_cell_type=ct,query_family=str(obs.loc[b,'query_family']))
                valid=len(rep)>=10 and len(np.unique(x))>=3 and np.ptp(x)>=.2
                status='no_expressed_markers' if nm==0 else ('ok' if valid else 'insufficient_context_variation')
                beta=np.nan
                if status=='ok':
                    xx=x-x.mean();yy=score[rep]-score[b];beta=float(.1*np.dot(xx,yy-yy.mean())/np.dot(xx,xx))
                slope_rows.append(dict(**meta,neighbor_family='Malignant',delta_logit_per_10pct=beta,status=status,
                    n_contexts=len(rep),n_expressed_markers=nm,context_range=float(np.ptp(x)) if len(x) else 0))
                rank_score=np.full(len(ix),np.nan);rank_beta=np.nan
                if ng>=10:
                    ranks=rankdata(prob[ix][:,expressed],axis=1,method='average')
                    mi=np.isin(names[expressed],markers)
                    if mi.any():rank_score=((ranks[:,mi]-1)/(ng-1)).mean(axis=1)
                    local_base=int(np.flatnonzero(ix==b)[0]);take=group.context_kind.eq('representative').to_numpy()
                    if status=='ok':
                        yy=rank_score[take]-rank_score[local_base];xx=x-x.mean();rank_beta=float(.1*np.dot(xx,yy-yy.mean())/np.dot(xx,xx))
                    rank_rows.append(dict(**meta,n_expressed_genes=ng,neighbor_family='Malignant',status=status,n_expressed_markers=nm,delta_rank_percentile_per_10pct=rank_beta))
                local_base=int(np.flatnonzero(ix==b)[0])
                original_updates.append(dict(slice=section,query_cell_id=q,n_expressed_malignant_markers=nm,
                    malignant_marker_logit=float(score[b]),malignant_marker_rank_percentile=float(rank_score[local_base])))
                if (ct,'Malignant') in EXAMPLES and nm>0:
                    for row,rankval in zip(ix,rank_score):
                        donor=str(obs.loc[row,'context_source_cell_id']);kind=str(obs.loc[row,'context_kind'])
                        donor_distance=distances.get(donor,np.nan);query_distance=distances.get(q,np.nan)
                        if not np.isfinite(donor_distance) or not np.isfinite(query_distance):raise ValueError('Missing cached donor distance')
                        record=dict(**meta,context_kind=kind,context_source_cell_id=donor,query_distance_um=float(query_distance),
                            n_expressed_genes=ng,n_expressed_malignant_markers=nm,
                            context_source_distance_um=float(donor_distance),delta_context_source_distance_um=float(donor_distance-query_distance),
                            delta_marker_logit=float(score[row]-score[b]),delta_marker_rank_percentile=float(rankval-rank_score[local_base]),
                            donor_malignant_fraction_k20=float(counts.loc[donor]),query_malignant_fraction_k20=float(counts.loc[q]),
                            delta_donor_malignant_fraction_k20=float(counts.loc[donor]-counts.loc[q]))
                        contexts.append(record)
                        if kind=='representative':
                            examples.append(dict(**meta,neighbor_family='Malignant',neighbor_proportion=float(malignant_prop[row]),
                                delta_marker_logit=record['delta_marker_logit'],markers=';'.join(names[marker_mask & expressed])))
                            if ng>=10:rank_examples.append(dict(**meta,n_expressed_genes=ng,neighbor_family='Malignant',neighbor_proportion=float(malignant_prop[row]),delta_marker_rank_percentile=record['delta_marker_rank_percentile']))
        print('32-gene refresh:',entry['label'],flush=True)
    logdir=destination/'summaries_logits';rankdir=destination/'summaries_ranks';distdir=destination/'summaries_distance'
    def replace_family(path,new):
        old=pd.read_csv(path);new=pd.DataFrame(new)
        updated=pd.concat([old.loc[~old.neighbor_family.eq('Malignant')],new],ignore_index=True)
        updated.to_csv(path,index=False,compression='infer');return updated
    slopes=replace_family(logdir/'query_family_slopes.csv.gz',slope_rows)
    good=slopes.loc[slopes.status.eq('ok')]
    good.groupby(['slice','query_cell_type','neighbor_family'],observed=True).agg(median_slope=('delta_logit_per_10pct','median'),n_queries=('query_cell_id','nunique')).reset_index().to_csv(logdir/'slice_family_slopes.csv',index=False)
    slopes.groupby(['slice','query_cell_type','neighbor_family','status'],observed=True).size().rename('n_queries').reset_index().to_csv(logdir/'support.csv',index=False)
    pd.DataFrame(examples).to_csv(logdir/'example_context_responses.csv.gz',index=False,compression='gzip')
    support=pd.read_csv(logdir/'marker_support.csv');pd.concat([support.loc[~support.family.eq('Malignant')],pd.DataFrame(marker_support)],ignore_index=True).to_csv(logdir/'marker_support.csv',index=False)
    ranks=replace_family(rankdir/'query_marker_rank_slopes.csv.gz',rank_rows)
    ranks.loc[ranks.status.eq('ok')].groupby(['slice','query_cell_type','neighbor_family']).agg(median_rank_slope=('delta_rank_percentile_per_10pct','median'),n_queries=('query_cell_id','nunique')).reset_index().to_csv(rankdir/'slice_marker_rank_slopes.csv',index=False)
    pd.DataFrame(rank_examples).to_csv(rankdir/'example_marker_rank_changes.csv.gz',index=False,compression='gzip')
    update=pd.DataFrame(original_updates).set_index(['slice','query_cell_id']);baseline=baseline.set_index(['slice','query_cell_id'])
    for column in update:baseline[column]=update[column]
    baseline=baseline.reset_index();baseline.to_csv(distdir/'query_original_distance_scores.csv',index=False)
    context=pd.DataFrame(contexts);context.to_csv(distdir/'example_distance_responses.csv.gz',index=False,compression='gzip')
    context.to_csv(distdir/'example_donor_fraction_responses.csv.gz',index=False,compression='gzip')
    correlations=[]
    for (sid,ct),group in baseline.groupby(['slice','query_cell_type']):
        for metric in ['mean_expressed_gene_logit','malignant_marker_logit','malignant_marker_rank_percentile']:
            good=group[['query_distance_um',metric]].dropna();status='ok'
            if len(good)<10:status='insufficient_queries'
            elif good.query_distance_um.nunique()<3 or good[metric].nunique()<3:status='insufficient_variation'
            rho=float(spearmanr(good.query_distance_um,good[metric]).statistic) if status=='ok' else np.nan
            correlations.append(dict(slice=sid,query_cell_type=ct,metric=metric,n_queries=len(good),spearman_rho=rho,status=status))
    pd.DataFrame(correlations).to_csv(distdir/'original_distance_correlations.csv',index=False)
    for xcol,filename in [('context_source_distance_um','query_context_distance_correlations.csv'),('donor_malignant_fraction_k20','query_context_donor_fraction_correlations.csv')]:
        correlations=[]
        for (sid,q),group in context.loc[context.context_kind.eq('representative')].groupby(['slice','query_cell_id']):
            for metric in ['delta_marker_logit','delta_marker_rank_percentile']:
                good=group[[xcol,metric]].dropna();status='ok' if len(good)>=10 and good[xcol].nunique()>=3 and good[metric].nunique()>1 else 'insufficient_variation'
                rho=float(spearmanr(good[xcol],good[metric]).statistic) if status=='ok' else np.nan
                correlations.append(dict(slice=sid,query_cell_id=q,query_cell_type=group.query_cell_type.iloc[0],metric=metric,spearman_rho=rho,n_contexts=len(good),status=status))
        pd.DataFrame(correlations).to_csv(distdir/filename,index=False)
    controls=context.loc[context.context_kind.eq('unchanged_control')]
    for col in ['delta_marker_logit','delta_marker_rank_percentile','delta_donor_malignant_fraction_k20']:
        if not np.allclose(controls[col].dropna(),0,atol=1e-12):raise ValueError('Unchanged controls differ')
    delta=destination/'summaries_delta_context';baselines=pd.read_csv(delta/'original_family_proportions.csv')
    attach_delta_neighbor_proportion(pd.DataFrame(examples),baselines).to_csv(delta/'logit_examples.csv.gz',index=False,compression='gzip')
    attach_delta_neighbor_proportion(pd.DataFrame(rank_examples),baselines).to_csv(delta/'rank_examples.csv.gz',index=False,compression='gzip')
    for name in ['summaries_logits','summaries_ranks','summaries_distance','summaries_delta_context']:
        metadata=json.loads((run/name/'complete.json').read_text());metadata['marker_genes']=list(markers);metadata['marker_refresh']='32-gene breast cancer panel; other-family/all-gene metrics preserved'
        (destination/name/'complete.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (destination/'complete.json').write_text(json.dumps(dict(marker_genes=list(markers),n_slices=8,n_original_queries=len(baseline)),indent=2)+'\n')
