"""Within-query ranking responsiveness from existing, untransformed probabilities."""
from pathlib import Path
import argparse,json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from all_types_panel import FAMILIES,MARKERS,EXAMPLES,dense


def fractional_top_membership(values,k):
    """Unit weights above cutoff; distribute remaining top-k mass across cutoff ties."""
    cutoff=np.partition(values,len(values)-k)[len(values)-k]
    weights=(values>cutoff).astype(float)
    tied=values==cutoff
    weights[tied]=(k-weights.sum())/tied.sum()
    return weights


def summarize_ranks(run):
    run=Path(run);out=run/'summaries_ranks';out.mkdir(exist_ok=False)
    design=json.loads((run/'design.json').read_text())
    contexts=[];queries=[];marker_slopes=[];examples=[];malignant_slopes=[]
    for entry in design['slices']:
        path=Path(entry['predictions'])
        if not path.with_suffix('.complete.json').exists():raise FileNotFoundError(path)
        a=ad.read_h5ad(path);obs=a.obs.reset_index(drop=True)
        probability=dense(a.layers['admix_probs'])
        if not np.isfinite(probability).all():raise ValueError('Nonfinite prediction')
        raw=sparse.csr_matrix(a.X)
        names=a.var.get('feature_name',pd.Series(a.var_names,index=a.var_names)).fillna('').astype(str).to_numpy()
        names=np.where(names=='',a.var_names.astype(str),names)
        comp=a.obsm['neighborhood_proportions']
        family_comp={f:comp.reindex(columns=types,fill_value=0).sum(axis=1).to_numpy() for f,types in FAMILIES.items()}
        for q,group in obs.groupby('focal_cell_id',sort=False,observed=True):
            ix=group.index.to_numpy();original=group.index[group.context_kind.eq('original')].to_numpy()
            if len(original)!=1:raise ValueError('Expected one baseline')
            b=original[0];genes=np.flatnonzero(raw[b].toarray().ravel()>0)
            meta=dict(slice=entry['slice'],query_cell_id=q,query_cell_type=obs.loc[b,'query_cell_type'],query_family=obs.loc[b,'query_family'],n_expressed_genes=len(genes))
            if len(genes)<10:
                queries.append({**meta,'status':'fewer_than_10_expressed_genes'});continue
            p=probability[ix][:,genes]
            ranks=rankdata(p,axis=1,method='average')
            local_base=int(np.flatnonzero(ix==b)[0]);r0=ranks[local_base]
            centered=ranks-ranks.mean(axis=1,keepdims=True);base_centered=centered[local_base]
            denominator=np.linalg.norm(centered,axis=1)*np.linalg.norm(base_centered)
            rho=np.divide(centered@base_centered,denominator,out=np.full(len(ix),np.nan),where=denominator>0)
            rho=np.clip(rho,-1,1)
            delta_percentile=(ranks-r0)/(len(genes)-1)
            overlaps={}
            for fraction in [.1,.2]:
                k=min(len(genes),max(3,int(np.ceil(fraction*len(genes)))))
                weights=np.stack([fractional_top_membership(v,k) for v in p])
                overlaps[fraction]=np.minimum(weights,weights[local_base]).sum(axis=1)/k
            for j,row in enumerate(ix):
                if obs.loc[row,'context_kind']=='original':continue
                contexts.append({**meta,'context_kind':obs.loc[row,'context_kind'],
                    'malignant_neighbor_proportion':family_comp['Malignant'][row],
                    'spearman_rho':rho[j],'rank_distance':1-rho[j],
                    'top10_overlap':overlaps[.1][j],'top20_overlap':overlaps[.2][j]})
            rep=group.context_kind.eq('representative').to_numpy()
            for kind in ['representative','unchanged_control']:
                take=group.context_kind.eq(kind).to_numpy()
                queries.append({**meta,'comparison':kind,'status':'ok' if np.isfinite(rho[take]).all() else 'constant_rank_vector',
                    'n_contexts':int(take.sum()),'mean_rank_distance':float(np.nanmean(1-rho[take])) if np.isfinite(rho[take]).any() else np.nan,
                    'mean_top10_overlap':float(overlaps[.1][take].mean()),'mean_top20_overlap':float(overlaps[.2][take].mean())})
            malignant_x=family_comp['Malignant'][ix][rep]
            if rep.sum()>=10 and len(np.unique(malignant_x))>=3 and np.ptp(malignant_x)>=.2 and np.isfinite(rho[rep]).all():
                xc=malignant_x-malignant_x.mean();y=1-rho[rep]
                malignant_slopes.append({**meta,'rank_distance_per_10pct_malignant':float(.1*np.dot(xc,y-y.mean())/np.dot(xc,xc))})
            for family,markers in MARKERS.items():
                mi=np.flatnonzero(np.isin(names[genes],markers));x=family_comp[family][ix][rep]
                status='ok'
                if rep.sum()<10 or len(np.unique(x))<3 or np.ptp(x)<.2:status='insufficient_context_variation'
                if len(mi)==0:status='no_expressed_markers'
                beta=np.nan
                if status=='ok':
                    y=delta_percentile[rep][:,mi].mean(axis=1);xc=x-x.mean()
                    beta=float(.1*np.dot(xc,y-y.mean())/np.dot(xc,xc))
                marker_slopes.append({**meta,'neighbor_family':family,'status':status,'n_expressed_markers':len(mi),'delta_rank_percentile_per_10pct':beta})
                if (meta['query_cell_type'],family) in EXAMPLES and len(mi):
                    y=delta_percentile[rep][:,mi].mean(axis=1)
                    examples.extend({**meta,'neighbor_family':family,'neighbor_proportion':float(xx),'delta_marker_rank_percentile':float(yy)} for xx,yy in zip(x,y))
        del a,raw,probability
    context=pd.DataFrame(contexts);query=pd.DataFrame(queries);markers=pd.DataFrame(marker_slopes)
    context.to_csv(out/'context_rank_changes.csv.gz',index=False,compression='gzip')
    query.to_csv(out/'query_rank_changes.csv.gz',index=False,compression='gzip')
    markers.to_csv(out/'query_marker_rank_slopes.csv.gz',index=False,compression='gzip')
    pd.DataFrame(examples).to_csv(out/'example_marker_rank_changes.csv.gz',index=False,compression='gzip')
    malign=pd.DataFrame(malignant_slopes);malign.to_csv(out/'query_malignant_rank_distance_slopes.csv',index=False)
    query[query.status.eq('ok')].groupby(['slice','query_cell_type','comparison']).agg(
        median_rank_distance=('mean_rank_distance','median'),median_top10_overlap=('mean_top10_overlap','median'),
        median_top20_overlap=('mean_top20_overlap','median'),n_queries=('query_cell_id','nunique')).reset_index().to_csv(out/'slice_rank_changes.csv',index=False)
    markers[markers.status.eq('ok')].groupby(['slice','query_cell_type','neighbor_family']).agg(
        median_rank_slope=('delta_rank_percentile_per_10pct','median'),n_queries=('query_cell_id','nunique')).reset_index().to_csv(out/'slice_marker_rank_slopes.csv',index=False)
    malign.groupby(['slice','query_cell_type']).agg(median_slope=('rank_distance_per_10pct_malignant','median'),n_queries=('query_cell_id','nunique')).reset_index().to_csv(out/'slice_malignant_rank_distance_slopes.csv',index=False)
    controls=query[query.comparison.eq('unchanged_control')&query.status.eq('ok')]
    if not np.allclose(controls.mean_rank_distance,0,atol=1e-12) or not np.allclose(controls.mean_top20_overlap,1):raise ValueError('Unchanged controls changed ranks')
    (out/'complete.json').write_text(json.dumps(dict(n_slices=len(design['slices']),n_ranked_queries=int(query.loc[query.comparison.eq('representative'),'query_cell_id'].count()),rank_source='raw probabilities, without clipping or logit',minimum_expressed_genes=10,top_k_minimum=3,ties='average ranks; fractional top-k memberships; overlap=sum(min(weights))/k'),indent=2)+'\n')
    print('Rank summaries saved:',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);summarize_ranks(parser.parse_args().run)
