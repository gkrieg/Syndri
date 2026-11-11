"""Within-query response versus malignant occupancy/order in the encoded context."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.stats import rankdata
from gene_context_slopes import T_CELL_TYPES

def _column(node):
    if isinstance(node, h5py.Group):
        codes = node['codes'][:]
        if (codes < 0).any():
            raise ValueError('Missing context metadata')
        return node['categories'].asstr()[:][codes]
    return node.asstr()[:]

def _matrix(node):
    if isinstance(node, h5py.Group):
        return sparse.csr_matrix((node['data'][:],node['indices'][:],node['indptr'][:]),shape=tuple(node.attrs['shape']))
    return node[:]

def _row_correlation(x, y):
    x = np.asarray(x,float); y = np.asarray(y,float)
    xc = x - x.mean()
    yc = y - y.mean(axis=1,keepdims=True)
    den = np.sqrt(np.dot(xc,xc)*np.sum(yc*yc,axis=1))
    return np.divide(yc@xc,den,out=np.full(len(y),np.nan),where=den>1e-12)

def _spearman(x, y):
    return _row_correlation(rankdata(x),rankdata(y,axis=1))

def analyze_encoded_context_rank(run, output_dir, marker_genes, min_queries=10):
    run, out = Path(run), Path(output_dir)
    out.mkdir(exist_ok=True)
    config=dict(marker_genes=list(marker_genes),response_scale='logit',logit_clip=1e-6,
        response='mean logit of selected genes expressed in the fixed query',min_queries_per_section=min_queries,
        pool_t_cells=True,context='same matched donor panel per section and original query type',
        rank='1-based position in saved ordered neighbor_celltype_dict',
        min_exact_one_contexts=4,min_rank_levels=3,min_mixed_contexts=8,
        partial_correlation='Pearson correlation of within-query rank residuals after categorical malignant-count demeaning',
        slot_coefficients='joint OLS of marker score on intercept and five malignant occupancy indicators')
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text()) != config:
            raise ValueError('Encoded-context cache definition differs')
        return
    query_frames=[]; curve_frames=[]; audits=[]; pattern_frames=[]; slot_frames=[]
    design=json.loads((run/'design.json').read_text())
    for entry in design['slices']:
        contexts_out=[]
        with h5py.File(entry['predictions']) as f:
            obs=pd.DataFrame({name:_column(f['obs/'+name]) for name in
                ['focal_cell_id','context_source_cell_id','context_kind','query_cell_type']})
            row_ids=_column(f['obs'][f['obs'].attrs['_index']])
            names=_column(f['var/feature_name'])
            selected=np.flatnonzero(np.isin(names,marker_genes))
            raw=_matrix(f['X']); p=_matrix(f['layers/admix_probs'])
            raw=raw[:,selected]; p=p[:,selected]
            expressed=(raw.toarray() if sparse.issparse(raw) else np.asarray(raw))>0
            pp=(p.toarray() if sparse.issparse(p) else np.asarray(p)).astype(float)
            if not np.isfinite(pp).all() or ((pp<0)|(pp>1)).any():
                raise ValueError('Invalid probabilities')
            pp=np.clip(pp,1e-6,1-1e-6); logits=np.log(pp)-np.log1p(-pp)
            nmarkers=expressed.sum(axis=1)
            score=np.divide((logits*expressed).sum(axis=1),nmarkers,
                out=np.full(len(obs),np.nan),where=nmarkers>0)
            for cell_type,part in obs.groupby('query_cell_type',observed=True):
                donors=sorted(part.loc[part.context_kind.eq('representative'),'context_source_cell_id'].unique())
                current=part.loc[~part.context_kind.eq('unchanged_control') & part.context_source_cell_id.isin(donors)].sort_values(['focal_cell_id','context_source_cell_id'])
                if current.duplicated(['focal_cell_id','context_source_cell_id']).any():
                    raise ValueError('Duplicate query/donor')
                sizes=current.groupby('focal_cell_id').size()
                if not sizes.eq(len(donors)).all():
                    raise ValueError('Queries do not share contexts')
                ids=current.focal_cell_id.drop_duplicates().to_numpy()
                nq,nc=len(ids),len(donors)
                ix=current.index.to_numpy().reshape(nq,nc)
                donor_matrix=current.context_source_cell_id.to_numpy().reshape(nq,nc)
                if not (donor_matrix==np.asarray(donors)[None,:]).all():
                    raise ValueError('Context donor order mismatch')
                patterns=[list(f['uns/neighbor_celltype_dict'][str(row_ids[i])].asstr()[:]) for i in ix[0]]
                if any(len(pat)!=5 for pat in patterns):
                    raise ValueError('Expected five encoded neighbor tokens')
                z=np.array([[t=='malignant cell' for t in pat] for pat in patterns],dtype=int)
                # Independently verify the saved composition agrees with encoded occupancy.
                saved=f['obsm/neighborhood_proportions/malignant cell'][:][ix[0]]
                count=z.sum(axis=1); proportion=count/5
                if not np.allclose(saved,proportion):
                    raise ValueError('Token composition differs from saved model context')
                group='T cells (pooled)' if cell_type in T_CELL_TYPES else str(cell_type)
                for j,donor in enumerate(donors):
                    pattern_frames.append(dict(slice=entry['slice'],query_cell_type=str(cell_type),query_group=group,
                        context_source_cell_id=donor,malignant_count=int(count[j]),malignant_proportion=float(proportion[j]),
                        token_sequence=' | '.join(patterns[j]),malignant_pattern=''.join(map(str,z[j]))))
                original=part.loc[part.context_kind.eq('original')].copy();original['position']=original.index
                if original.focal_cell_id.duplicated().any():raise ValueError('Duplicate original context')
                baseline_ix=original.set_index('focal_cell_id').reindex(ids).position.to_numpy(dtype=int)
                scores=score[ix]; baseline=score[baseline_ix]
                valid=np.isfinite(baseline)&np.isfinite(scores).all(axis=1)
                delta=scores-baseline[:,None]
                centered=scores-scores.mean(axis=1,keepdims=True)
                rho_proportion=np.full(nq,np.nan)
                if nc>=10 and np.unique(count).size>=3:
                    rho_proportion[valid]=_spearman(count,scores[valid])
                one=count==1
                sole_rank=np.argmax(z,axis=1)+1
                rho_one=np.full(nq,np.nan); slope_one=np.full(nq,np.nan)
                one_centered=np.full((nq,nc),np.nan)
                if one.any():
                    one_centered[:,one]=scores[:,one]-scores[:,one].mean(axis=1,keepdims=True)
                if one.sum()>=4 and np.unique(sole_rank[one]).size>=3:
                    rho_one[valid]=_spearman(sole_rank[one],scores[valid][:,one])
                    x=sole_rank[one].astype(float);xc=x-x.mean()
                    slope_one[valid]=(scores[valid][:,one]@xc)/np.dot(xc,xc)
                mixed=(count>0)&(count<5)
                rank_mean=np.divide((z*np.arange(1,6)).sum(axis=1),count,
                    out=np.full(nc,np.nan),where=count>0)
                rho_partial=np.full(nq,np.nan)
                if mixed.sum()>=8 and np.unique(rank_mean[mixed]).size>=3:
                    xr=rankdata(rank_mean[mixed]).astype(float)
                    yr=rankdata(scores[valid][:,mixed],axis=1).astype(float)
                    for k in np.unique(count[mixed]):
                        strata=count[mixed]==k
                        xr[strata]-=xr[strata].mean()
                        yr[:,strata]-=yr[:,strata].mean(axis=1,keepdims=True)
                    rho_partial[valid]=_row_correlation(xr,yr)
                design_matrix=np.column_stack([np.ones(nc),z])
                full_rank=np.linalg.matrix_rank(design_matrix)==6
                coefficients=np.full((nq,5),np.nan)
                if full_rank and valid.any():
                    coefficients[valid]=np.linalg.lstsq(design_matrix,scores[valid].T,rcond=None)[0][1:].T
                relative=coefficients-coefficients.mean(axis=1,keepdims=True)
                qframe=pd.DataFrame(dict(slice=entry['slice'],query_cell_id=ids,query_cell_type=str(cell_type),query_group=group,
                    n_expressed_markers=nmarkers[baseline_ix],n_contexts=nc,n_exact_one_contexts=int(one.sum()),
                    rho_encoded_malignant_proportion=rho_proportion,rho_exact_one_rank=rho_one,
                    exact_one_rank_slope=slope_one,rho_rank_adjusted_for_count=rho_partial,
                    early_minus_late_slot_coefficient=coefficients[:,0]-coefficients[:,4]))
                query_frames.append(qframe)
                for pos in range(5):
                    slot_frames.append(pd.DataFrame(dict(slice=entry['slice'],query_cell_id=ids,query_group=group,
                        slot=pos+1,coefficient=coefficients[:,pos],relative_coefficient=relative[:,pos])))
                audits.append(dict(slice=entry['slice'],query_cell_type=str(cell_type),n_queries=nq,
                    n_supported_queries=int(valid.sum()),n_contexts=nc,n_exact_one_contexts=int(one.sum()),
                    n_exact_one_rank_levels=len(np.unique(sole_rank[one])),n_mixed_contexts=int(mixed.sum()),
                    joint_slot_design_rank=int(np.linalg.matrix_rank(design_matrix))))
                for j in range(nc):
                    contexts_out.append(pd.DataFrame(dict(slice=entry['slice'],query_cell_id=ids,query_group=group,
                        context_source_cell_id=donors[j],malignant_count=int(count[j]),malignant_proportion=float(proportion[j]),
                        sole_malignant_rank=int(sole_rank[j]) if one[j] else np.nan,
                        mean_malignant_rank=rank_mean[j],marker_logit=scores[:,j],delta_marker_logit=delta[:,j],
                        centered_marker_logit=centered[:,j],exact_one_centered_logit=one_centered[:,j])))
        long=pd.concat(contexts_out,ignore_index=True)
        long.to_csv(out/(entry['slice']+'_context_responses.csv.gz'),index=False,compression='gzip')
        # Equalize contexts within each query before aggregating query medians in a section.
        for curve,xcol,ycol in [('composition','malignant_proportion','centered_marker_logit'),
                                ('exact_one_rank','sole_malignant_rank','exact_one_centered_logit')]:
            by_query=long.dropna(subset=[xcol,ycol]).groupby(['query_group','query_cell_id',xcol],observed=True)[ycol].mean().reset_index()
            curves=by_query.groupby(['query_group',xcol],observed=True).agg(
                median_response=(ycol,'median'),n_queries=('query_cell_id','nunique')).reset_index().rename(columns={xcol:'x'})
            curves['median_response']=curves.median_response.where(curves.n_queries>=min_queries)
            curve_frames.append(curves.assign(slice=entry['slice'],curve=curve))
        print('Encoded order analysis:',entry['label'],len(long),'query/context pairs',flush=True)
    queries=pd.concat(query_frames,ignore_index=True)
    metrics=['rho_encoded_malignant_proportion','rho_exact_one_rank','exact_one_rank_slope',
             'rho_rank_adjusted_for_count','early_minus_late_slot_coefficient']
    section=queries.melt(id_vars=['slice','query_group','query_cell_id'],value_vars=metrics,
        var_name='metric',value_name='value').groupby(['slice','query_group','metric'],observed=True).agg(
        median_value=('value','median'),n_queries=('value','count')).reset_index()
    section['median_value']=section.median_value.where(section.n_queries>=min_queries)
    slots=pd.concat(slot_frames,ignore_index=True)
    section_slots=slots.groupby(['slice','query_group','slot'],observed=True).agg(
        median_coefficient=('coefficient','median'),median_relative_coefficient=('relative_coefficient','median'),
        n_queries=('coefficient','count')).reset_index()
    for col in ['median_coefficient','median_relative_coefficient']:
        section_slots[col]=section_slots[col].where(section_slots.n_queries>=min_queries)
    queries.to_csv(out/'query_statistics.csv',index=False)
    section.to_csv(out/'section_statistics.csv',index=False)
    section_slots.to_csv(out/'section_slot_coefficients.csv',index=False)
    pd.concat(curve_frames,ignore_index=True).to_csv(out/'section_curves.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'support_audit.csv',index=False)
    pd.DataFrame(pattern_frames).to_csv(out/'encoded_context_patterns.csv',index=False)
    (out/'complete.json').write_text(json.dumps(config,indent=2)+'\n')
