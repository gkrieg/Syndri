"""Stratified responsiveness queries and fixed per-malignant-count context quotas."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from scipy import sparse



def balanced_allocation(capacities,total,rng):
    """Fill query quota fairly across available strata, without replacement."""
    capacities=np.asarray(capacities,dtype=int);allocation=np.zeros(len(capacities),dtype=int)
    remaining=min(int(total),int(capacities.sum()))
    while remaining:
        available=np.flatnonzero(allocation<capacities)
        for i in rng.permutation(available):
            allocation[i]+=1;remaining-=1
            if not remaining:break
    return allocation


def burden_quantiles(values):
    values=np.asarray(values,dtype=float)
    if len(values)==0:return np.array([],dtype=str),[]
    codes,edges=pd.qcut(values,q=3,labels=False,retbins=True,duplicates='drop')
    n=len(edges)-1
    if n<=0:return np.repeat('constant',len(values)),edges.tolist()
    labels=['low','medium','high'] if n==3 else (['low','high'] if n==2 else ['single_bin'])
    return np.array([labels[int(i)] for i in codes]),edges.tolist()


def prepare(run,index):
    import anndata as ad
    from Heimdall.admixture_sampling import build_neighbor_dict,build_neighbor_celltype_dict
    run=Path(run);design=json.loads((run/'design.json').read_text());entry=design['slices'][index]
    TYPE_FAMILY=design['families']
    out=Path(entry['contexts'])
    if out.exists():raise FileExistsError(out)
    source=ad.read_h5ad(entry['input'])
    labels=source.obs.harmony_cell_type
    xy=source.obs[['x_centroid','y_centroid']].apply(pd.to_numeric,errors='coerce')
    keep=labels.notna().to_numpy() & np.isfinite(xy.to_numpy()).all(axis=1)
    if set(labels.loc[keep].astype(str))-set(TYPE_FAMILY):raise ValueError('Unknown cell types')
    source=ad.AnnData(X=sparse.csr_matrix(source.X,dtype=np.float32)[keep],obs=source.obs.loc[keep].copy(),var=source.var.copy())
    source.uns['cell_type_dict']={t:-(i+1) for i,t in enumerate(sorted(TYPE_FAMILY))}
    source=build_neighbor_dict(source,k_neighbors=5,coord_cols=('x_centroid','y_centroid'),tissue_col=None,
        cell_type_col='harmony_cell_type',min_source_counts=10,use_same_cell_type_neighbors=False)
    source=build_neighbor_celltype_dict(source,cell_type_col='harmony_cell_type')
    ids=source.obs_names.astype(str).to_numpy();types=source.obs.harmony_cell_type.astype(str).to_numpy()
    contexts=source.uns['neighbor_celltype_dict'];neighbors=source.uns['neighbor_dict']
    total=np.asarray(source.X.sum(axis=1)).ravel();ngenes=np.asarray((source.X>0).sum(axis=1)).ravel()
    names=source.var.get('feature_name',pd.Series(source.var_names,index=source.var_names)).fillna('').astype(str).to_numpy()
    names=np.where(names=='',source.var_names.astype(str),names)
    marker_genes=design['marker_genes'];mi=np.isin(names,marker_genes)
    marker_total=np.asarray(source.X[:,mi].sum(axis=1)).ravel()
    burden=np.divide(marker_total,total,out=np.zeros(len(total),dtype=float),where=total>0)
    ncontext=np.array([len(contexts[c]) for c in ids]);mcount=np.array([sum(str(t)=='malignant cell' for t in contexts[c]) for c in ids])
    qlevel=np.select([mcount==0,mcount<=2],['0','1-2'],default='3-5')
    type_order=list(TYPE_FAMILY);type_lookup={t:i for i,t in enumerate(type_order)}
    composition=np.zeros((len(ids),len(type_order)),dtype=np.float32)
    for i,cid in enumerate(ids):
        for t in contexts[cid]:composition[i,type_lookup[str(t)]]+=1
    composition/=5
    rng=np.random.default_rng(design['seed']+index)
    rows=[];row_positions=[];context_positions=[];row_neighbor_types=[];query_records=[];context_records=[];stratum_records=[];coverage=[];cuts=[]
    donor_panels={};all_source_positions=set()
    for ct in design['query_types']:
        candidates=np.flatnonzero((types==ct)&(total>=10)&(ngenes>=5)&(ncontext==5))
        burden_labels,edges=burden_quantiles(burden[candidates])
        cuts.append(dict(slice=entry['slice'],cell_type=ct,population='eligible cells of this type in this slice',
            n_eligible=len(candidates),burden_quantile_edges_json=json.dumps(edges),
            n_effective_burden_strata=len(set(burden_labels)),marker_genes_json=json.dumps(marker_genes)))
        if not len(candidates):
            coverage.append(dict(slice=entry['slice'],cell_type=ct,n_eligible=0,n_queries=0,n_contexts=0,status='no_eligible_cells'));continue
        joint=np.array([f'{qlevel[c]}|{b}' for c,b in zip(candidates,burden_labels)])
        strata=sorted(set(joint));capacities=[int((joint==s).sum()) for s in strata]
        quotas=balanced_allocation(capacities,design['n_queries'],rng)
        queries=[];qmeta={}
        for stratum,capacity,quota in zip(strata,capacities,quotas):
            pool=candidates[joint==stratum];chosen=np.sort(rng.choice(pool,int(quota),replace=False))
            stratum_records.append(dict(slice=entry['slice'],cell_type=ct,query_joint_stratum=stratum,n_available=capacity,n_selected=int(quota)))
            for q in chosen:
                qmeta[int(q)]=dict(query_malignant_token_count=int(mcount[q]),query_neighbor_level=str(qlevel[q]),
                    query_marker_transcripts=float(marker_total[q]),query_total_transcripts=float(total[q]),
                    query_marker_burden=float(burden[q]),query_burden_quantile=str(burden_labels[np.flatnonzero(candidates==q)[0]]),
                    query_joint_stratum=stratum,query_stratum_available=int(capacity),query_stratum_selected=int(quota))
            queries.extend(chosen.tolist())
        donors=[];donor_counts={};available_counts={}
        for count in range(6):
            pool=candidates[mcount[candidates]==count];take=min(5,len(pool));chosen=np.sort(rng.choice(pool,take,replace=False))
            donors.extend(chosen.tolist());donor_counts[count]=take;available_counts[count]=len(pool)
            if take>5 or take>len(pool):raise ValueError('Invalid context quota')
            for c in chosen:
                context_records.append(dict(slice=entry['slice'],cell_type=ct,context_source_cell_id=ids[c],
                    context_malignant_token_count=count,context_sampling_stratum=str(count),
                    n_available_in_stratum=len(pool),n_requested_in_stratum=5,n_selected_in_stratum=take,
                    donor_marker_burden=float(burden[c]),token_sequence=' | '.join(map(str,contexts[ids[c]]))))
            # Include empty/undersupplied counts explicitly in the coverage table.
            stratum_records.append(dict(slice=entry['slice'],cell_type=ct,context_malignant_token_count=count,
                n_contexts_available=len(pool),n_contexts_requested=5,n_contexts_selected=take))
        if design.get('replay_sampling', True):
            sampling=Path(design['sampling_dir'])
            archived_queries=pd.read_csv(sampling/f'{entry["slice"]}_queries.csv',dtype={'query_neighbor_level':str})
            archived_donors=pd.read_csv(sampling/f'{entry["slice"]}_context_donors.csv')
            aq=archived_queries.loc[archived_queries.query_cell_type.eq(ct)]
            ac=archived_donors.loc[archived_donors.cell_type.eq(ct)]
            positions={str(cid):i for i,cid in enumerate(ids)}
            queries=[positions[str(cid)] for cid in aq.query_cell_id]
            donors=[positions[str(cid)] for cid in ac.context_source_cell_id]
            if not set(queries+donors).issubset(set(candidates)):
                raise ValueError('Archived query/donor no longer eligible')
            metadata=[k for k in aq if k not in ['slice','query_cell_id','query_cell_type']]
            qmeta={positions[str(row['query_cell_id'])]:{k:row[k] for k in metadata} for row in aq.to_dict('records')}
            for row in ac.to_dict('records'):
                c=positions[str(row['context_source_cell_id'])]
                if int(mcount[c])!=int(row['context_malignant_token_count']):
                    raise ValueError('Archived neighbor composition changed')
            # Replace audit records for this type with the archived donor records.
            context_records=[r for r in context_records if r['cell_type']!=ct]+ac.to_dict('records')
            donor_counts={count:sum(int(mcount[c])==count for c in donors) for count in range(6)}
        donors=sorted(donors);queries=sorted(queries)
        if len(donors)!=len(set(donors)) or len(queries)!=len(set(queries)):raise ValueError('Duplicate sampled cells')
        donor_panels[ct]=[str(ids[c]) for c in donors]
        for q in queries:
            query_records.append(dict(slice=entry['slice'],query_cell_id=ids[q],query_cell_type=ct,**qmeta[q]))
            for c,kind in [(q,'original'),(q,'unchanged_control')]+[(c,'representative') for c in donors if c!=q]:
                all_source_positions.add(c)
                rows.append(dict(focal_cell_id=ids[q],context_source_cell_id=ids[c],query_cell_type=ct,
                    query_family=TYPE_FAMILY[ct],context_kind=kind,is_original=kind=='original',
                    is_representative_context=kind=='representative',sampling_repeat=0,context_source_position=int(c),
                    **qmeta[q],context_malignant_token_count=int(mcount[c]),context_malignant_proportion=float(mcount[c]/5),
                    context_sampling_stratum=str(int(mcount[c])),context_selected_for_shared_panel=bool(c in donors),
                    context_stratum_available=int(available_counts[int(mcount[c])]),context_stratum_requested=5,
                    context_stratum_selected=int(donor_counts[int(mcount[c])]),context_donor_marker_burden=float(burden[c])))
                row_positions.append(q);context_positions.append(c);row_neighbor_types.append(list(contexts[ids[c]]))
        coverage.append(dict(slice=entry['slice'],cell_type=ct,n_eligible=len(candidates),n_queries=len(queries),n_contexts=len(donors),
            status='ok',n_query_strata=len(strata),n_unique_encoded_sequences=len({tuple(contexts[ids[c]]) for c in donors})))
    if not rows:raise ValueError('No eligible stratified query cells')
    obs=source.obs.iloc[np.asarray(row_positions)].copy();obs.index=[f'stratified_{index}_{i:07d}' for i in range(len(rows))]
    for key in rows[0]:obs[key]=[row[key] for row in rows]
    obs['split']='val';obs['panel_slice']=entry['slice']
    result=ad.AnnData(X=source.X[np.asarray(row_positions)].copy(),obs=obs,var=source.var.copy())
    result.uns['cell_type_dict']=source.uns['cell_type_dict']
    result.uns['neighbor_celltype_dict']={rid:ctypes for rid,ctypes in zip(obs.index,row_neighbor_types)}
    result.uns['matched_context_donors_by_query_type']=donor_panels
    result.uns['context_donor_neighbor_ids']={str(ids[c]):list(map(str,neighbors[ids[c]])) for c in sorted(all_source_positions)}
    result.uns['responsiveness_panel_design_json']=json.dumps(design)
    result.obsm['neighborhood_proportions']=pd.DataFrame(composition[np.asarray(context_positions)],index=obs.index,columns=type_order)
    out.parent.mkdir(exist_ok=True)
    result.write_h5ad(out,compression='gzip')
    for suffix,records in [('queries',query_records),('context_donors',context_records),('strata',stratum_records if not design.get('replay_sampling',True) else []),('coverage',coverage),('burden_cutpoints',cuts)]:
        pd.DataFrame(records).to_csv(run/'contexts'/f'{entry["slice"]}_{suffix}.csv',index=False)
    out.with_suffix('.complete.json').write_text(json.dumps(dict(n_rows=result.n_obs,n_queries=len(query_records),n_context_donors=len(context_records),
        marker_genes=marker_genes,burden_quantiles='per slice per cell type',context_quotas='up to five distinct donors per exact malignant count; no compensation'),indent=2)+'\n')
    print('Prepared stratified sampling:',entry['label'],result.shape,len(query_records),'queries',flush=True)

