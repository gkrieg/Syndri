"""Matched native-marker AUROC and nine-stratum per-gene win responses from saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.stats import spearmanr


def column(node):
    if isinstance(node,h5py.Group):
        codes=node['codes'][:]
        if (codes<0).any():raise ValueError('Missing categorical metadata')
        return node['categories'].asstr()[:][codes]
    return node.asstr()[:]


def matrix(node):
    if isinstance(node,h5py.Group):
        return sparse.csr_matrix((node['data'][:],node['indices'][:],node['indptr'][:]),shape=tuple(node.attrs['shape']))
    return node[:]


def compute_native_marker_analysis(run,min_queries=10,min_delta_levels=3):
    run=Path(run);design=json.loads((run/'design.json').read_text())
    markers=design['marker_genes'];native_sets=design['query_native_marker_sets']
    out=run/'marker_auroc_context_responses';win_out=run/'gene_native_win_stratified_context_correlations'
    out.mkdir(exist_ok=True);win_out.mkdir(exist_ok=True)
    config=dict(positive_genes=markers,negative_genes_by_exact_query_type=native_sets,
                eligible='raw query counts >0, fixed across contexts; at least one expressed positive and native negative',
                auroc='mean(score_positive > score_native) + 0.5*mean(equal); original-relative change',
                gene_win='strict mean(score_gene > score_native); ties count zero; original-relative change',
                strata='burden low/medium/high x original malignant neighbors 0/1-2/3-5',
                correlation='within section/family/burden/original stratum Spearman(delta malignant count, delta gene win fraction)',
                summary='equal-unit median over supported section x burden x original-context correlations',
                min_queries=min_queries,min_delta_levels=min_delta_levels,
                reference_workbook_sha256=design['native_marker_workbook_sha256'])
    if (run/'analysis_complete.json').exists():
        if json.loads((run/'analysis_config.json').read_text())!=config:raise ValueError('Native-marker analysis definition differs')
        return out,win_out
    (run/'analysis_config.json').write_text(json.dumps(config,indent=2)+'\n')
    mapping=pd.read_csv(run/'burden_stratified_context_auroc_response/cell_family_mapping.csv').set_index('query_cell_type').query_family.to_dict()
    audits=[];section_rows=[];gene_coverage=[];source_frames=[]
    for entry in design['slices']:
        with h5py.File(entry['predictions']) as f:
            string_keys=['focal_cell_id','context_source_cell_id','context_kind','query_cell_type','query_burden_quantile','query_neighbor_level']
            obs=pd.DataFrame({k:column(f['obs/'+k]) for k in string_keys})
            for k in ['query_malignant_token_count','context_malignant_token_count','query_marker_burden','context_selected_for_shared_panel']:
                obs[k]=f['obs/'+k][:]
            obs['query_family']=obs.query_cell_type.map(mapping)
            names=column(f['var/feature_name'])
            if len(names)!=len(set(names)):raise ValueError('Gene symbols not unique')
            lookup={g:i for i,g in enumerate(names)}
            mi=np.array([lookup.get(g,-1) for g in markers]);raw=matrix(f['X']);p=matrix(f['layers/admix_probs'])
            if sparse.issparse(p):p=p.toarray()
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
            for typ,genes in native_sets.items():
                gene_coverage.append(dict(slice=entry['slice'],query_cell_type=typ,n_author_genes=len(genes),n_measured_genes=sum(g in lookup for g in genes),missing_genes='; '.join(g for g in genes if g not in lookup)))
            auc=np.full(len(obs),np.nan);original_auc=np.full(len(obs),np.nan)
            npositive=np.zeros(len(obs),int);nnegative=np.zeros(len(obs),int)
            change=np.full((len(obs),len(markers)),np.nan,dtype=np.float32)
            delta=np.zeros(len(obs),dtype=np.int8);selected=np.zeros(len(obs),bool)
            query_counts=f['obs/query_malignant_token_count'][:];context_counts=f['obs/context_malignant_token_count'][:]
            for query,part in obs.groupby('focal_cell_id',sort=False,observed=True):
                ix=part.index.to_numpy();bases=part.index[part.context_kind.eq('original')].to_numpy()
                if len(bases)!=1:raise ValueError('One original per query required')
                b=int(bases[0]);typ=str(obs.loc[b,'query_cell_type']);rr=raw[b]
                counts=rr.toarray().ravel() if sparse.issparse(rr) else np.asarray(rr).ravel()
                posj=np.array([j for j,g in enumerate(mi) if g>=0 and counts[g]>0],int)
                native=np.array([lookup[g] for g in native_sets.get(typ,[]) if g in lookup and counts[lookup[g]]>0],int)
                if set(mi[posj]) & set(native):raise ValueError('Positive and native sets overlap')
                npositive[ix]=len(posj);nnegative[ix]=len(native)
                status='no_native_annotation' if typ not in native_sets else 'no_expressed_malignant_marker' if not len(posj) else 'no_expressed_native_marker' if not len(native) else 'valid'
                audits.append(dict(slice=entry['slice'],slice_label=entry['label'],query_cell_id=query,query_cell_type=typ,query_family=mapping[typ],
                    query_burden_quantile=str(obs.loc[b,'query_burden_quantile']),query_neighbor_level=str(obs.loc[b,'query_neighbor_level']),
                    n_expressed_malignant_markers=len(posj),n_expressed_native_markers=len(native),status=status))
                if status!='valid':continue
                targets=p[ix][:,mi[posj]];native_scores=p[ix][:,native]
                wins=(targets[:,:,None]>native_scores[:,None,:]).mean(axis=2)
                ties=(targets[:,:,None]==native_scores[:,None,:]).mean(axis=2)
                values=(wins+.5*ties).mean(axis=1);base=int(np.flatnonzero(ix==b)[0])
                auc[ix]=values;original_auc[ix]=values[base]
                change[np.ix_(ix,posj)]=wins-wins[base]
                assert np.isclose((auc-original_auc)[b],0)
                delta[ix]=context_counts[ix]-query_counts[b]
                donors=f['uns/matched_context_donors_by_query_type'][typ].asstr()[:]
                selected[ix]=part.context_source_cell_id.isin(donors).to_numpy()&part.context_kind.ne('unchanged_control').to_numpy()
            frame=obs.copy().rename(columns={'focal_cell_id':'query_cell_id'})
            frame['slice']=entry['slice'];frame['marker_auroc']=auc;frame['original_marker_auroc']=original_auc;frame['delta_marker_auroc']=auc-original_auc
            frame['n_eligible_marker_genes']=npositive;frame['n_eligible_native_marker_genes']=nnegative
            # Compatibility field identifies the negative count; actual negatives are native markers only.
            frame['n_eligible_nonmarker_genes']=nnegative
            source_frames.append(frame)
            for family,fam in obs.groupby('query_family',observed=True):
                for burden in ['low','medium','high']:
                    for original in ['0','1-2','3-5']:
                        part=fam.loc[fam.query_burden_quantile.eq(burden)&fam.query_neighbor_level.eq(original)]
                        ix=part.index.to_numpy();ix=ix[selected[ix]]
                        for j,gene in enumerate(markers):
                            use=ix[np.isfinite(change[ix,j])];nq=obs.loc[use,'focal_cell_id'].nunique();levels=np.unique(delta[use]).size
                            status='insufficient_queries' if nq<min_queries else 'insufficient_delta_levels' if levels<min_delta_levels else 'constant_response' if np.unique(change[use,j]).size<2 else 'valid'
                            rho=float(spearmanr(delta[use],change[use,j]).statistic) if status=='valid' else np.nan
                            section_rows.append(dict(slice=entry['slice'],slice_label=entry['label'],group_kind='family',query_group=family,gene=gene,burden_stratum=burden,
                                original_context_stratum=original,rho=rho,n_queries=nq,n_context_pairs=len(use),n_delta_levels=levels,status=status))
        print('Matched-native marker analysis:',entry['label'],flush=True)
    # Retain the query and donor metadata carried by local predictions.
    scores=pd.concat(source_frames,ignore_index=True);keys=['slice','query_cell_id','context_source_cell_id','context_kind','query_cell_type']
    # Sampling metadata comes from the local prediction file, not another run.
    result=scores
    if result.duplicated(keys).any():
        raise ValueError('Duplicate query/context records')
    controls=result.loc[result.context_kind.isin(['original','unchanged_control']),'delta_marker_auroc'].dropna();
    if not np.allclose(controls,0,atol=1e-12):
        raise ValueError('Unchanged-context controls differ from originals')
    result.to_csv(out/'marker_auroc_context_responses.csv.gz',index=False,compression='gzip')
    coverage=pd.DataFrame(audits);coverage.to_csv(run/'query_marker_eligibility.csv.gz',index=False,compression='gzip')
    coverage.groupby(['slice','slice_label','query_cell_type','query_family','status'],observed=True).size().rename('n_queries').reset_index().to_csv(run/'query_marker_coverage_by_slice_type.csv',index=False)
    pd.DataFrame(gene_coverage).to_csv(run/'native_marker_gene_coverage.csv',index=False)
    section=pd.DataFrame(section_rows);assert len(section)==len(markers)*4*9*len(design['slices'])
    section.to_csv(win_out/'section_gene_win_correlations.csv',index=False)
    section.groupby(['slice','slice_label','group_kind','query_group','gene'],sort=False).agg(median_rho=('rho','median'),n_supported_strata=('rho','count')).reset_index().to_csv(win_out/'section_stratum_medians.csv',index=False)
    summary=[]
    for (kind,family,gene),part in section.groupby(['group_kind','query_group','gene'],sort=False):
        valid=part.loc[part.rho.notna()];med=valid.rho.median() if len(valid) else np.nan;per_section=valid.groupby('slice').rho.median()
        summary.append(dict(group_kind=kind,query_group=family,gene=gene,median_rho=med,n_sections=len(per_section),
            n_positive=int((per_section>0).sum()),n_negative=int((per_section<0).sum()),
            n_same_sign=int((np.sign(per_section)==np.sign(med)).sum()) if len(valid) else 0,n_section_stratum_units=len(valid),
            n_possible_units=len(design['slices'])*9,n_distinct_strata=len(valid[['burden_stratum','original_context_stratum']].drop_duplicates())))
    pd.DataFrame(summary).to_csv(win_out/'gene_win_correlation_summary.csv',index=False)
    completion=dict(n_queries=len(coverage),n_valid_queries=int(coverage.status.eq('valid').sum()),n_rows=len(result),unchanged_controls_zero=True)
    (run/'analysis_complete.json').write_text(json.dumps(completion,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(config,indent=2)+'\n');(win_out/'complete.json').write_text(json.dumps(config,indent=2)+'\n')
    print(completion,flush=True)
    return out,win_out

if __name__=='__main__':
    import sys
    compute_native_marker_analysis(sys.argv[1])
