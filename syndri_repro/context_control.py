"""Controlled within-query and native across-query marker-AUROC slopes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def weighted_slope(x,y,w=None):
    x,y=np.asarray(x,float),np.asarray(y,float)
    w=np.ones(len(x),float) if w is None else np.asarray(w,float)
    w=w/w.sum();xc=x-np.sum(w*x);yc=y-np.sum(w*y)
    den=np.sum(w*xc*xc)
    return float(np.sum(w*xc*yc)/den) if den>1e-12 else np.nan


def compute_control_native(run,min_queries_per_type_stratum=5,min_query_context_levels=3):
    run=Path(run);out=run/'auroc_control_native';out.mkdir(exist_ok=True)
    config=dict(min_queries_per_type_stratum=min_queries_per_type_stratum,
        min_query_context_levels=min_query_context_levels,
        controlled='OLS with intercept per fixed query: delta_AUROC ~ delta_malignant_count, selected donors once each; then weighted mean of query slopes',
        native='weighted OLS with intercept: original_AUROC ~ original_malignant_count, on exactly the same queries',
        weights='equal original-context strata within section/family/burden; equal query types within each stratum; equal queries within type/stratum',
        baseline='each query original context',score='32-panel malignant markers versus expressed query-type ST4 native markers AUROC',
        summary='median and IQR across valid section coefficients; no pair-level p values')
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())!=config:raise ValueError('Control/native configuration differs')
        return out
    table=pd.read_csv(run/'marker_auroc_context_responses/marker_auroc_context_responses.csv.gz',dtype={'query_neighbor_level':str})
    pairs=table.loc[table.context_selected_for_shared_panel.eq(True)&table.context_kind.ne('unchanged_control')&np.isfinite(table.delta_marker_auroc)].copy()
    pairs['delta_n']=pairs.context_malignant_token_count-pairs.query_malignant_token_count
    original=table.loc[table.context_kind.eq('original')].copy()
    keys=['slice','query_cell_id']
    assert not original.duplicated(keys).any()
    records=[]
    for (section,query),group in pairs.groupby(keys,sort=False,observed=True):
        x=group.delta_n.to_numpy(float);y=group.delta_marker_auroc.to_numpy(float)
        slope=weighted_slope(x,y) if len(np.unique(x))>=min_query_context_levels else np.nan
        records.append(dict(slice=section,query_cell_id=query,beta_ctrl_query=slope,n_contexts=len(group),n_context_levels=len(np.unique(x))))
    query=original[['slice','query_cell_id','query_cell_type','query_burden_quantile','query_neighbor_level','query_malignant_token_count','marker_auroc','query_marker_burden']].merge(pd.DataFrame(records),on=keys,how='inner',validate='one_to_one')
    mapping=pd.read_csv(run/'burden_stratified_context_auroc_response/cell_family_mapping.csv').set_index('query_cell_type').query_family
    query['query_family']=query.query_cell_type.map(mapping)
    if query.query_family.isna().any():raise ValueError('Unmapped family')
    query=query.loc[np.isfinite(query.beta_ctrl_query)&np.isfinite(query.marker_auroc)&query.query_burden_quantile.isin(['low','medium','high'])].copy()
    base=['slice','query_family','query_burden_quantile']
    strata=base+['query_neighbor_level','query_cell_type']
    query['n_queries_type_stratum']=query.groupby(strata,observed=True).query_cell_id.transform('size')
    query['included']=query.n_queries_type_stratum.ge(min_queries_per_type_stratum)
    kept=query.loc[query.included].copy()
    kept['n_types_in_original_stratum']=kept.groupby(base+['query_neighbor_level'],observed=True).query_cell_type.transform('nunique')
    kept['n_original_strata']=kept.groupby(base,observed=True).query_neighbor_level.transform('nunique')
    kept['weight']=1/(kept.n_queries_type_stratum*kept.n_types_in_original_stratum*kept.n_original_strata)
    summaries=[]
    for group_id,part in kept.groupby(base,observed=True):
        w=part.weight.to_numpy(float);assert np.isclose(w.sum(),1)
        x=part.query_malignant_token_count.to_numpy(float);y=part.marker_auroc.to_numpy(float)
        status='valid' if part.query_cell_id.nunique()>=10 and np.unique(x).size>=3 else 'insufficient_native_support'
        ctrl=float(np.sum(w*part.beta_ctrl_query));native=weighted_slope(x,y,w) if status=='valid' else np.nan
        if status=='valid':
            expected=LinearRegression().fit(x[:,None],y,sample_weight=w).coef_[0]
            assert np.isclose(native,expected,atol=1e-12)
        summaries.append(dict(zip(base,group_id),beta_ctrl=ctrl,beta_native=native,
            beta_ctrl_unweighted=part.beta_ctrl_query.mean(),beta_native_unweighted=weighted_slope(x,y),
            n_queries=len(part),n_query_types=part.query_cell_type.nunique(),n_original_strata=part.query_neighbor_level.nunique(),
            n_original_count_levels=np.unique(x).size,status=status))
    section=pd.DataFrame(summaries)
    rows=[]
    for group_id,part in section.groupby(['query_family','query_burden_quantile'],observed=True):
        valid=part.loc[part.status.eq('valid')&np.isfinite(part.beta_ctrl)&np.isfinite(part.beta_native)]
        row=dict(zip(['query_family','query_burden_quantile'],group_id),n_sections=len(valid))
        for col in ['beta_ctrl','beta_native']:
            row[col]=valid[col].median();row[col+'_q25']=valid[col].quantile(.25);row[col+'_q75']=valid[col].quantile(.75)
        rows.append(row)
    query=query.merge(kept[keys+['weight']],on=keys,how='left',validate='one_to_one')
    query.to_csv(out/'query_controlled_slopes.csv.gz',index=False,compression='gzip')
    section.to_csv(out/'section_control_native_slopes.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'control_native_summary.csv',index=False)
    (out/'complete.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Controlled/native coefficients:',len(kept),'queries;',len(section),'section/family/burden groups',flush=True)
    return out

if __name__=='__main__':
    import sys
    compute_control_native(sys.argv[1])
