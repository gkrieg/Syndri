"""Estimate native associations within exact query types before superclass aggregation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .context_control import compute_control_native, weighted_slope


def compute_control_native_within_type(run,min_queries_per_type_stratum=5,min_queries_per_fit=10,min_native_levels=3):
    run=Path(run);out=run/'auroc_control_native_within_type';out.mkdir(exist_ok=True)
    config=dict(native_fit='weighted OLS with intercept within section x exact query cell type x burden',
        controlled='same queries: weighted mean of within-query context-swap OLS slopes',
        weights='equal original-neighbor strata within exact type/burden/section, equal queries within each',
        superclass='equal arithmetic mean of paired-valid exact-type coefficients within section/family/burden',
        sections='coordinate-wise median and IQR of section coefficients',
        min_queries_per_type_stratum=min_queries_per_type_stratum,min_queries_per_fit=min_queries_per_fit,min_native_levels=min_native_levels)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())!=config:raise ValueError('Within-type config differs')
        return out
    previous=compute_control_native(run,min_queries_per_type_stratum=min_queries_per_type_stratum)
    q=pd.read_csv(previous/'query_controlled_slopes.csv.gz',dtype={'query_neighbor_level':str})
    # Recalculate weights for exact-type native fits; do not reuse superclass weights.
    q=q.drop(columns=['weight','included','n_queries_type_stratum'],errors='ignore')
    base=['slice','query_family','query_cell_type','query_burden_quantile']
    strata=base+['query_neighbor_level']
    q['n_queries_type_stratum']=q.groupby(strata,observed=True).query_cell_id.transform('size')
    q['included']=q.n_queries_type_stratum.ge(min_queries_per_type_stratum)
    kept=q.loc[q.included].copy()
    kept['n_original_strata']=kept.groupby(base,observed=True).query_neighbor_level.transform('nunique')
    kept['within_type_weight']=1/(kept.n_queries_type_stratum*kept.n_original_strata)
    type_rows=[]
    for gid,part in kept.groupby(base,observed=True):
        w=part.within_type_weight.to_numpy(float)
        assert np.isclose(w.sum(),1)
        x=part.query_malignant_token_count.to_numpy(float);y=part.marker_auroc.to_numpy(float)
        status='valid' if len(part)>=min_queries_per_fit and np.unique(x).size>=min_native_levels else 'insufficient_native_support'
        ctrl=float(w@part.beta_ctrl_query.to_numpy(float));native=weighted_slope(x,y,w) if status=='valid' else np.nan
        if status=='valid':
            assert np.isclose(native,LinearRegression().fit(x[:,None],y,sample_weight=w).coef_[0],atol=1e-12)
        type_rows.append(dict(zip(base,gid),beta_ctrl=ctrl,beta_native=native,n_queries=len(part),
                              n_original_strata=part.query_neighbor_level.nunique(),n_original_count_levels=np.unique(x).size,status=status))
    types=pd.DataFrame(type_rows)
    paired=types.loc[types.status.eq('valid')&np.isfinite(types.beta_ctrl)&np.isfinite(types.beta_native)].copy()
    keys=['slice','query_family','query_burden_quantile']
    section=paired.groupby(keys,observed=True).agg(beta_ctrl=('beta_ctrl','mean'),beta_native=('beta_native','mean'),
        n_queries=('n_queries','sum'),n_query_types=('query_cell_type','nunique'),
        contributing_cell_types=('query_cell_type',lambda s:'; '.join(sorted(s)))).reset_index()
    section['status']='valid'
    summaries=[]
    for gid,part in section.groupby(['query_family','query_burden_quantile'],observed=True):
        row=dict(zip(['query_family','query_burden_quantile'],gid),n_sections=len(part))
        for col in ['beta_ctrl','beta_native']:
            row[col]=part[col].median();row[col+'_q25']=part[col].quantile(.25);row[col+'_q75']=part[col].quantile(.75)
        summaries.append(row)
    # Retain per-query local weights and whether its type supplied both coefficients.
    paired_types=paired[base].assign(paired_type_valid=True)
    kept=kept.merge(paired_types,on=base,how='left',validate='many_to_one')
    kept['paired_type_valid']=kept.paired_type_valid.fillna(False).astype(bool)
    kept.to_csv(out/'query_weights_and_support.csv.gz',index=False,compression='gzip')
    types.to_csv(out/'cell_type_control_native_slopes.csv',index=False)
    section.to_csv(out/'section_control_native_slopes.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'control_native_summary.csv',index=False)
    (out/'complete.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Within-type native fits:',len(paired),'/',len(types),'; superclass/section/burden summaries:',len(section),flush=True)
    return out

if __name__=='__main__':
    import sys
    compute_control_native_within_type(sys.argv[1])
