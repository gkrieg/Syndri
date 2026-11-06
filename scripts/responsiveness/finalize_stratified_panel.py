"""Finalize a stratified responsiveness experiment and preserve sampling metadata."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
import anndata as ad
from all_types_panel import summarize
from rank_responsiveness import summarize_ranks
from malignant_distance import summarize_distance
from query_distance_responsiveness import summarize as summarize_spread


def enrich_query_results(run,queries,context_metadata):
    query_columns=[c for c in queries if c not in {'query_cell_type'}]
    donor_columns=[c for c in context_metadata if c not in {'context_kind'}]
    for path in run.rglob('*.csv*'):
        if not (path.name.endswith('.csv') or path.name.endswith('.csv.gz')):continue
        if 'contexts' in path.relative_to(run).parts:continue
        try:header=pd.read_csv(path,nrows=0)
        except pd.errors.EmptyDataError:continue
        if not {'slice','query_cell_id'}.issubset(header.columns):continue
        extra=[c for c in query_columns if c not in header.columns and c not in {'slice','query_cell_id'}]
        context_extra=[c for c in donor_columns if c not in header.columns and c not in {'slice','context_source_cell_id'}]
        if not extra and not ('context_source_cell_id' in header and context_extra):continue
        table=pd.read_csv(path)
        if extra:table=table.merge(queries[['slice','query_cell_id',*extra]],on=['slice','query_cell_id'],how='left',validate='many_to_one')
        if 'context_source_cell_id' in table and context_extra:
            table=table.merge(context_metadata[['slice','context_source_cell_id',*context_extra]],on=['slice','context_source_cell_id'],how='left',validate='many_to_one')
        temp=path.with_name(path.name+'.strata.tmp')
        table.to_csv(temp,index=False,compression='gzip' if path.name.endswith('.gz') else None)
        temp.replace(path)


def enrich_query_arrays(run,queries):
    if queries.query_cell_id.duplicated().any():raise ValueError('Query IDs must be globally unique for array annotation')
    lookup=queries.set_index('query_cell_id')
    fields=['query_neighbor_level','query_malignant_token_count','query_marker_burden',
            'query_burden_quantile','query_joint_stratum']
    for path in run.rglob('*.npz'):
        with np.load(path,allow_pickle=False) as stored:
            if 'query_cell_id' not in stored:continue
            arrays={key:stored[key] for key in stored.files}
        ids=np.asarray(arrays['query_cell_id']).astype(str)
        selected=lookup.reindex(ids)
        if selected.query_joint_stratum.isna().any():raise ValueError('Unmatched query strata in array output')
        for field in fields:
            values=selected[field].to_numpy()
            arrays[field]=values.astype(str) if values.dtype.kind in 'OU' else values
        temporary=path.with_name(path.name+'.strata.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(path)


def finalize(run):
    run=Path(run);design=json.loads((run/'design.json').read_text());markers=design['marker_genes']
    queries=pd.concat([pd.read_csv(run/'contexts'/f'{e["slice"]}_queries.csv') for e in design['slices']],ignore_index=True)
    queries.to_csv(run/'sampled_query_manifest.csv',index=False)
    all_metadata=[]
    required=['query_neighbor_level','query_marker_burden','query_burden_quantile','query_joint_stratum',
              'context_sampling_stratum','context_malignant_token_count','context_selected_for_shared_panel']
    for e in design['slices']:
        path=Path(e['predictions'])
        if not path.with_suffix('.complete.json').exists():raise FileNotFoundError(path)
        a=ad.read_h5ad(path,backed='r')
        try:obs=a.obs.copy()
        finally:a.file.close()
        missing=set(required)-set(obs)
        if missing:raise ValueError(f'Inference dropped sampling metadata: {missing}')
        obs=obs.rename(columns={'focal_cell_id':'query_cell_id'});obs['slice']=e['slice']
        columns=['slice','query_cell_id','query_cell_type','context_source_cell_id','context_kind',*required,
                 'query_malignant_token_count','query_marker_transcripts','query_total_transcripts',
                 'query_stratum_available','query_stratum_selected','context_stratum_available','context_stratum_selected']
        all_metadata.append(obs[columns])
    rowmeta=pd.concat(all_metadata,ignore_index=True)
    rowmeta.to_csv(run/'prediction_sampling_strata.csv.gz',index=False,compression='gzip')
    context_columns=['slice','context_source_cell_id','context_sampling_stratum','context_malignant_token_count',
                     'context_selected_for_shared_panel','context_stratum_available','context_stratum_selected']
    context_metadata=rowmeta[context_columns].drop_duplicates()
    if context_metadata.duplicated(['slice','context_source_cell_id']).any():raise ValueError('Context stratum metadata is inconsistent')
    summarize(run,response_scale='logit')
    summarize_ranks(run)
    summarize_distance(run)
    summarize_spread(run,marker_genes=markers,output_subdir='summaries_query_distance_responsiveness_markers32_20261019')
    subprocess.run([sys.executable,str(run/'context_baselines.py'),str(run)],check=True)
    for family in ['summaries_logits','summaries_ranks','summaries_distance','summaries_delta_context']:
        path=run/family/'complete.json';meta=json.loads(path.read_text());meta['marker_genes']=markers
        path.write_text(json.dumps(meta,indent=2)+'\n')
    design['actual_n_queries']=int(len(queries));design['expected_n_queries']=int(len(queries))
    (run/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    # All summary and plotting cells run against this experiment's saved predictions.
    staged=run/'executed_notebook.ipynb'
    shutil.copy2(run/'notebook_template.ipynb',staged)
    subprocess.run([sys.executable,str(run/'execute_panel_notebook.py'),str(staged)],check=True)
    enrich_query_results(run,queries,context_metadata)
    enrich_query_arrays(run,queries)
    destination=Path(design['notebook'])
    expected=design['notebook_source_sha256']
    published=False
    if hashlib.sha256(destination.read_bytes()).hexdigest()==expected:
        shutil.copy2(staged,destination);published=True
    (run/'complete.json').write_text(json.dumps(dict(n_queries=len(queries),n_slices=len(design['slices']),
        notebook=str(destination),published=published,executed_notebook=str(staged),marker_genes=markers,
        strata_in_predictions=True,strata_manifest=str(run/'prediction_sampling_strata.csv.gz')),indent=2)+'\n')
    print('Completed stratified run:',run,'published:',published,flush=True)

if __name__=='__main__':finalize(Path(sys.argv[1]))
