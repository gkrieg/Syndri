"""Expand the existing T-cell pseudobulk cosine to all annotated types."""
from pathlib import Path
import gc,json,os,sys
import h5py,anndata as ad
import numpy as np,pandas as pd
from scipy import sparse
sys.path.insert(0,'/home/jwinchel/Syndri/eval')
import metrics_eval as ev


def metadata(path):
 with h5py.File(path) as handle:return ad.io.read_elem(handle['obs']),ad.io.read_elem(handle['var'])

def matrix(path,layer=None):
 with h5py.File(path) as handle:result=ad.io.read_elem(handle['X'] if not layer else handle['layers'][layer])
 result=sparse.csr_matrix(result)
 if not np.isfinite(result.data).all() or (result.data<0).any():raise ValueError(f'Invalid counts: {path}:{layer}')
 return result

def settings(config):
 parser=ev.build_parser();known={a.dest for a in parser._actions}
 parser.set_defaults(**{k:v for k,v in config.items() if k in known})
 return parser.parse_args([])

def run(entry):
 out=Path(entry['output']);out.mkdir(parents=True,exist_ok=False)
 source=Path(entry['input']);obs,var=metadata(source)
 if not obs.index.is_unique or not var.index.is_unique:raise ValueError('Nonunique cell/gene identifiers')
 cfg=json.loads(Path(entry['config']).read_text());args=settings(cfg)
 args.n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK',4))
 args.tcell_cosine_cell_type_pattern='';args.tcell_cosine_cell_types=''
 args.reference_data=entry['reference'];args.reference_cell_type_key=entry['reference_label']
 args.reference_counts_layer=entry['reference_layer'];args.tcell_cell_type_key=entry['spatial_label']
 args.tcell_cosine_cell_type_map=entry.get('cell_type_map','')
 # Match prior cancer T-cell eligibility. Brain datasets require valid labels,
 # but no spatial coordinates are needed for their pseudobulk calculation.
 valid=obs[args.tcell_cell_type_key].notna().to_numpy()
 valid &= obs[args.tcell_cell_type_key].astype(str).str.strip().str.lower().ne('nan').to_numpy()
 excluded=ev.parse_str_list(args.tcell_exclude_labels) or ev.parse_str_list(args.exclude_labels)
 valid &= ~obs[args.tcell_cell_type_key].astype(str).isin(excluded).to_numpy()
 if args.cell_type_specific_removal:
  group=args.removal_group_key or args.label_key
  valid &= obs[group].notna().to_numpy() & obs[group].astype(str).str.lower().ne('nan').to_numpy()
 if entry['match_cancer_tcell_eligibility'] and args.filter_tcell_invalid_cells:
  valid &= np.isfinite(obs[[args.coord_x,args.coord_y]].to_numpy(dtype=float)).all(axis=1)
 idx=np.flatnonzero(valid);selected_obs=obs.iloc[idx].copy()
 raw=matrix(source)[idx]
 probabilities=matrix(source,args.admix_probs_layer)[idx]
 ref=ad.read_h5ad(entry['reference'],backed='r');cache={};frames=[];audit=[]
 def score(counts,method,condition,input_file,layer,target=np.nan,cleaning=None):
  current=ad.AnnData(X=counts,obs=selected_obs.copy(),var=var.copy())
  table=ev.compute_tcell_cosine_similarity(current,ref,spatial_counts_layer=None,tissue=source.stem,
   source_file=str(input_file),condition=condition,args=args,reference_profile_cache=cache)
  if table.empty:raise ValueError('No cosine rows')
  if 'spatial_cell_type' not in table:raise ValueError(table.to_dict('records'))
  table['method']=method;table['dataset']=entry['dataset'];table['slice']=source.stem
  table['target_removed_fraction']=target;table['cleaning_mode']=cleaning or ''
  table['matrix_file']=str(input_file);table['matrix_layer']=layer or 'X';table['reference_file']=entry['reference']
  table['status']=np.where(table['error'].fillna('').eq(''),'ok','missing_reference_type')
  errors=table['error'].fillna('')
  unexpected=errors.ne('') & ~errors.str.startswith('Missing reference cell type ')
  if unexpected.any():raise ValueError(table.loc[unexpected,'error'].tolist())
  table.loc[table.status.eq('ok')&table.cosine_similarity.isna(),'status']='zero_profile'
  frames.append(table)
  audit.append(dict(method=method,input=str(input_file),layer=layer or 'X',status='evaluated',n_types=len(table),n_success=int(table.status.eq('ok').sum())))
  del current;gc.collect()
 try:
  score(raw,'Raw','raw_X',source,'X')
  split_layer=entry['split_layer']
  score(matrix(source,split_layer)[idx],'SPLIT','SPLIT',source,split_layer)
  groups=selected_obs[args.removal_group_key or args.label_key] if args.cell_type_specific_removal else None
  target=float(entry['syndri_target'])
  cleaned,*_=ev.cleaned_counts_for_removed_fraction_maybe_grouped(raw,probabilities,target,args,labels=groups)
  score(cleaned,'Syndri',f'cleaned_admixed_removed_frac_{target:g}',source,'X + '+args.admix_probs_layer,target,args.cleaning_mode)
  del cleaned,probabilities;gc.collect()
  for competitor in entry['competitors']:
   if not competitor.get('input'):
    audit.append(dict(method=competitor['method'],status='unavailable',reason=competitor.get('reason','No saved correction')));continue
   path=Path(competitor['input']);co,cv=metadata(path)
   if not co.index.is_unique or not cv.index.is_unique:raise ValueError(f'Nonunique competitor IDs: {path}')
   rows=co.index.get_indexer(obs.index);cols=cv.index.get_indexer(var.index)
   if (rows<0).any() or (cols<0).any():raise ValueError(f'Missing competitor cells/genes: {path}')
   counts=matrix(path,competitor['layer'])[rows[idx]][:,cols]
   score(counts,competitor['method'],competitor['condition'],path,competitor['layer'])
   del counts;gc.collect()
 finally:ref.file.close()
 table=pd.concat(frames,ignore_index=True)
 if table.duplicated(['method','spatial_cell_type','reference_cell_type']).any():raise ValueError('Duplicate type/method rows')
 table.to_csv(out/'reference_cosine_similarity.csv',index=False)
 pd.DataFrame(audit).to_csv(out/'method_coverage.csv',index=False)
 provenance=dict(**entry,n_input_cells=len(obs),n_eligible_cells=len(idx),settings=vars(args),
  calculation='Existing metrics_eval.compute_tcell_cosine_similarity; sum counts per type, normalize pseudobulk to target_sum, log1p, cosine on shared genes',
  cell_types='All annotated types, empty selection regex; corresponding reference label by existing mapping',
  eligible_cell_ids_sha256=__import__('hashlib').sha256('\n'.join(selected_obs.index.astype(str)).encode()).hexdigest())
 (out/'config.json').write_text(json.dumps(provenance,indent=2)+'\n')
 (out/'complete.json').write_text(json.dumps(dict(n_rows=len(table),n_types=table.spatial_cell_type.nunique(),n_methods=table.method.nunique(),n_success=int(table.status.eq('ok').sum())),indent=2)+'\n')
 print('Completed all-type reference cosine:',out,flush=True)

if __name__=='__main__':
 manifest=Path(sys.argv[1]);entry=json.loads(manifest.read_text())[int(sys.argv[2])];run(entry)
