import numpy as np
import os
import anndata as ad
import scanpy as sc
import pandas as pd
import pickle
import sys
sys.path.append('/home/jwinchel/Heimdall/')
from Heimdall.utils import convert_to_ensembl_ids

def load_GSE127465(file_path,st_file='/compute/oven-0-13/jwinchel/Lung5_Rep1/Lung5_Rep1-Flat_files_and_images/Lung5_Rep1_exprMat_file.csv'):
    if os.path.join(file_path,'adata/GSE127465_tumor.h5ad'):
        print("Preprocessed data file found. Loading...")
        adata = sc.read_h5ad(os.path.join(file_path,'adata/GSE127465_tumor.h5ad'))
        return adata
    metadata_file = [x for x in os.listdir(file_path) if 'human_cell_metadata' in x]
    if metadata_file:
        metadata_file = metadata_file[0]
    else:
        raise ValueError('Metadata file not found in the specified directory.')
    
    meta = pd.read_csv(os.path.join(file_path, metadata_file), sep='\t')

    tissue_subset = meta[meta['Tissue']=='tumor']['Library'].unique()
    raw_count_files = sorted([x for x in os.listdir(file_path) if x.endswith('_raw_counts.tsv') and 'human' in x])
    
    # check that raw count contains of the tissues in the subset
    raw_count_files = [x for x in raw_count_files if any(tissue in x for tissue in tissue_subset)]

    print(f"Number of raw count files found: {len(raw_count_files)}")
    print(f"Metadata file found: {metadata_file}")
    print(f"Raw count files found: {raw_count_files}")

    common_genes = get_common_genes(os.path.join(file_path, raw_count_files[0]), st_file)

    # read in raw count files and concatenate them into a single anndata object
    adata_list = []
    for i,f in enumerate(raw_count_files):
        
        f_adata = sc.read_csv(os.path.join(file_path, f), delimiter='\t')
        f_adata,_ = convert_to_ensembl_ids(f_adata,file_path)
        f_adata = f_adata[:,common_genes]
        print(f_adata.shape)
        fname = f.split('/')[-1]
        fname_fields = fname.split('_')
        f_adata.obs['species'] = fname_fields[1]
        f_adata.obs['filename'] = fname.split('.')[0]
        f_adata.obs['sample'] = fname_fields[2]
        f_adata.obs['GSM_ID'] = fname_fields[0]
        adata_list.append(f_adata)

    adata = ad.concat(adata_list,join='outer')

    zero_genes = adata.var_names[adata.X.sum(axis=0) == 0].tolist()
    print(f"Genes never expressed (dropped): {zero_genes}")
    adata = adata[:, ~adata.var_names.isin(zero_genes)]
    
    adata.obs['cell_ID_sample'] = adata.obs.index + '_' + adata.obs['sample']
    adata.obs['cell_ID'] = adata.obs.index
    adata.obs.set_index('cell_ID_sample',inplace=True)
    
    meta['Barcode_Library'] = meta['Barcode'] + '_' + meta['Library']

    adata.obs = adata.obs.merge(meta, left_on='cell_ID_sample', right_on='Barcode_Library',how='left')
    obj_cols = [c for c in adata.obs.columns if adata.obs[c].dtype== 'object']
    adata.obs[obj_cols] = adata.obs[obj_cols].fillna('NA').astype(str)
    adata.write_h5ad(os.path.join(file_path,'adata/GSE127465_tumor.h5ad'))
    adata[~adata.obs['Major cell type'].isna()].write_h5ad(os.path.join(file_path,'adata/GSE127465_tumor_annotated.h5ad'))
    return adata
    
    
def get_common_genes(sc_file, st_file,save_dir='/compute/oven-0-13/jwinchel/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)

    if os.path.exists(os.path.join(save_dir,'gene_overlap.csv')):
        gene_overlap = pd.read_csv(os.path.join(save_dir,'gene_overlap.csv'))
        return gene_overlap['ensembl_gene'].tolist() 
    st_data = sc.read_csv(st_file)
    st_data = st_data[:10]
    st_data.obs[['fov', 'cell_ID']] = st_data.X[:,:2]

    sc_data = sc.read_csv(sc_file, delimiter='\t')
    sc_data = sc_data[:10]
    
    st_data = st_data[:,~st_data.var_names.isin(['fov','cell_ID'])]

    st_data, st_gene_mapping = convert_to_ensembl_ids(st_data,save_dir)

    sc_data, _ = convert_to_ensembl_ids(sc_data,save_dir)

    gene_overlap = sorted(list(set(sc_data.var_names) & set(st_data.var_names)))

    # Pickling (saving to a file)
    with open(os.path.join(save_dir, 'gene_mapping.pickle'), 'wb') as f:
        # Use the highest protocol for best performance and features
        pickle.dump(st_gene_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

    return gene_overlap



def load_nsclc_cosmx(data_dir,sc_file='/compute/oven-0-13/jwinchel/scRNA-seq/GSM3635278_human_p1t1_raw_counts.tsv',cell_meta_file='/compute/oven-0-13/jwinchel/NSCLC_ST_cell_metadata.csv',save_dir='/compute/oven-0-13/jwinchel/spatial_adata/'):
    # get spatial transcriptomics files
    st_files = sorted([os.path.join(data_dir,x,x+'-Flat_files_and_images/',x+"_exprMat_file.csv") for x in os.listdir(data_dir) if x.startswith('Lung')])
    met_files = sorted([os.path.join(data_dir,x,x+'-Flat_files_and_images/',x+"_metadata_file.csv") for x in os.listdir(data_dir) if x.startswith('Lung')])

    ad_list = []
    common_genes = get_common_genes(sc_file, st_files[0],data_dir)

    metas = []
    for i,f in enumerate(st_files):
        df = pd.read_csv(f)

        # read metadata
        met_df = pd.read_csv(met_files[i])
        met_df['tissue'] = "_".join(f.split('/')[-1].split('_')[:-2])
        metas.append(met_df)

        # separate metadata from gene counts
        obs = df[['fov', 'cell_ID']].copy()
        expr = df.drop(columns=['fov', 'cell_ID']).copy()

        # build AnnData
        st_sample = ad.AnnData(X=expr.values, obs=obs)
        st_sample.var_names = expr.columns
        st_sample.obs_names = obs['cell_ID'].astype(str)

        # convert genes to ensembl IDs
        st_sample, _ = convert_to_ensembl_ids(st_sample, data_dir)

        # restrict to common genes
        st_sample = st_sample[:, st_sample.var_names.isin(common_genes)].copy()
        st_sample.obs['tissue'] = f.split('/')[4]

        ad_list.append(st_sample)

    metadata = pd.concat(metas, join='outer')
    st_data = ad.concat(ad_list, join='outer')
    st_data.obs = st_data.obs.reset_index(drop=True)
    
    st_data.obs = st_data.obs.merge(metadata[['tissue','fov','cell_ID','CenterX_local_px','CenterY_local_px','CenterX_global_px','CenterY_global_px']],on=['tissue','fov','cell_ID'],how='left')

    st_data.obs['cell_ID'] = st_data.obs['cell_ID'].astype(str)

    cell_meta = pd.read_csv(cell_meta_file)
    cell_meta['cell_ID'] = cell_meta['cell_ID'].astype(str)
    cell_meta = cell_meta.rename(columns={'cell_ID':'cell_ID_long'})
    cell_meta['cell_ID'] = cell_meta['cell_ID_long'].str.split('_').str[-1].astype(str)
    st_data.obs = st_data.obs.merge(cell_meta.drop(columns=['tissue']),left_on=['tissue','fov','cell_ID'],right_on=['Run_Tissue_name','fov','cell_ID'],how='left')
    
    cat_cols = [c for c in st_data.obs.columns if st_data.obs[c].dtype in ['category', 'O']]
    st_data.obs[cat_cols] = st_data.obs[cat_cols].astype(str)

    st_data.obs.reset_index(drop=True, inplace=True)
    st_data.obs.index = st_data.obs.index.astype(str)
    st_data.obs["cell_type_merged"] = st_data.obs["cell_type"].map({"T CD4 naive": "T CD4+", 
                                                                    "T CD4 memory": "T CD4+", 
                                                                    "Treg": "T CD4+",
                                                                    "T CD8 naive": "T CD8+", 
                                                                    "T CD8 memory": "T CD8+",
                                                                    "pDC":"DC",
                                                                    "mDC":"DC"}).fillna(st_data.obs["cell_type"])


    st_data.obs = st_data.obs.rename(columns={'cell_type_merged':'cell_type','cell_type':'cell_type_orig'})
    st_data.obs['cell_type'] = st_data.obs['cell_type'].apply(lambda x: 'tumor' if x.startswith('tumor') else x)
    
    tissue_list = st_data.obs['tissue'].unique()
    for tissue in tissue_list:
        tissue_adata = st_data[st_data.obs['tissue'] == tissue]
        # tissue_adata.obs = tissue_adata.obs.fillna('NA')
        tissue_adata.write_h5ad(os.path.join(save_dir, f'cosmx_nsclc_{tissue.replace("_","-")}_adata.h5ad'))
        tissue_adata = tissue_adata[tissue_adata.obs['cell_type']!='NA']
        tissue_adata.write_h5ad(os.path.join(save_dir, f'cosmx_nsclc_{tissue.replace("_","-")}_adata_annotated.h5ad'))

    return st_data

