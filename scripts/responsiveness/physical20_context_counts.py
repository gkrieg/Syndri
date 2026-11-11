"""Cached physical malignant-cell counts among 20 nearest non-self cells."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def ensure_physical20_context_counts(run):
    import anndata as ad
    from scipy.spatial import cKDTree
    run = Path(run)
    out = run / 'physical20_context_counts'
    out.mkdir(exist_ok=True)
    paths = {}
    for entry in json.loads((run / 'design.json').read_text())['slices']:
        path = out / (entry['slice'] + '.csv.gz')
        paths[entry['slice']] = path
        if path.exists():
            continue
        source = ad.read_h5ad(entry['input'], backed='r')
        try:
            obs = source.obs.copy()
        finally:
            source.file.close()
        xy = obs[['x_centroid','y_centroid']].apply(pd.to_numeric,errors='coerce').to_numpy()
        finite = np.flatnonzero(np.isfinite(xy).all(axis=1))
        if len(finite) < 21:
            raise ValueError('Fewer than 20 non-self neighbors available')
        tree = cKDTree(xy[finite])
        distances, indices = tree.query(xy[finite], k=21, workers=1)
        nonself = indices != np.arange(len(finite))[:, None]
        selected = nonself & (nonself.cumsum(axis=1) <= 20)
        if not np.all(selected.sum(axis=1) == 20):
            raise ValueError('Expected 20 non-self neighbors')
        malignant = obs.harmony_cell_type.astype(str).eq('malignant cell').to_numpy()[finite]
        counts = (malignant[indices] & selected).sum(axis=1)
        radius = np.where(selected, distances, 0).max(axis=1)
        pd.DataFrame(dict(cell_id=obs.index.astype(str).to_numpy()[finite],
            malignant_neighbors_k20=counts, distance_to_20th_neighbor_um=radius)).to_csv(path,index=False,compression='gzip')
        print('Physical 20-neighbor counts:',entry['label'],len(finite),'cells',flush=True)
    (out/'definition.json').write_text(json.dumps(dict(k=20,exclude_self=True,
        label='harmony_cell_type == malignant cell',neighbor_pool='all finite-coordinate cells in the same section',
        coordinate_columns=['x_centroid','y_centroid'],metric='2D Euclidean',counts_filter=None),indent=2)+'\n')
    return paths
