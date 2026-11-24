"""Transcript locations for existing, held-out CosMx synthetic admixture counts.

Coordinates retain the legacy tx_file.csv convention (global/local pixels and
discrete z image numbers). Only distance weighting uses micrometers. Pixel size
is 0.18 um in NanoString's 2021 SMI export documentation:
https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/SMI-ReadMe.html
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

PIXEL_SIZE_UM = 0.18
TEST_SLICES = ('Lung12', 'Lung13', 'Lung9_Rep1', 'Lung9_Rep2')
FORMAT_SOURCE = 'https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/SMI-ReadMe.html'


def _signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + '\n')


def integer_counts(matrix):
    result = sparse.csr_matrix(matrix).copy()
    result.eliminate_zeros()
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError('Transcript counts must be finite and nonnegative')
    if not np.equal(result.data, np.rint(result.data)).all():
        raise ValueError('Transcript locations require integer counts; no rounding is performed')
    return result.astype(np.int64)


def _same_counts(left, right, message):
    difference = sparse.csr_matrix(left) - sparse.csr_matrix(right)
    difference.eliminate_zeros()
    if difference.nnz:
        raise AssertionError(message)


def real_transcript_profile(path, cache):
    """Stream one real slice once: global z frequencies, targets, coordinate examples."""
    import pyarrow.csv as pacsv
    path, cache = Path(path), Path(cache)
    folder = cache / ('real_transcripts_' + _key(_signature(path)))
    profile_path = folder / 'profile.json'
    if profile_path.exists():
        return json.loads(profile_path.read_text()), pd.read_csv(folder / 'examples.csv')
    folder.mkdir(parents=True, exist_ok=True)
    with path.open() as stream:
        columns = next(csv.reader(stream))
    required = ['fov', 'cell_ID', 'x_global_px', 'y_global_px', 'x_local_px', 'y_local_px', 'z', 'target']
    if not set(required).issubset(columns):
        raise ValueError(f'Unsupported tx_file.csv schema: {columns}')
    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=8 * 1024**2),
                           convert_options=pacsv.ConvertOptions(include_columns=required))
    frequencies, targets, examples, seen = {}, set(), [], {}
    n_rows = 0
    for batch in reader:
        frame = batch.to_pandas()
        z = frame.z.to_numpy()
        if not np.isfinite(z).all():
            raise ValueError('Real transcript z contains nonfinite values')
        values, counts = np.unique(z, return_counts=True)
        for value, count in zip(values, counts):
            frequencies[float(value)] = frequencies.get(float(value), 0) + int(count)
        targets.update(frame.target.unique())
        n_rows += len(frame)
        for fov, group in frame.loc[frame.cell_ID.gt(0)].groupby('fov', sort=False):
            take = max(0, 256 - seen.get(int(fov), 0))
            if take:
                current = group.head(take)
                examples.append(current)
                seen[int(fov)] = seen.get(int(fov), 0) + len(current)
    if not n_rows:
        raise ValueError('Original transcript file is empty')
    planes = sorted(frequencies)
    if all(float(z).is_integer() for z in planes):
        planes = [int(z) for z in planes]
    profile = dict(columns=columns, z_planes=planes,
                   z_counts=[frequencies[float(z)] for z in planes], targets=sorted(targets),
                   n_real_transcripts=n_rows, input_signature=_signature(path))
    observed = pd.concat(examples, ignore_index=True)
    observed.to_csv(folder / 'examples.csv', index=False)
    _json(profile_path, profile)
    return profile, observed


@dataclass
class Region:
    fov: int
    cell_id: int
    rows: np.ndarray
    cols: np.ndarray
    edge_centers: np.ndarray
    edge_halfwidths: np.ndarray
    tree: cKDTree


class MaskGeometry:
    """Bounded memory cache plus persistent mask, bounding-box, and pixel caches."""
    def __init__(self, raw, slice_name, cache, pixel_size_um=PIXEL_SIZE_UM):
        self.raw, self.cache = Path(raw), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.scale = float(pixel_size_um)
        if self.scale <= 0:
            raise ValueError('Pixel size must be positive')
        positions = pd.read_csv(self.raw / f'{slice_name}_fov_positions_file.csv')
        if positions.fov.duplicated().any():
            raise ValueError('Duplicate FOV origins')
        self.origins = positions.set_index('fov')[['x_global_px', 'y_global_px']]
        self.masks, self.regions, self.flip_y = OrderedDict(), OrderedDict(), {}

    def _load_fov(self, fov):
        fov = int(fov)
        if fov in self.masks:
            self.masks.move_to_end(fov)
            return self.masks[fov]
        source = self.raw / 'CellLabels' / f'CellLabels_F{fov:03d}.tif'
        folder = self.cache / ('mask_' + _key(_signature(source)))
        folder.mkdir(exist_ok=True)
        path = folder / 'labels.npy'
        bbox_path = folder / 'boxes.npz'
        if not path.exists() or not bbox_path.exists():
            # Pillow/libtiff decodes the original LZW TIFFs without imagecodecs.
            with Image.open(source) as image:
                labels = np.asarray(image).copy()
            if labels.ndim != 2 or labels.dtype.kind not in 'iu':
                raise ValueError('CellLabels must be a two-dimensional integer label image')
            boxes = ndimage.find_objects(labels)
            ids = np.array([i+1 for i, box in enumerate(boxes) if box is not None], dtype=np.int32)
            bounds = np.array([[boxes[i-1][0].start, boxes[i-1][0].stop,
                                boxes[i-1][1].start, boxes[i-1][1].stop] for i in ids], dtype=np.int32)
            np.save(path, labels)
            np.savez(bbox_path, ids=ids, bounds=bounds)
        labels = np.load(path, mmap_mode='r')
        with np.load(bbox_path) as boxes:
            bounds = dict(zip(boxes['ids'].tolist(), boxes['bounds']))
        self.masks[fov] = labels, bounds, folder
        while len(self.masks) > 2:
            self.masks.popitem(last=False)
        return self.masks[fov]

    def origin(self, fov):
        if int(fov) not in self.origins.index:
            raise ValueError(f'Missing origin for FOV {fov}')
        return self.origins.loc[int(fov)].to_numpy(float)

    def pixels(self, fov, cell_id):
        labels, boxes, folder = self._load_fov(fov)
        if int(cell_id) not in boxes:
            raise ValueError(f'Missing segmentation region fov={fov}, cell_ID={cell_id}')
        path = folder / f'cell_{int(cell_id)}.npz'
        if path.exists():
            with np.load(path) as data:
                return data['rows'], data['cols']
        r0, r1, c0, c1 = boxes[int(cell_id)]
        rows, cols = np.nonzero(labels[r0:r1, c0:c1] == int(cell_id))
        rows, cols = (rows+r0).astype(np.int32), (cols+c0).astype(np.int32)
        np.savez_compressed(path, rows=rows, cols=cols)
        return rows, cols

    def local(self, fov, rows, cols):
        height = self._load_fov(fov)[0].shape[0]
        return np.column_stack([cols, height-rows if self.flip_y[int(fov)] else rows])

    def global_xy(self, fov, local):
        return np.asarray(local) + self.origin(fov)

    def contains(self, fov, cell_id, local):
        labels = self._load_fov(fov)[0]
        x, y = np.asarray(local).T
        cols = np.floor(x).astype(np.int64)
        rows = np.floor(labels.shape[0]-y if self.flip_y[int(fov)] else y).astype(np.int64)
        valid = (rows >= 0) & (rows < labels.shape[0]) & (cols >= 0) & (cols < labels.shape[1])
        result = np.zeros(len(rows), dtype=bool)
        result[valid] = labels[rows[valid], cols[valid]] == int(cell_id)
        return result

    def calibrate(self, fov, metadata, examples):
        """Infer raster orientation from observed tx/cell labels, check all offsets."""
        fov = int(fov)
        rows = metadata.loc[metadata.fov.eq(fov)]
        if rows.empty:
            raise ValueError(f'No spatial metadata for FOV {fov}')
        local = rows[['CenterX_local_px', 'CenterY_local_px']].to_numpy(float)
        global_xy = rows[['CenterX_global_px', 'CenterY_global_px']].to_numpy(float)
        error = float(np.abs(global_xy - local - self.origin(fov)).max())
        if error > 1e-5:
            raise ValueError(f'AnnData/FOV-origin coordinate mismatch in FOV {fov}: {error}')
        sample = examples.loc[examples.fov.eq(fov)]
        if sample.empty:
            raise ValueError(f'No observed coordinate examples in FOV {fov}')
        points = sample[['x_local_px', 'y_local_px']].to_numpy(float)
        observed_error = float(np.abs(sample[['x_global_px', 'y_global_px']].to_numpy(float)
                                      - self.global_xy(fov, points)).max())
        if observed_error > 1e-5:
            raise ValueError(f'Observed local/global transform differs in FOV {fov}')
        labels = self._load_fov(fov)[0]
        scores = []
        for flip in [False, True]:
            self.flip_y[fov] = flip
            cols = np.floor(points[:, 0]).astype(int)
            yy = labels.shape[0]-points[:, 1] if flip else points[:, 1]
            rr = np.floor(yy).astype(int)
            valid = (rr>=0)&(rr<labels.shape[0])&(cols>=0)&(cols<labels.shape[1])
            matches = np.zeros(len(points), bool)
            matches[valid] = labels[rr[valid], cols[valid]] == sample.cell_ID.to_numpy()[valid]
            scores.append(float(matches.mean()))
        if max(scores) < .8 or abs(scores[0]-scores[1]) < .5:
            raise ValueError(f'Ambiguous mask orientation in FOV {fov}: {scores}')
        self.flip_y[fov] = bool(np.argmax(scores))
        return dict(fov=fov, shape=list(labels.shape), flip_image_rows=self.flip_y[fov],
                    metadata_max_pixel_error=error, observed_max_pixel_error=observed_error,
                    observed_label_match_fraction=max(scores))

    def region(self, fov, cell_id):
        key = int(fov), int(cell_id)
        if key in self.regions:
            self.regions.move_to_end(key)
            return self.regions[key]
        rows, cols = self.pixels(*key)
        labels, boxes, _ = self._load_fov(fov)
        r0, r1, c0, c1 = boxes[int(cell_id)]
        crop = labels[r0:r1, c0:c1] == int(cell_id)
        padded = np.pad(crop, 1)
        centers, halfwidths = [], []
        # Exposed unit-pixel edges give the exact raster-region boundary.
        for dr, dc, rr_shift, cc_shift in [(0,-1,.5,0.),(0,1,.5,1.),(-1,0,0.,.5),(1,0,1.,.5)]:
            neighbor = padded[1+dr:1+dr+crop.shape[0], 1+dc:1+dc+crop.shape[1]]
            rr, cc = np.nonzero(crop & ~neighbor)
            local = self.local(fov, rr+r0+rr_shift, cc+c0+cc_shift)
            centers.append(self.global_xy(fov, local)*self.scale)
            halfwidths.append(np.tile([0., .5*self.scale] if dc else [.5*self.scale, 0.], (len(rr),1)))
        centers, halfwidths = np.vstack(centers), np.vstack(halfwidths)
        result = Region(*key, rows, cols, centers, halfwidths, cKDTree(centers))
        self.regions[key] = result
        while len(self.regions) > 128:
            self.regions.popitem(last=False)
        return result

    def distance(self, points_um, donor):
        """Exact distance to raster-region edges; vectorized adaptive KD queries."""
        local = points_um/self.scale - self.origin(donor.fov)
        inside = self.contains(donor.fov, donor.cell_id, local)
        result = np.zeros(len(points_um), dtype=float)
        pending = np.flatnonzero(~inside)
        k = min(8, len(donor.edge_centers))
        while len(pending):
            distances, indices = donor.tree.query(points_um[pending], k=k)
            distances, indices = np.asarray(distances).reshape(len(pending), k), np.asarray(indices).reshape(len(pending), k)
            delta = np.maximum(np.abs(points_um[pending,None,:] - donor.edge_centers[indices])
                               - donor.edge_halfwidths[indices], 0.)
            best = np.sqrt(np.square(delta).sum(axis=2)).min(axis=1)
            # Every edge has half-length scale/2. More distant midpoints cannot
            # improve best once this conservative lower bound is reached.
            done = (distances[:,-1] - self.scale/2 >= best-1e-12) | (k == len(donor.edge_centers))
            result[pending[done]] = best[done]
            pending = pending[~done]
            k = min(2*k, len(donor.edge_centers))
        return result

    def sample(self, recipient, n, rng, donor=None, decay_um=2.):
        if donor is None:
            indices = rng.integers(len(recipient.rows), size=n)
        else:
            centers = self.local(recipient.fov, recipient.rows+.5, recipient.cols+.5)
            distance = self.distance(self.global_xy(recipient.fov, centers)*self.scale, donor)
            weights = np.exp(-(distance-distance.min())/decay_um)
            indices = rng.choice(len(weights), size=n, p=weights/weights.sum())
        # Uniform offsets within each selected pixel; reflection preserves area.
        offsets = rng.random((n,2))
        local = self.local(recipient.fov, recipient.rows[indices]+offsets[:,1],
                           recipient.cols[indices]+offsets[:,0])
        return local, self.global_xy(recipient.fov, local)


def load_inputs(path, spatial_path):
    obj = ad.read_h5ad(path)
    if 'spatial_slice' not in obj.obs or obj.obs.spatial_slice.nunique() != 1:
        raise ValueError('Input must contain one held-out test slice')
    slice_name = str(obj.obs.spatial_slice.iloc[0])
    if slice_name not in TEST_SLICES or 'split' not in obj.obs or not obj.obs.split.astype(str).eq('test').all():
        raise ValueError('Transcript generation is restricted to the held-out SYNDRI-SIM test set')
    if not obj.obs_names.is_unique:
        raise ValueError('Nonunique simulated cell identifiers')
    native = integer_counts(obj.layers['original_counts'])
    added = integer_counts(obj.layers['synthetic_admixture_counts'])
    _same_counts(obj.X, native+added, 'Original + synthetic counts do not reconstruct X')
    with h5py.File(spatial_path) as handle:
        spatial_obs, spatial_var = ad.io.read_elem(handle['obs']), ad.io.read_elem(handle['var'])
    if not spatial_obs.index.is_unique:
        raise ValueError('Nonunique spatial observation identifiers')
    ids = obj.obs['matched_st_id'].astype(str) if 'matched_st_id' in obj.obs else obj.obs_names.astype(str)
    rows = spatial_obs.index.get_indexer(ids)
    if (rows < 0).any():
        raise ValueError('Simulated matched_st_id does not occur in the spatial AnnData')
    cells = spatial_obs.iloc[rows][['fov','cell_ID','CenterX_local_px','CenterY_local_px',
                                  'CenterX_global_px','CenterY_global_px']].copy()
    cells.index = obj.obs_names
    for key in ['fov', 'cell_ID']:
        values = pd.to_numeric(cells[key]).to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError(f'Invalid {key}')
        cells[key] = values.astype(np.int64)
    if cells[['fov','cell_ID']].duplicated().any() or (cells.cell_ID <= 0).any():
        raise ValueError('Simulated cells must have unique nonzero (fov, cell_ID) keys')
    if not np.allclose(cells[['CenterX_global_px','CenterY_global_px']],
                       obj.obs[['CenterX_global_px','CenterY_global_px']], atol=1e-6, rtol=0):
        raise ValueError('Simulated/spatial centroids differ')
    return obj, native, added, cells, spatial_var, slice_name


def donor_allocations(obj, native, added, recipients, seed):
    """Attribute fixed added counts to known neighbor pools by gene proportions.

    Complete pooled genes have exactly the original donor counts. For a partial
    pooled gene, conditional sampling without replacement uses the same gene-
    specific proportions and cannot allocate more than a donor supplied.
    """
    from Heimdall.admixture_sampling import build_neighbor_dict
    if 'neighbor_dict' in obj.uns:
        neighbors = obj.uns['neighbor_dict']
        origin = 'saved neighbor_dict'
    else:
        cfg = obj.uns.get('synthetic_admixture_config')
        if cfg is None:
            raise ValueError('No saved neighbor graph or synthetic_admixture_config; cannot recover source pool')
        base = ad.AnnData(native, obs=obj.obs[['cell_type','spatial_slice','CenterX_global_px','CenterY_global_px']].copy())
        base = build_neighbor_dict(base, k_neighbors=int(cfg['k_neighbors']),
            coord_cols=('CenterX_global_px','CenterY_global_px'), tissue_col='spatial_slice',
            cell_type_col='cell_type', min_source_counts=int(cfg['min_source_counts']),
            use_same_cell_type_neighbors=bool(cfg['use_same_cell_type_neighbors']))
        neighbors = base.uns['neighbor_dict']
        origin = 'reconstructed using original builder, native counts, and saved configuration'
    ids = obj.obs_names.astype(str)
    recipient_ids, donor_ids, gene_ids, counts, proportions = [], [], [], [], []
    for recipient in recipients:
        key = str(ids[recipient])
        source_ids = list(map(str, neighbors.get(key, [])))
        donors = ids.get_indexer(source_ids)
        if (donors < 0).any() or recipient in donors or len(set(donors)) != len(donors):
            raise ValueError('Invalid source neighbors')
        added_row = added[recipient]
        genes = added_row.indices
        if not len(genes):
            continue
        pool = native[donors][:, genes].toarray()
        totals = pool.sum(axis=0)
        amounts = added_row.data
        if (totals <= 0).any() or (amounts > totals).any():
            raise ValueError('Saved admixture is inconsistent with the known native neighbor gene pool')
        # The existing sampler adds entire pooled genes; enforce that fact when
        # reconstructing a removed graph rather than silently guessing donors.
        if origin.startswith('reconstructed') and not np.array_equal(totals, amounts):
            raise ValueError('Recovered graph does not exactly explain saved whole-gene admixture')
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(recipient), 731]))
        for j, gene in enumerate(genes):
            allocation = pool[:,j] if amounts[j] == totals[j] else rng.multivariate_hypergeometric(pool[:,j], int(amounts[j]))
            keep = np.flatnonzero(allocation)
            recipient_ids.extend([key]*len(keep))
            donor_ids.extend(ids[donors[keep]].tolist())
            gene_ids.extend([str(obj.var_names[gene])]*len(keep))
            counts.extend(allocation[keep].tolist())
            proportions.extend((pool[keep,j]/totals[j]).tolist())
    table = pd.DataFrame(dict(assigned_cell=recipient_ids, source_cell=donor_ids,
                              gene_id=gene_ids, count=np.asarray(counts, dtype=np.int64),
                              gene_specific_source_proportion=proportions))
    return table, origin


def saved_donor_allocations(path, obj, native, added, recipients):
    """Read the simulator's authoritative integer transfers without reassigning donors."""
    import pyarrow.parquet as pq
    columns = ['assigned_row', 'source_row', 'gene_index', 'count',
               'assigned_cell', 'source_cell', 'gene_id',
               'native_donor_count', 'donor_weight']
    source = pq.ParquetFile(path)
    if not set(columns).issubset(source.schema_arrow.names):
        raise ValueError('Saved donor allocations lack the weighted simulator identifiers/counts')
    ids, genes = obj.obs_names.astype(str).to_numpy(), obj.var_names.astype(str).to_numpy()
    inverse = np.full(obj.n_obs, -1, dtype=np.int64)
    inverse[recipients] = np.arange(len(recipients))
    recovered = sparse.csr_matrix((len(recipients), obj.n_vars), dtype=np.int64)
    parts = []
    for batch in source.iter_batches(batch_size=250000, columns=columns):
        frame = batch.to_pandas()
        q = frame.assigned_row.to_numpy(np.int64)
        if (q < 0).any() or (q >= obj.n_obs).any():
            raise AssertionError('Invalid saved recipient row')
        frame = frame.loc[inverse[q] >= 0].copy()
        if frame.empty:
            continue
        q, d, g = (frame[key].to_numpy(np.int64) for key in
                   ['assigned_row', 'source_row', 'gene_index'])
        count = frame['count'].to_numpy()
        if (d < 0).any() or (d >= obj.n_obs).any() or (g < 0).any() or (g >= obj.n_vars).any():
            raise AssertionError('Invalid saved donor/gene row')
        if (q == d).any() or not np.equal(count, np.rint(count)).all() or (count <= 0).any():
            raise AssertionError('Saved transfers must be positive integers from another cell')
        for column, expected in [('assigned_cell', ids[q]), ('source_cell', ids[d]), ('gene_id', genes[g])]:
            if not np.array_equal(frame[column].to_numpy(), expected):
                raise AssertionError(f'Saved allocation identifiers differ: {column}')
        if not np.array_equal(np.asarray(native[d, g]).ravel(), frame.native_donor_count):
            raise AssertionError('Saved donor native counts differ from the clean input')
        if np.asarray(native[q, g]).any():
            raise AssertionError('Saved donation touches a native-positive gene')
        if not np.isfinite(frame.donor_weight).all() or (frame.donor_weight < 0).any():
            raise AssertionError('Invalid saved donor weight')
        expectation = frame.native_donor_count.to_numpy() * frame.donor_weight.to_numpy()
        if ((count < np.floor(expectation)) | (count > np.ceil(expectation))).any():
            raise AssertionError('Saved integer contribution differs from weighted stochastic rounding')
        recovered += sparse.coo_matrix((count.astype(np.int64), (inverse[q], g)),
                                      shape=recovered.shape).tocsr()
        amounts = np.asarray(added[q, g]).ravel()
        if (amounts <= 0).any():
            raise AssertionError('Saved transfer has no corresponding added gene count')
        frame['gene_specific_source_proportion'] = count / amounts
        parts.append(frame)
    _same_counts(recovered, added[recipients], 'Saved donor allocations do not reconstruct added counts')
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {key: pd.Series(dtype='object' if key in ['assigned_cell', 'source_cell', 'gene_id'] else 'float64')
         for key in columns + ['gene_specific_source_proportion']})
    return table, 'saved donor-recipient-gene integer allocations; neighbor graph not reconstructed'


def gene_targets(obj, spatial_var, observed_targets):
    observed = set(observed_targets)
    mapping = obj.uns.get('gene_mapping:ensembl_to_symbol', {})
    symbols = []
    for gene in obj.var_names.astype(str):
        candidates = [gene]
        if gene in spatial_var.index and 'gene_symbol' in spatial_var:
            candidates.append(str(spatial_var.loc[gene, 'gene_symbol']))
        if gene in mapping:
            candidates.extend(np.atleast_1d(mapping[gene]).astype(str).tolist())
        matches = {name for candidate in candidates for name in
                   [candidate, candidate.replace('-', '.'), candidate.replace('.', '-')]
                   if name in observed}
        if len(matches) != 1:
            raise ValueError(f'{gene}: expected one original tx target, found {matches}; no genes are discarded')
        symbols.append(matches.pop())
    if len(set(symbols)) != len(symbols):
        raise ValueError('Multiple matrix genes map to the same tx target')
    return np.asarray(symbols, dtype=object)


def _frame(obj, cells, recipient, donor, genes, geometry, profile, symbols, rng, decay_um):
    region = geometry.region(int(cells.fov.iloc[recipient]), int(cells.cell_ID.iloc[recipient]))
    donor_region = None if donor == recipient else geometry.region(
        int(cells.fov.iloc[donor]), int(cells.cell_ID.iloc[donor]))
    local, global_xy = geometry.sample(region, len(genes), rng, donor_region, decay_um)
    planes = np.asarray(profile['z_planes'])
    probabilities = np.asarray(profile['z_counts'], dtype=float)
    z = rng.choice(planes, size=len(genes), p=probabilities/probabilities.sum())
    n = len(genes)
    # Dictionaries contain whole vector columns, never per-transcript records.
    data = {key: np.full(n, '', dtype=object) for key in profile['columns']}
    data.update(fov=np.full(n, region.fov), cell_ID=np.full(n, region.cell_id),
                x_global_px=global_xy[:,0], y_global_px=global_xy[:,1],
                x_local_px=local[:,0], y_local_px=local[:,1], z=z,
                target=symbols[genes], gene_id=obj.var_names.to_numpy()[genes],
                source_cell=np.full(n, str(obj.obs_names[donor]), dtype=object),
                assigned_cell=np.full(n, str(obj.obs_names[recipient]), dtype=object),
                is_admixed=np.full(n, donor != recipient),
                source_fov=np.full(n, int(cells.fov.iloc[donor])),
                source_cell_ID=np.full(n, int(cells.cell_ID.iloc[donor])),
                assigned_fov=np.full(n, region.fov), assigned_cell_ID=np.full(n, region.cell_id))
    if 'source_sc_id' in obj.obs:
        data['source_sc_id'] = np.full(n, str(obj.obs.source_sc_id.iloc[donor]), dtype=object)
        data['assigned_sc_id'] = np.full(n, str(obj.obs.source_sc_id.iloc[recipient]), dtype=object)
    # CellComp and any other unmodeled instrument fields remain empty.
    return pd.DataFrame(data)


_TRANSCRIPT_WORKER_CONTEXT = None


def _write_recipient_transcripts(stream, recipients, context):
    """Stream one bounded batch; random streams depend on cell IDs, not workers."""
    obj, native, cells, allocations, by_recipient, geometry, profile, symbols, seed, decay_um = context
    header = True
    for recipient in recipients:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(recipient), 101]))
        clean = native[recipient]
        genes = np.repeat(clean.indices, clean.data)
        rng.shuffle(genes)
        if len(genes):
            frame = _frame(obj,cells,recipient,recipient,genes,geometry,profile,symbols,rng,decay_um)
            frame.to_csv(stream, header=header, index=False)
            header = False
        indices = by_recipient.get(str(obj.obs_names[recipient]))
        if indices is not None:
            part = allocations.iloc[indices]
            for source, group in part.groupby('source_cell', sort=False):
                donor = int(obj.obs_names.get_loc(source))
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(recipient), donor, 211]))
                genes = np.repeat(obj.var_names.get_indexer(group.gene_id), group['count'].to_numpy(np.int64))
                rng.shuffle(genes)
                frame = _frame(obj,cells,recipient,donor,genes,geometry,profile,symbols,rng,decay_um)
                frame.to_csv(stream, header=header, index=False)
                header = False
    if header:
        empty = _frame(obj,cells,int(recipients[0]),int(recipients[0]),np.array([],dtype=int),
                       geometry,profile,symbols,np.random.default_rng(seed),decay_um)
        empty.to_csv(stream, index=False)


def _write_transcript_chunk(task):
    number, recipients, path = task
    if _TRANSCRIPT_WORKER_CONTEXT is None:
        raise RuntimeError('Transcript worker context was not initialized')
    with Path(path).open('x', newline='') as stream:
        _write_recipient_transcripts(stream, recipients, _TRANSCRIPT_WORKER_CONTEXT)
    return number, len(recipients)


def validate_transcripts(path, obj, native, added, recipients, cells, allocations,
                         geometry, profile, symbols, chunksize=250000):
    """Independently stream the written CSV and verify all seven invariants."""
    selected = obj.obs_names[recipients].astype(str)
    all_ids = obj.obs_names.astype(str)
    genes = obj.var_names.astype(str)
    shape = len(recipients), obj.n_vars
    total_counts, native_counts = sparse.csr_matrix(shape, dtype=np.int64), sparse.csr_matrix(shape, dtype=np.int64)
    allocation_index = pd.MultiIndex.from_frame(allocations[['assigned_cell','source_cell','gene_id']])
    if not allocation_index.is_unique:
        raise AssertionError('Duplicate donor allocation keys')
    allocation_counts = np.zeros(len(allocations), dtype=np.int64)
    n_transcripts, n_native, n_added, n_cross_fov = 0, 0, 0, 0
    max_transform_error = 0.
    with pd.read_csv(path, chunksize=chunksize, keep_default_na=False,
                     dtype={'is_admixed': bool}) as reader:
        for frame in reader:
            row = selected.get_indexer(frame.assigned_cell)
            col = genes.get_indexer(frame.gene_id)
            sources = all_ids.get_indexer(frame.source_cell)
            if (row < 0).any() or (col < 0).any() or (sources < 0).any():
                raise AssertionError('Unknown cell or gene ID in transcript table')
            if not np.array_equal(frame.target.to_numpy(), symbols[col]):
                raise AssertionError('Gene symbols and matrix identifiers differ')
            admixed = frame.is_admixed.to_numpy(dtype=bool)
            if not np.array_equal(admixed, frame.source_cell.ne(frame.assigned_cell).to_numpy()):
                raise AssertionError('is_admixed does not agree with source/assigned IDs')
            for prefix, positions in [('source', sources), ('assigned', recipients[row])]:
                if not np.array_equal(frame[prefix+'_fov'].to_numpy(), cells.fov.to_numpy()[positions]) or not np.array_equal(
                        frame[prefix+'_cell_ID'].to_numpy(), cells.cell_ID.to_numpy()[positions]):
                    raise AssertionError('Cell and FOV identifiers disagree')
            if not np.array_equal(frame.fov, frame.assigned_fov) or not np.array_equal(frame.cell_ID, frame.assigned_cell_ID):
                raise AssertionError('CosMx assignment columns disagree with ground truth')
            if not np.isin(frame.z, profile['z_planes']).all():
                raise AssertionError('Simulated z not in original discrete planes')
            if not np.isfinite(frame[['x_local_px','y_local_px','x_global_px','y_global_px']]).all().all():
                raise AssertionError('Nonfinite simulated coordinates')
            for (fov, cell_id), indices in frame.groupby(['fov','cell_ID'], sort=False).indices.items():
                local = frame.iloc[indices][['x_local_px','y_local_px']].to_numpy(float)
                xy = frame.iloc[indices][['x_global_px','y_global_px']].to_numpy(float)
                if not geometry.contains(int(fov), int(cell_id), local).all():
                    raise AssertionError(f'Transcript outside assigned segmentation: {(fov,cell_id)}')
                # Validate membership again through the inverse global transform.
                back = xy-geometry.origin(fov)
                error = float(np.abs(back-local).max())
                max_transform_error = max(max_transform_error, error)
                if error > 1e-6 or not geometry.contains(int(fov), int(cell_id), back).all():
                    raise AssertionError('Global coordinates do not map back to the assigned mask')
            total_counts += sparse.coo_matrix((np.ones(len(frame), dtype=np.int64),(row,col)), shape=shape).tocsr()
            clean = ~admixed
            native_counts += sparse.coo_matrix((np.ones(int(clean.sum()), dtype=np.int64),(row[clean],col[clean])), shape=shape).tocsr()
            if admixed.any():
                keys = pd.MultiIndex.from_frame(frame.loc[admixed,['assigned_cell','source_cell','gene_id']])
                index = allocation_index.get_indexer(keys)
                if (index < 0).any():
                    raise AssertionError('Transcript with an unknown donor-recipient-gene allocation')
                allocation_counts += np.bincount(index, minlength=len(allocations))
            n_transcripts += len(frame)
            n_native += int(clean.sum())
            n_added += int(admixed.sum())
            n_cross_fov += int((admixed & frame.source_fov.ne(frame.assigned_fov).to_numpy()).sum())
    _same_counts(total_counts, native[recipients]+added[recipients], 'Transcript table does not reconstruct admixed counts')
    _same_counts(native_counts, native[recipients], 'Native transcript counts do not reconstruct clean counts')
    _same_counts(total_counts-native_counts, added[recipients], 'Donated transcript counts do not reconstruct added counts')
    if not np.array_equal(allocation_counts, allocations['count'].to_numpy(np.int64)):
        raise AssertionError('Donor-recipient-gene counts differ from saved allocations')
    if n_transcripts != int((native[recipients]+added[recipients]).sum()):
        raise AssertionError('Transcripts were discarded')
    return dict(n_cells=len(recipients), n_transcripts=n_transcripts, n_native=n_native,
        n_admixed=n_added, n_cross_fov_admixed=n_cross_fov, max_transform_error_px=max_transform_error,
        checks=dict(admixed_matrix_exact=True, native_matrix_exact=True,
                    donor_recipient_gene_allocations_exact=True, all_xy_in_assigned_mask=True,
                    all_z_in_observed_planes=True, transformed_cell_fov_ids_consistent=True,
                    no_transcripts_discarded=True))


def simulate_transcripts(input_path, spatial_path, raw_dir, output_dir, fov=None,
                         max_cells=None, seed=20261113, decay_um=2., batch_cells=16,
                         donor_allocations_path=None, workers=1):
    if decay_um <= 0 or batch_cells <= 0 or workers <= 0 or (max_cells is not None and max_cells <= 0):
        raise ValueError('Decay length, batch size, worker count, and cell limit must be positive')
    input_path, raw_dir, output_dir = Path(input_path), Path(raw_dir), Path(output_dir)
    signature = _signature(input_path)
    obj, native, added, cells, spatial_var, slice_name = load_inputs(input_path, spatial_path)
    recipients = np.flatnonzero(cells.fov.eq(int(fov)).to_numpy()) if fov is not None else np.arange(obj.n_obs)
    if max_cells is not None:
        if len(recipients) < max_cells:
            raise ValueError(f'Only {len(recipients)} cells available, fewer than requested {max_cells}')
        selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 17]))
        recipients = np.sort(selection_rng.choice(recipients, max_cells, replace=False))
    if not len(recipients):
        raise ValueError('No selected recipients')
    # Spatial ordering improves mask-cache locality; per-cell seeds keep the
    # same coordinates independent of ordering and process count.
    recipients = recipients[np.lexsort((cells.cell_ID.to_numpy()[recipients], cells.fov.to_numpy()[recipients]))]
    name = f'fov_{int(fov):03d}' if fov is not None else 'all_fovs'
    if max_cells is not None:
        name += f'_n{max_cells}'
    output = output_dir / slice_name / name
    path = output / f'{slice_name}_tx_file.csv'
    if path.exists():
        raise FileExistsError(f'{path}: choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    cache = output_dir / 'cache' / slice_name
    cache.mkdir(parents=True, exist_ok=True)
    print('Attributing existing admixture to its source gene pools...', flush=True)
    allocation_signature = _signature(donor_allocations_path) if donor_allocations_path is not None else None
    if donor_allocations_path is not None:
        allocations, graph_origin = saved_donor_allocations(donor_allocations_path, obj, native, added, recipients)
    else:
        config = obj.uns.get('synthetic_admixture_config', {})
        if 'donor_type_weights' in config or config.get('admixture_target_mode') == 'pool_fraction':
            raise ValueError('Weighted simulations require --donor-allocations; donor identities cannot be reconstructed from an unweighted pool')
        allocations, graph_origin = donor_allocations(obj, native, added, recipients, seed)
    for prefix in ['source', 'assigned']:
        rows = obj.obs_names.get_indexer(allocations[prefix+'_cell'])
        allocations[prefix+'_fov'] = cells.fov.to_numpy()[rows]
        allocations[prefix+'_cell_ID'] = cells.cell_ID.to_numpy()[rows]
    allocations.to_parquet(output/'donor_allocations.parquet', index=False)
    print('Reading empirical slice-wide z-plane distribution...', flush=True)
    profile, examples = real_transcript_profile(raw_dir/f'{slice_name}_tx_file.csv', cache)
    symbols = gene_targets(obj, spatial_var, profile['targets'])
    geometry = MaskGeometry(raw_dir, slice_name, cache)
    donors = obj.obs_names.get_indexer(allocations.source_cell.unique())
    used = np.unique(np.r_[recipients, donors])
    calibration = [geometry.calibrate(int(number), cells, examples) for number in np.unique(cells.fov.iloc[used])]
    # Visit each FOV together to avoid repeatedly evicting its memory-mapped mask.
    for row in used[np.lexsort((cells.cell_ID.to_numpy()[used], cells.fov.to_numpy()[used]))]:
        geometry.pixels(int(cells.fov.iloc[row]), int(cells.cell_ID.iloc[row]))
    by_recipient = allocations.groupby('assigned_cell', sort=False).indices
    context = obj, native, cells, allocations, by_recipient, geometry, profile, symbols, seed, decay_um
    if workers == 1:
        with path.open('x', newline='') as stream:
            _write_recipient_transcripts(stream, recipients, context)
        print(f'Sampled {len(recipients)}/{len(recipients)} recipients', flush=True)
    else:
        import multiprocessing as mp
        import shutil
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if 'fork' not in mp.get_all_start_methods():
            raise ValueError('Multiple transcript workers require fork support; use --workers 1 here')
        # Fork shares immutable count/allocation arrays. Do not send full
        # AnnData objects through the multiprocessing task queue.
        global _TRANSCRIPT_WORKER_CONTEXT
        _TRANSCRIPT_WORKER_CONTEXT = context
        parts = output/'.transcript_parts'
        parts.mkdir()
        tasks = [(i, recipients[start:start+batch_cells], parts/f'{i:06d}.csv')
                 for i, start in enumerate(range(0, len(recipients), batch_cells))]
        print(f'Sampling with {workers} processes and {batch_cells} recipients per batch...', flush=True)
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
                futures = [pool.submit(_write_transcript_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    _, size = future.result()
                    completed += size
                    print(f'Sampled {completed}/{len(recipients)} recipients', flush=True)
        finally:
            _TRANSCRIPT_WORKER_CONTEXT = None
        print('Combining transcript batches in deterministic cell order...', flush=True)
        with path.open('xb') as destination:
            for number, _, source in tasks:
                with source.open('rb') as stream:
                    if number:
                        stream.readline()  # All fragments have the same header.
                    shutil.copyfileobj(stream, destination, length=8*1024**2)
                source.unlink()
        parts.rmdir()
    print('Validating the written transcript table...', flush=True)
    # Transfers were checked against the source sidecar before writing; avoid
    # holding a second copy of millions of allocation records during validation.
    report = validate_transcripts(path,obj,native,added,recipients,cells,allocations,
                                  geometry,profile,symbols)
    if signature != _signature(input_path):
        raise AssertionError('The source simulation file changed during coordinate generation')
    if allocation_signature is not None and allocation_signature != _signature(donor_allocations_path):
        raise AssertionError('The source donor allocations changed during coordinate generation')
    selected = ad.AnnData(native[recipients]+added[recipients], obs=obj.obs.iloc[recipients].copy(), var=obj.var.copy())
    selected.layers['original_counts'] = native[recipients]
    selected.layers['synthetic_admixture_counts'] = added[recipients]
    selected.var['target'] = symbols
    for key in ['fov','cell_ID','CenterX_local_px','CenterY_local_px']:
        selected.obs[key] = cells.iloc[recipients][key].to_numpy()
    selected.write_h5ad(output/'simulated_counts.h5ad', compression='gzip')
    cells.iloc[used].to_csv(output/'source_and_assigned_cells.csv', index_label='cell')
    _json(output/'validation.json', report)
    _json(output/'simulation.json', dict(input_signature=signature, spatial_signature=_signature(spatial_path),
        seed=seed, decay_length_um=decay_um, pixel_size_um=PIXEL_SIZE_UM,
        workers=workers, batch_cells=batch_cells,
        pixel_size_source=FORMAT_SOURCE, coordinate_units='legacy global/local pixels',
        z_convention='observed discrete image-plane indices, global slice distribution',
        z_planes=profile['z_planes'], z_counts=profile['z_counts'],
        graph_origin=graph_origin, donor_allocations_signature=allocation_signature,
        donor_attribution=('existing saved integer transfers; no donor resampling' if allocation_signature is not None
                           else 'gene-specific proportions in original neighbor pool; conditional sampling without replacement'),
        xy_method='uniform native pixels; exponential donor-region distance at recipient pixel centers; uniform subpixel offsets',
        unmodeled_columns_left_empty=[x for x in profile['columns'] if x not in
            ['fov','cell_ID','x_global_px','y_global_px','x_local_px','y_local_px','z','target']],
        fov_calibration=calibration, validation_passed=True))
    print(json.dumps(report, indent=2), flush=True)
    print(f'Transcripts: {path}', flush=True)
    return path, report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admixed-slice', type=Path, required=True)
    parser.add_argument('--spatial-adata', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True, help='The slice-Flat_files_and_images directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fov', type=int)
    parser.add_argument('--max-cells', type=int)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--decay-um', type=float, default=2.)
    parser.add_argument('--batch-cells', type=int, default=16)
    parser.add_argument('--donor-allocations', type=Path,
                        help='Use authoritative donor allocations from a weighted whole-gene simulation')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel sampling processes (fork); cell-specific random streams are unchanged')
    args = parser.parse_args(argv)
    simulate_transcripts(args.admixed_slice,args.spatial_adata,args.raw_dir,args.output_dir,
                         args.fov,args.max_cells,args.seed,args.decay_um,args.batch_cells,args.donor_allocations,args.workers)
