"""
utils_data_prep_v2.py
---------------------
Data preparation utilities for the CHARM joint training pipeline.

Improvements over utils_data_prep_cosmo_vel_conc_peak.py:
- Vectorised mask construction (no Python loops)
- Unified halo property sorting
- Sub-voxel position offset support (zeros placeholder when data not yet available)
- Dedicated mask_conc and mask_pos keys (no reuse of mask_vel)
- HDF5 save / load with LZF compression and rank-aware chunking
- All preprocessing produces flat (nsims_total, ...) arrays that load_from_hdf5
  reshapes into the (n_outer_batches, nsims_per_batch, ...) format the model expects
"""

import os
import numpy as np
import h5py
from tqdm import tqdm


# ── Mask construction ─────────────────────────────────────────────────────────

def build_halo_masks(N_halos: np.ndarray, Nmax: int):
    """
    Build occupancy masks from per-voxel halo counts.

    Parameters
    ----------
    N_halos : (nsims, nvox) int array
    Nmax    : maximum halos per voxel

    Returns
    -------
    mask_M1    : (nsims, nvox)          float16 — 1 if voxel has ≥ 1 halo
    mask_Mdiff : (nsims, nvox, Nmax-1)  float16 — 1 for each additional halo slot
    mask_halo  : (nsims, nvox, Nmax)    float16 — 1 for each halo slot present
    """
    N = np.clip(N_halos, 0, Nmax)
    idx  = np.arange(Nmax,   dtype=np.int16)[None, None, :]
    idxd = np.arange(Nmax-1, dtype=np.int16)[None, None, :]
    mask_halo  = (idx  < N[..., None]).astype(np.float16)       # (nsims, nvox, Nmax)
    mask_Mdiff = (idxd < np.clip(N-1, 0, None)[..., None]).astype(np.float16)
    mask_M1    = mask_halo[..., 0]                               # (nsims, nvox)
    return mask_M1, mask_Mdiff, mask_halo


# ── Sorting ───────────────────────────────────────────────────────────────────

def sort_halos_by_mass(arr: np.ndarray, argsort: np.ndarray, Nmax: int,
                       n_trailing_dims: int = 0) -> np.ndarray:
    """
    Sort halo array by descending mass order and truncate to Nmax.

    Parameters
    ----------
    arr              : (..., Nmax_raw)  or  (..., Nmax_raw, n_trailing_dims)
    argsort          : (..., Nmax_raw) — descending mass argsort indices
    Nmax             : number of halos to keep
    n_trailing_dims  : 0 for scalar-per-halo, 1 for vector-per-halo (e.g. velocity)

    Returns
    -------
    sorted and truncated array: (..., Nmax)  or  (..., Nmax, n_trailing_dims)
    """
    if n_trailing_dims > 0:
        sort_idx = np.tile(argsort[..., np.newaxis], n_trailing_dims)
        return np.take_along_axis(arr, sort_idx, axis=-2)[..., :Nmax, :]
    return np.take_along_axis(arr, argsort, axis=-1)[..., :Nmax]


# ── Normalisation helpers ─────────────────────────────────────────────────────

def normalize_masses(M_sorted: np.ndarray, Mmin: float, Mmax: float,
                     rescale_sub: float = 0.0) -> np.ndarray:
    """Normalise log-masses to [rescale_sub, 1+rescale_sub]."""
    out = rescale_sub + (M_sorted - Mmin) / (Mmax - Mmin)
    return np.maximum(out, rescale_sub).astype(np.float16)


def normalize_velocities(v_sorted: np.ndarray, vmin: float,
                         vmax: float) -> np.ndarray:
    # return ((v_sorted - vmin) / (vmax - vmin)).astype(np.float16)
    return ((v_sorted) / (vmax - vmin)).astype(np.float16)


def normalize_concentrations(c_sorted: np.ndarray, cmin: float,
                              cmax: float) -> np.ndarray:
    return ((c_sorted - cmin) / (cmax - cmin)).astype(np.float16)


def compute_subvoxel_positions(pos_halos_sorted: np.ndarray,
                               nax_h: int) -> np.ndarray:
    """
    Compute sub-voxel position offsets from voxel centres.

    Parameters
    ----------
    pos_halos_sorted : (nsims, nvox, Nmax, 3)
        Absolute halo positions in voxel units [0, nax_h), already sorted by
        mass. Pass None to get zero placeholder.
    nax_h : int — voxels per side in one sub-cube

    Returns
    -------
    pos_norm : (nsims, nvox, Nmax, 3) float16 in [-0.5, 0.5]

    Notes
    -----
    Real position data requires halo catalogs that store 3D halo positions.
    Until those are available, pass pos_halos_sorted=None to get zeros,
    which trains the position head on a degenerate target. Re-run
    prep_training_data_hdf5.py once the catalogs are updated.
    """
    if pos_halos_sorted is None:
        return None   # caller fills zeros

    # Voxel centres: voxel j has centre at j + 0.5 along each axis.
    # Offset = absolute_pos - voxel_centre
    # nsims, nvox, Nmax, _ = pos_halos_sorted.shape
    # nax_h3 = nax_h ** 3

    # Build (nvox, 3) voxel-centre array
    # iz = np.arange(nax_h, dtype=np.float32)
    # cx, cy, cz = np.meshgrid(iz, iz, iz, indexing='ij')
    # centres = np.stack([cx.ravel() + 0.5,
    #                     cy.ravel() + 0.5,
    #                     cz.ravel() + 0.5], axis=-1)   # (nvox, 3)

    # pos_offset = pos_halos_sorted - centres[None, :, None, :]   # broadcast
    # return np.clip(pos_offset, -0.5, 0.5).astype(np.float16)
    return np.clip(pos_halos_sorted, -0.5, 0.5).astype(np.float16)


# ── Single-batch catalogue processing ────────────────────────────────────────

def prep_halo_catalog(
        df_Mh:   np.ndarray,            # (nsims, nx, ny, nz, Nmax_raw)
        df_Nh:   np.ndarray,            # (nsims, nx, ny, nz)
        cosmo:   np.ndarray,            # (nsims, nx, ny, nz, ncosmo)
        Mmin:    float,
        Mmax:    float,
        Nmax:    int,
        rescale_sub: float   = 0.0,
        df_v:    np.ndarray  = None,    # (nsims, nx, ny, nz, Nmax_raw, 3)
        df_c:    np.ndarray  = None,    # (nsims, nx, ny, nz, Nmax_raw)
        df_pos:  np.ndarray  = None,    # (nsims, nx, ny, nz, Nmax_raw, 3) absolute voxel coords
        vmin: float  = -1000.,
        vmax: float  =  1000.,
        cmin: float  = -8.,
        cmax: float  =  8.,
        sigv: float  = 0.05,
    ) -> dict:
    """
    Process one batch of halo catalogue data.

    Returns a dict with all normalised arrays shaped (nsims, nvox, ...).
    """
    nsims = df_Mh.shape[0]
    nvox  = df_Nh.shape[1] * df_Nh.shape[2] * df_Nh.shape[3]
    nax_h = df_Nh.shape[1]  # assumes cubic sub-cube

    # ── reshape to (nsims, nvox, ...) ──────────────────────────────────────
    N_halos = np.clip(
        df_Nh.reshape(nsims, nvox).astype(np.int16), 0, Nmax
    )
    M = df_Mh.reshape(nsims, nvox, df_Mh.shape[-1])

    # sort halos by descending mass
    argsort = np.flip(np.argsort(M, axis=-1), axis=-1)  # (nsims, nvox, Nmax_raw)
    M_sort  = sort_halos_by_mass(M, argsort, Nmax)       # (nsims, nvox, Nmax)

    M_norm  = normalize_masses(M_sort, Mmin, Mmax, rescale_sub)
    M1_norm = M_norm[..., 0]                              # (nsims, nvox)
    Mdiff_norm = (M_norm[..., :-1] - M_norm[..., 1:]).astype(np.float16)

    mask_M1, mask_Mdiff, mask_halo = build_halo_masks(N_halos, Nmax)

    # ── velocities ─────────────────────────────────────────────────────────
    v_norm_flat = None
    mask_vel_flat = None
    if df_v is not None:
        v = df_v.reshape(nsims, nvox, df_v.shape[-2], 3)
        v_sort = sort_halos_by_mass(np.clip(v, vmin, vmax), argsort, Nmax,
                                    n_trailing_dims=1)              # (nsims, nvox, Nmax, 3)
        v_norm      = normalize_velocities(v_sort, vmin, vmax)
        v_norm_flat = v_norm.reshape(nsims, nvox, Nmax * 3)         # flatten halo×component
        mask_vel_3d  = mask_halo[..., np.newaxis].repeat(3, axis=-1)
        mask_vel_flat = mask_vel_3d.reshape(nsims, nvox, Nmax * 3)

    # ── concentrations ─────────────────────────────────────────────────────
    c_norm = None
    if df_c is not None:
        c = df_c.reshape(nsims, nvox, df_c.shape[-1])
        c_sort = sort_halos_by_mass(np.clip(c, cmin, cmax), argsort, Nmax)
        c_norm = normalize_concentrations(c_sort, cmin, cmax)       # (nsims, nvox, Nmax)

    # ── sub-voxel positions ────────────────────────────────────────────────
    pos_norm_flat = None
    mask_pos_flat = None
    if df_pos is not None:
        pos = df_pos.reshape(nsims, nvox, df_pos.shape[-2], 3)
        pos_sort = sort_halos_by_mass(pos, argsort, Nmax, n_trailing_dims=1)
        pos_norm = compute_subvoxel_positions(pos_sort, nax_h)      # (nsims, nvox, Nmax, 3)
        pos_norm_flat  = pos_norm.reshape(nsims, nvox, Nmax * 3)
        mask_pos_flat  = mask_vel_flat if mask_vel_flat is not None else \
                         mask_halo[..., np.newaxis].repeat(3, axis=-1).reshape(nsims, nvox, Nmax * 3)
    else:
        # Zeros placeholder — position head will not learn until real data is provided
        pos_norm_flat = np.zeros((nsims, nvox, Nmax * 3), dtype=np.float16)
        mask_pos_flat = mask_halo[..., np.newaxis].repeat(3, axis=-1).reshape(nsims, nvox, Nmax * 3)

    # ── SumGauss target encoding for Ntot ─────────────────────────────────
    mu_all  = (np.arange(Nmax + 1) + 1).astype(np.float32)
    sig_all = (sigv * np.ones(Nmax + 1)).astype(np.float32)

    # ── cosmo: reshape to (nsims, nvox, ncosmo) ───────────────────────────
    cosmo_flat = cosmo.reshape(nsims, nvox, cosmo.shape[-1]).astype(np.float32)

    out = dict(
        N_halos   = N_halos,                                        # (nsims, nvox)
        M_norm    = M_norm,                                         # (nsims, nvox, Nmax)
        M1_norm   = M1_norm,                                        # (nsims, nvox)
        Mdiff_norm= (Mdiff_norm * mask_Mdiff).astype(np.float16),  # (nsims, nvox, Nmax-1)
        mask_M1   = mask_M1,                                        # (nsims, nvox)
        mask_Mdiff= mask_Mdiff,                                     # (nsims, nvox, Nmax-1)
        mask_halo = mask_halo,                                      # (nsims, nvox, Nmax)
        mu_all    = mu_all,                                         # (Nmax+1,)
        sig_all   = sig_all,                                        # (Nmax+1,)
        cosmo     = cosmo_flat,                                     # (nsims, nvox, ncosmo)
        pos_norm  = pos_norm_flat,                                  # (nsims, nvox, Nmax*3)
        mask_pos  = mask_pos_flat,                                  # (nsims, nvox, Nmax*3)
    )
    if df_v is not None:
        out['v_norm']    = v_norm_flat                              # (nsims, nvox, Nmax*3)
        out['mask_vel']  = mask_vel_flat
    if df_c is not None:
        out['c_norm']    = c_norm                                   # (nsims, nvox, Nmax)
        out['mask_conc'] = mask_halo.copy()                         # (nsims, nvox, Nmax) — separate key
    return out


def prep_density_fields(df_d: np.ndarray, df_d_nsh: np.ndarray) -> dict:
    """
    Process one batch of density fields.

    Parameters
    ----------
    df_d     : (nsims, ninp, D_pad, D_pad, D_pad) — padded CNN input (already padded)
    df_d_nsh : (nsims, ninp, nax_h, nax_h, nax_h) — non-shifted at halo grid resolution

    Returns
    -------
    dict with 'dm_cube' (nsims, ninp, D, D, D) and 'dm_nsh' (nsims, nvox, ninp)
    """
    nsims, ninp = df_d.shape[:2]
    nvox = df_d_nsh.shape[2] * df_d_nsh.shape[3] * df_d_nsh.shape[4]
    # nsh stored as (nsims, ninp, nx, ny, nz) → reshape to (nsims, nvox, ninp)
    dm_nsh = np.moveaxis(df_d_nsh, 1, -1).reshape(nsims, nvox, ninp)
    return dict(
        dm_cube = df_d.astype(np.float16),
        dm_nsh  = dm_nsh.astype(np.float16),
    )


# ── Full batched pipeline ─────────────────────────────────────────────────────

def prep_training_data_v2(
        df_d_all_inp:     np.ndarray,
        df_d_nsh_all_inp: np.ndarray,
        df_Mh_all_inp:    np.ndarray,
        df_Nh_inp:        np.ndarray,
        cosmo_val_all_inp: np.ndarray,
        Mmin:   float,
        Mmax:   float,
        Nmax:   int,
        nsims_per_chunk: int,
        rescale_sub: float = 0.0,
        df_v_inp:   np.ndarray = None,
        df_c_inp:   np.ndarray = None,
        df_pos_inp: np.ndarray = None,
        vmin: float = -1000., vmax: float = 1000.,
        cmin: float = -8.,    cmax: float = 8.,
        sigv: float = 0.05,
        verbose: bool = True,
) -> dict:
    """
    Process the full training dataset in chunks of nsims_per_chunk.

    Returns a dict where every array has (nsims_total, ...) as its first axis.
    This flat layout is what save_to_hdf5 and load_from_hdf5 expect.
    """
    nsims_total = df_Mh_all_inp.shape[0]
    n_chunks = (nsims_total + nsims_per_chunk - 1) // nsims_per_chunk
    chunks = range(n_chunks)
    if verbose:
        chunks = tqdm(chunks, desc='Preprocessing batches')

    accum = {}
    for jb in chunks:
        s = jb * nsims_per_chunk
        e = min(s + nsims_per_chunk, nsims_total)

        halo_dict = prep_halo_catalog(
            df_Mh    = df_Mh_all_inp[s:e],
            df_Nh    = df_Nh_inp[s:e],
            cosmo    = cosmo_val_all_inp[s:e],
            Mmin=Mmin, Mmax=Mmax, Nmax=Nmax,
            rescale_sub=rescale_sub,
            df_v     = df_v_inp[s:e]   if df_v_inp   is not None else None,
            df_c     = df_c_inp[s:e]   if df_c_inp   is not None else None,
            df_pos   = df_pos_inp[s:e] if df_pos_inp is not None else None,
            vmin=vmin, vmax=vmax, cmin=cmin, cmax=cmax, sigv=sigv,
        )
        dens_dict = prep_density_fields(df_d_all_inp[s:e], df_d_nsh_all_inp[s:e])

        batch = {**halo_dict, **dens_dict}

        # Accumulate
        for k, v in batch.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                accum.setdefault(k, []).append(v)
            else:
                accum[k] = v   # scalars / 1-D metadata (mu_all, sig_all, etc.)

    # Concatenate along the sim axis
    result = {}
    for k, v in accum.items():
        if isinstance(v, list):
            result[k] = np.concatenate(v, axis=0)
        else:
            result[k] = v
    return result


# ── HDF5 I/O ─────────────────────────────────────────────────────────────────

# Keys that live under halos/ group in HDF5
_HALO_KEYS = ['N_halos', 'M_norm', 'M1_norm', 'Mdiff_norm',
              'mask_M1', 'mask_Mdiff', 'mask_halo',
              'v_norm', 'mask_vel', 'c_norm', 'mask_conc',
              'pos_norm', 'mask_pos', 'cosmo']

# Keys that live under density/ group
_DENSITY_KEYS = ['dm_cube', 'dm_nsh']

# Scalar / 1-D metadata
_META_KEYS = ['mu_all', 'sig_all']


def save_to_hdf5(
        output_path:  str,
        data:         dict,
        Nmax:         int,
        Mmin:         float,
        Mmax:         float,
        vmin:         float,
        vmax:         float,
        cmin:         float,
        cmax:         float,
        nsims_per_chunk: int,
) -> None:
    """
    Write the flat training data dict to an HDF5 file.

    Chunks are sized to nsims_per_chunk along the sim axis so that each
    rank reads exactly one contiguous chunk.  LZF compression is used
    (fast, lossless, well-suited for float16 data).
    """
    nsims_total = data['N_halos'].shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # ── top-level metadata ─────────────────────────────────────────────
        f.attrs['nsims_total']    = nsims_total
        f.attrs['nsims_per_chunk']= nsims_per_chunk
        f.attrs['Nmax']   = Nmax
        f.attrs['Mmin']   = Mmin
        f.attrs['Mmax']   = Mmax
        f.attrs['vmin']   = vmin
        f.attrs['vmax']   = vmax
        f.attrs['cmin']   = cmin
        f.attrs['cmax']   = cmax
        if 'dm_cube' in data:
            f.attrs['ninp']  = data['dm_cube'].shape[1]
            f.attrs['D_pad'] = data['dm_cube'].shape[2]
        if 'dm_nsh' in data:
            f.attrs['nvox']  = data['dm_nsh'].shape[1]
            f.attrs['ninp']  = data['dm_nsh'].shape[2]

        # ── density group ──────────────────────────────────────────────────
        dg = f.create_group('density')
        for k in _DENSITY_KEYS:
            if k not in data:
                continue
            arr  = data[k]
            chnk = (nsims_per_chunk, *arr.shape[1:])
            dg.create_dataset(k, data=arr, chunks=chnk, compression='lzf')

        # ── halo / target group ────────────────────────────────────────────
        hg = f.create_group('halos')
        for k in _HALO_KEYS:
            if k not in data:
                continue
            arr  = data[k]
            chnk = (nsims_per_chunk, *arr.shape[1:])
            hg.create_dataset(k, data=arr, chunks=chnk, compression='lzf')

        # ── scalar metadata group ──────────────────────────────────────────
        mg = f.create_group('metadata')
        for k in _META_KEYS:
            if k in data:
                mg.create_dataset(k, data=data[k])

    total_gb = sum(
        v.nbytes for v in data.values() if isinstance(v, np.ndarray)
    ) / 1024**3
    print(f'Saved {nsims_total} sims to {output_path}  ({total_gb:.1f} GB uncompressed)')


def load_shard(h5_path: str, nsims_per_batch: int) -> dict:
    """
    Load a single per-GPU HDF5 shard produced by build_training_shards.py.

    The shard contains n_total_subvols rows.  This function reshapes them into
    (n_outer_batches, nsims_per_batch * nvox, ...) ready for CHARM_Model.forward,
    exactly as load_from_hdf5 does — but without any rank-slicing (each GPU
    opens its own dedicated shard file).

    Parameters
    ----------
    h5_path         : path to CHARM_train_shard_r.h5 or CHARM_val_shard.h5
    nsims_per_batch : sub-volumes per outer batch (must divide n_total_subvols)

    Returns
    -------
    dict of numpy arrays (CPU, not yet pinned)
    """
    with h5py.File(h5_path, 'r', swmr=True) as f:
        n_total = int(f.attrs['n_total_subvols'])

        if n_total % nsims_per_batch != 0:
            valid = [d for d in range(1, n_total + 1) if n_total % d == 0]
            raise ValueError(
                f'n_total_subvols ({n_total}) in {h5_path} is not divisible by '
                f'nsims_per_batch ({nsims_per_batch}). '
                f'Set nsims_per_batch to one of: {valid}'
            )
        n_outer = n_total // nsims_per_batch

        out = {}
        dg = f['density']
        for k in _DENSITY_KEYS:
            if k not in dg:
                continue
            arr = dg[k][:]                          # (n_total, ...)
            if k == 'dm_cube':
                out[k] = arr.reshape(n_outer, nsims_per_batch, *arr.shape[1:])
            else:
                out[k] = arr.reshape(n_outer,
                                     nsims_per_batch * arr.shape[1],
                                     *arr.shape[2:])

        hg = f['halos']
        for k in _HALO_KEYS:
            if k not in hg:
                continue
            arr = hg[k][:]                          # (n_total, nvox, ...)
            out[k] = arr.reshape(n_outer,
                                 nsims_per_batch * arr.shape[1],
                                 *arr.shape[2:])

        mg = f['metadata']
        for k in _META_KEYS:
            if k in mg:
                out[k] = mg[k][:]

        out['n_outer']         = n_outer
        out['nsims_per_batch'] = nsims_per_batch
        out['n_total_subvols'] = n_total

    return out


def load_from_hdf5(
        h5_path:         str,
        rank:            int,
        world_size:      int,
        nsims_per_batch: int,
) -> dict:
    """
    Load this rank's slice of the training data from HDF5.

    Each rank receives a contiguous block of (nsims_total // world_size) sims.
    Arrays are reshaped to (n_outer_batches, nsims_per_batch, ...) ready for
    CHARM_Model.forward, except dm_cube which keeps its spatial dimensions as
    (n_outer, nsims_per_batch, ninp, D, D, D).

    Parameters
    ----------
    h5_path         : path to the HDF5 file created by save_to_hdf5
    rank            : global DDP rank
    world_size      : total number of DDP ranks
    nsims_per_batch : number of sims per outer batch (must divide sims_per_rank)

    Returns
    -------
    dict of numpy arrays (CPU, not yet pinned)
    """
    with h5py.File(h5_path, 'r', swmr=True) as f:
        nsims_total = int(f.attrs['nsims_total'])

        if nsims_total % world_size != 0:
            raise ValueError(
                f'nsims_total ({nsims_total}) must be divisible by '
                f'world_size ({world_size}).  '
                f'Re-run prep_training_data_hdf5.py with a compatible --world_size.'
            )
        sims_per_rank = nsims_total // world_size
        if sims_per_rank % nsims_per_batch != 0:
            valid = [d for d in range(1, sims_per_rank + 1) if sims_per_rank % d == 0]
            raise ValueError(
                f'sims_per_rank ({sims_per_rank}) must be divisible by '
                f'nsims_per_batch ({nsims_per_batch}). '
                f'Set nsims_per_batch to one of: {valid}'
            )
        n_outer = sims_per_rank // nsims_per_batch
        rs = rank * sims_per_rank
        re = rs + sims_per_rank

        out = {}
        for k in _DENSITY_KEYS:
            if k not in f['density']:
                continue
            arr = f['density'][k][rs:re]                    # (sims_per_rank, ...)
            if k == 'dm_cube':
                # keep spatial dims: (n_outer, nsims_per_batch, ninp, D, D, D)
                out[k] = arr.reshape(n_outer, nsims_per_batch, *arr.shape[1:])
            else:
                # (n_outer, nsims_per_batch * nvox, ninp)
                out[k] = arr.reshape(n_outer, nsims_per_batch * arr.shape[1],
                                     *arr.shape[2:])

        for k in _HALO_KEYS:
            if k not in f['halos']:
                continue
            arr = f['halos'][k][rs:re]                      # (sims_per_rank, nvox, ...)
            out[k] = arr.reshape(n_outer,
                                 nsims_per_batch * arr.shape[1],
                                 *arr.shape[2:])

        mg = f['metadata']
        for k in _META_KEYS:
            if k in mg:
                out[k] = mg[k][:]

        out['n_outer']        = n_outer
        out['nsims_per_batch']= nsims_per_batch
        out['sims_per_rank']  = sims_per_rank

    return out
