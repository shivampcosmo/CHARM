#!/usr/bin/env python
"""
build_training_shards.py
------------------------
Assemble per-GPU HDF5 training/validation shards from:
  - Per-sim HDF5 halo files produced by process_halos_quijote_v2.py
  - Pre-computed FastPM density / velocity full-field pickle files

Output:
  shard_dir/CHARM_train_shard_{r}.h5  for r in [0, n_shards)
  shard_dir/CHARM_val_shard.h5

Each shard contains n_subvols = (nsims_this_shard * nsubvol_per_ji) rows.
Rows are sub-volumes; each row has:
  /density/dm_cube  (ninp, D_pad, D_pad, D_pad)  — padded CNN input
  /density/dm_nsh   (nvox, ninp)                 — non-shifted voxel input
  /halos/{all arrays}  (nvox, ...)               — halo targets
  /metadata/sim_ids, subvol_ids                  — provenance

Usage — single training shard (Slurm array):
    python prep_data/build_training_shards.py \\
        --config run_configs/TRAIN_CHARM_JOINT.yaml --shard_rank 0

Usage — validation shard:
    python prep_data/build_training_shards.py \\
        --config run_configs/TRAIN_CHARM_JOINT.yaml --split val

Usage — Slurm array (shards 0-7 train, shard 8 val):
    sbatch --array=0-8 prep_data/run_build_shards.sh
"""

import argparse
import os
import pickle
import sys
import traceback

import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

# Make charm package importable when run from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))
from utils_data_prep_v2 import prep_halo_catalog, prep_density_fields  # noqa: E402
from config_loader      import load_config  # noqa: E402


# ── configuration helpers ─────────────────────────────────────────────────────

def derive_padding(cfg: dict) -> int:
    """n_pad = (nf - 1) // 2  *  n_cnn_layers"""
    sc = cfg['sim_settings']
    nf = int(sc['nf'])
    nc = sum(1 if lt == 'cnn' else 2 for lt in sc['layers_types'])
    return (nf - 1) // 2 * nc


def count_ninp(cfg: dict) -> int:
    """1 density channel + 3 velocity channels = 4 for z_all_FP=[0.5,'v_0.5']."""
    z_all_FP = cfg['sim_settings'].get('z_all_FP', [])
    n_density = sum(1 for z in z_all_FP if 'v' not in str(z))
    n_vel     = sum(1 for z in z_all_FP if 'v'     in str(z))
    return n_density + 3 * n_vel


# ── sub-volume extraction (vectorised, no Python loops) ───────────────────────

def subvols_unpadded(arr: np.ndarray, nb: int, nax: int) -> np.ndarray:
    """
    Split a (nb*nax, nb*nax, nb*nax[, ...]) field into (nb³, nax, nax, nax[, ...])
    sub-volumes using a single reshape + transpose (zero-copy view).

    Sub-volume flat index: jc = jx*(nb²) + jy*nb + jz
    """
    extra = arr.shape[3:]          # trailing dims beyond the 3 spatial axes
    arr2 = arr.reshape(nb, nax, nb, nax, nb, nax, *extra)
    # bring sub-volume indices (0,2,4) to front, spatial (1,3,5) after
    ndim_extra = len(extra)
    perm = (0, 2, 4, 1, 3, 5) + tuple(range(6, 6 + ndim_extra))
    return arr2.transpose(perm).reshape(nb**3, nax, nax, nax, *extra)


def subvols_padded(arr: np.ndarray, nb: int, nax: int, n_pad: int) -> np.ndarray:
    """
    Extract (nb³, nax+2*n_pad, nax+2*n_pad, nax+2*n_pad) padded sub-volumes
    from a ALREADY-PADDED array of shape (nb*nax+2*n_pad, ...).

    Uses as_strided to create a zero-copy VIEW — call .copy() on the result
    before any modification.
    """
    D_pad = nax + 2 * n_pad
    ndim  = arr.ndim
    s     = arr.strides

    if ndim == 3:
        # scalar field: (grid+2p, grid+2p, grid+2p)
        view = as_strided(
            arr,
            shape  =(nb, nb, nb, D_pad, D_pad, D_pad),
            strides=(nax*s[0], nax*s[1], nax*s[2], s[0], s[1], s[2]),
        )
        return view.reshape(nb**3, D_pad, D_pad, D_pad)

    elif ndim == 4 and arr.shape[0] != nb * nax + 2 * n_pad:
        # channel-first: (C, grid+2p, grid+2p, grid+2p)
        C = arr.shape[0]
        view = as_strided(
            arr,
            shape  =(nb, nb, nb, C, D_pad, D_pad, D_pad),
            strides=(nax*s[1], nax*s[2], nax*s[3], s[0], s[1], s[2], s[3]),
        )
        return view.reshape(nb**3, C, D_pad, D_pad, D_pad)

    else:
        raise ValueError(f"Unsupported array shape for padded subvol extraction: {arr.shape}")


# ── density / velocity loading ────────────────────────────────────────────────

def load_density_full(fastpm_dir: str, isim: int, grid: int, z_snap: str) -> np.ndarray:
    """Load full-box CIC density contrast  (grid, grid, grid)  float32."""
    fname = os.path.join(
        fastpm_dir, str(isim),
        f'density_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk',
    )
    with open(fname, 'rb') as f:
        arr = pickle.load(f)['density_cic_unpad_combined']
    return np.asarray(arr, dtype=np.float32)


def load_velocity_full(fastpm_dir: str, isim: int, grid: int, z_snap: str) -> np.ndarray:
    """Load full-box CIC velocity field  (3, grid, grid, grid)  float32  [km/s]."""
    fname = os.path.join(
        fastpm_dir, str(isim),
        f'velocity_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk',
    )
    with open(fname, 'rb') as f:
        arr = pickle.load(f)['velocity_cic_unpad_combined']
    return (np.asarray(arr) * 1000.0).astype(np.float32)   # stored /1000 → km/s


# ── halo loading + sub-volume splitting ──────────────────────────────────────

def load_halo_h5(halo_hdf5_dir: str, isim: int, redshift: float) -> dict:
    """Load the per-sim halo HDF5 produced by process_halos_quijote_v2.py."""
    fpath = os.path.join(halo_hdf5_dir, str(isim), f'halos_rockstar_200c_z{redshift}.h5')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Halo HDF5 not found: {fpath}")
    out = {}
    with h5py.File(fpath, 'r') as f:
        for k in ['N_halos', 'M_halos', 'pos_halos',
                  'c_halos_sim', 'c_halos_diff', 'v_halos_true', 'v_halos_diff']:
            out[k] = f[k][:]
    return out


def split_halos_to_subvols(halos: dict, nb: int, nax: int) -> dict:
    """
    Reshape every full-grid (128³[,...]) halo array into (512, 16, 16, 16[,...])
    sub-volumes.  No loops — all done via reshape + transpose.
    """
    return {k: subvols_unpadded(v, nb, nax) for k, v in halos.items()}


def compute_subvoxel_pos_from_offset(
    pos_offset_vox: np.ndarray,   # (nsubvols, nax, nax, nax, nMax_h, 3)  in [-0.5, 0.5]
    idx: np.ndarray,              # (nsubvols,)  flat sub-volume indices
    nb: int,
    nax: int,
) -> np.ndarray:
    """
    Positions stored by NGP_xyz_prop are already sub-voxel offsets from the
    ROUNDED voxel centre, in voxel units [-0.5, 0.5].  They are correct as-is
    for the model's position head.  This function just returns pos_offset_vox
    unchanged — the selection of sub-volumes is already applied by the caller.
    """
    return pos_offset_vox   # shape: (nsubvols_sel, nax, nax, nax, nMax_h, 3)


# ── HDF5 shard helpers ────────────────────────────────────────────────────────

def create_shard(
    path: str,
    n_total: int,
    nvox: int,
    ninp: int,
    D_pad: int,
    Nmax: int,
    cfg: dict,
) -> h5py.File:
    """
    Pre-allocate all datasets in the output shard HDF5.
    Returns the open (writable) file handle — caller must close it.
    """
    sc = cfg['sim_settings']
    dc = cfg['data_settings']
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    f = h5py.File(path, 'w')
    f.attrs.update({
        'n_total_subvols':  n_total,
        'Nmax':             Nmax,
        'lgMmin':           float(sc['lgMmin']),
        'lgMmax':           float(sc['lgMmax']),
        'vmin':             float(sc['vmin']),
        'vmax':             float(sc['vmax']),
        'cmin':             float(sc['cmin']),
        'cmax':             float(sc['cmax']),
        'ninp':             ninp,
        'D_pad':            D_pad,
        'nax_h':            nvox ** (1/3),   # store side length
        'nvox':             nvox,
        'nsims_per_batch':  int(cfg['train_settings']['nsims_per_batch']),
        'num_cosmo_params': int(sc['num_cosmo_params']),
    })

    N = n_total
    kw = dict(compression='lzf')
    chunk1 = (1,)

    dg = f.create_group('density')
    dg.create_dataset('dm_cube', shape=(N, ninp, D_pad, D_pad, D_pad),
                      dtype='float16', chunks=(1, ninp, D_pad, D_pad, D_pad), **kw)
    dg.create_dataset('dm_nsh',  shape=(N, nvox, ninp),
                      dtype='float16', chunks=(1, nvox, ninp), **kw)

    hg = f.create_group('halos')
    hg.create_dataset('N_halos',    shape=(N, nvox),        dtype='int16',   chunks=(1, nvox), **kw)
    hg.create_dataset('M_norm',     shape=(N, nvox, Nmax),  dtype='float16', chunks=(1, nvox, Nmax), **kw)
    hg.create_dataset('M1_norm',    shape=(N, nvox),        dtype='float16', chunks=(1, nvox), **kw)
    hg.create_dataset('Mdiff_norm', shape=(N, nvox, Nmax-1),dtype='float16', chunks=(1, nvox, Nmax-1), **kw)
    hg.create_dataset('mask_M1',    shape=(N, nvox),        dtype='float16', chunks=(1, nvox), **kw)
    hg.create_dataset('mask_Mdiff', shape=(N, nvox, Nmax-1),dtype='float16', chunks=(1, nvox, Nmax-1), **kw)
    hg.create_dataset('mask_halo',  shape=(N, nvox, Nmax),  dtype='float16', chunks=(1, nvox, Nmax), **kw)
    hg.create_dataset('v_norm',     shape=(N, nvox, Nmax*3),dtype='float16', chunks=(1, nvox, Nmax*3), **kw)
    hg.create_dataset('mask_vel',   shape=(N, nvox, Nmax*3),dtype='float16', chunks=(1, nvox, Nmax*3), **kw)
    hg.create_dataset('c_norm',     shape=(N, nvox, Nmax),  dtype='float16', chunks=(1, nvox, Nmax), **kw)
    hg.create_dataset('mask_conc',  shape=(N, nvox, Nmax),  dtype='float16', chunks=(1, nvox, Nmax), **kw)
    hg.create_dataset('pos_norm',   shape=(N, nvox, Nmax*3),dtype='float16', chunks=(1, nvox, Nmax*3), **kw)
    hg.create_dataset('mask_pos',   shape=(N, nvox, Nmax*3),dtype='float16', chunks=(1, nvox, Nmax*3), **kw)
    hg.create_dataset('cosmo',      shape=(N, nvox, 5),     dtype='float32', chunks=(1, nvox, 5), **kw)

    mg = f.create_group('metadata')
    # mu_all / sig_all written after first sim
    mg.create_dataset('sim_ids',    shape=(N,), dtype='int32',  chunks=chunk1)
    mg.create_dataset('subvol_ids', shape=(N,), dtype='int32',  chunks=chunk1)

    return f


def write_rows(f: h5py.File, row: int, batch: dict, sim_ids: list, subvol_ids: list) -> None:
    """Write one processed sim's sub-volumes (nsubvol_per_ji rows) into the shard."""
    ns = len(sim_ids)   # nsubvol_per_ji

    f['density/dm_cube'][row:row+ns]    = batch['dm_cube'].astype(np.float16)
    f['density/dm_nsh'][row:row+ns]     = batch['dm_nsh'].astype(np.float16)

    for k in ['N_halos', 'M_norm', 'M1_norm', 'Mdiff_norm',
              'mask_M1', 'mask_Mdiff', 'mask_halo',
              'v_norm', 'mask_vel', 'c_norm', 'mask_conc',
              'pos_norm', 'mask_pos', 'cosmo']:
        if k in batch:
            arr = batch[k]
            dtype = np.float32 if k == 'cosmo' else np.float16
            if k == 'N_halos':
                dtype = np.int16
            f[f'halos/{k}'][row:row+ns] = arr.astype(dtype)

    f['metadata/sim_ids'][row:row+ns]    = np.array(sim_ids,    dtype=np.int32)
    f['metadata/subvol_ids'][row:row+ns] = np.array(subvol_ids, dtype=np.int32)


# ── per-sim processing ────────────────────────────────────────────────────────

def process_sim_to_batch(
    isim: int,
    cfg: dict,
    nb: int,
    nax: int,
    n_pad: int,
    D_pad: int,
    nsubvol: int,
    Nmax: int,
    cosmo_vals: np.ndarray,
) -> tuple:
    """
    Load one simulation, extract nsubvol randomly selected sub-volumes,
    run normalisation, and return (batch_dict, sim_id_list, subvol_id_list).
    """
    sc = cfg['sim_settings']
    dc = cfg['data_settings']
    grid   = nb * nax          # 128
    nvox   = nax ** 3          # 4096
    z_snap = str(dc['z_snap'])
    z_dict = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0, -1: 99.0}
    redshift = z_dict[int(dc['snapnum'])]
    seed = int(dc.get('subvol_seed', 42)) * 1000 + isim

    # ── load density + velocity full fields ──────────────────────────────────
    rho = load_density_full(dc['fastpm_dir'], isim, grid, z_snap)  # (128,128,128)
    vel = load_velocity_full(dc['fastpm_dir'], isim, grid, z_snap) # (3,128,128,128)

    # ── derive padded + unpadded sub-volumes ─────────────────────────────────
    rho_pad  = np.pad(rho, n_pad, 'wrap')
    vel_pad  = np.pad(vel, [(0,0)] + [(n_pad,n_pad)]*3, 'wrap')

    # padded: views (no copy until indexed), shape (nb³, D_pad, D_pad, D_pad)
    rho_sub_pad = subvols_padded(rho_pad, nb, nax, n_pad)  # (512, D_pad, D_pad, D_pad)
    vel_sub_pad = subvols_padded(vel_pad, nb, nax, n_pad)  # (512, 3, D_pad, D_pad, D_pad)

    # unpadded: exact reshape/transpose
    rho_sub_unp = subvols_unpadded(rho, nb, nax)            # (512, 16, 16, 16)
    vel_sub_unp = subvols_unpadded(
        vel.transpose(1, 2, 3, 0), nb, nax                  # → (128,128,128,3)
    )  # (512, 16, 16, 16, 3) → then move channel back
    vel_sub_unp = np.moveaxis(vel_sub_unp, -1, 1)           # (512, 3, 16, 16, 16)

    # ── load halo data + split into sub-volumes ───────────────────────────────
    halos = load_halo_h5(dc['halo_hdf5_dir'], isim, redshift)
    # Each array: (128,128,128[,...]) → split → (512, 16, 16, 16[,...])
    halo_sub = split_halos_to_subvols(halos, nb, nax)

    # ── select nsubvol_per_ji sub-volumes (deterministic per isim) ───────────
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(nb ** 3, nsubvol, replace=False)

    # copy selected sub-volumes (materialise views from as_strided)
    rho_sel  = rho_sub_pad[idx].copy()    # (nsubvol, D_pad, D_pad, D_pad)
    vel_sel  = vel_sub_pad[idx].copy()    # (nsubvol, 3, D_pad, D_pad, D_pad)
    rho_nsh  = rho_sub_unp[idx].copy()   # (nsubvol, 16, 16, 16)
    vel_nsh  = vel_sub_unp[idx].copy()   # (nsubvol, 3, 16, 16, 16)

    # ── stack CNN input: density + velocity channels ──────────────────────────
    dm_cube = np.concatenate([
        rho_sel[:, np.newaxis],   # (nsubvol, 1, D_pad, D_pad, D_pad)
        vel_sel,                  # (nsubvol, 3, D_pad, D_pad, D_pad)
    ], axis=1)                    # → (nsubvol, 4, D_pad, D_pad, D_pad)

    # dm_nsh: (nsubvol, 4, 16, 16, 16) → (nsubvol, nvox, 4) via moveaxis+reshape
    dm_nsh_4d = np.concatenate([rho_nsh[:, np.newaxis], vel_nsh], axis=1)
    dm_nsh = np.moveaxis(dm_nsh_4d, 1, -1).reshape(nsubvol, nvox, -1)

    # ── halo data for selected sub-volumes ────────────────────────────────────
    # Each: (nsubvol, 16, 16, 16[, nMax_h[, 3]])
    N_sel      = halo_sub['N_halos'][idx]        # (nsubvol, 16, 16, 16)  int16
    M_sel      = halo_sub['M_halos'][idx]        # (nsubvol, 16, 16, 16, nMax_h)
    pos_sel    = halo_sub['pos_halos'][idx]      # (nsubvol, 16, 16, 16, nMax_h, 3)
    c_sim_sel = halo_sub['c_halos_sim'][idx]   # (nsubvol, 16, 16, 16, nMax_h)
    v_diff_sel = halo_sub['v_halos_diff'][idx]   # (nsubvol, 16, 16, 16, nMax_h, 3)

    # pos_sel is already in voxel-unit offsets [-0.5, 0.5] per component
    # (produced by process_halos_quijote_v2.py)

    # ── cosmology vector: broadcast over all sub-volumes and voxels ───────────
    cosmo_flat = np.broadcast_to(
        cosmo_vals[np.newaxis, np.newaxis, :],
        (nsubvol, nvox, 5),
    ).copy()

    # ── normalise, mask, build targets via prep_halo_catalog ─────────────────
    halo_dict = prep_halo_catalog(
        df_Mh    = M_sel.astype(np.float32),
        df_Nh    = N_sel.astype(np.int16),
        cosmo    = cosmo_flat.reshape(nsubvol, nax, nax, nax, 5),
        Mmin     = float(sc['lgMmin']),
        Mmax     = float(sc['lgMmax']),
        Nmax     = Nmax,
        rescale_sub = float(sc.get('rescale_sub', 0.0)),
        df_v     = v_diff_sel.astype(np.float32),
        df_c     = c_sim_sel.astype(np.float32),
        df_pos   = pos_sel.astype(np.float32),
        vmin=float(sc['vmin']), vmax=float(sc['vmax']),
        cmin=float(sc['cmin']), cmax=float(sc['cmax']),
    )

    dens_dict = prep_density_fields(
        df_d     = dm_cube[:, np.newaxis] if dm_cube.ndim == 4 else dm_cube.reshape(
            nsubvol, 1, *dm_cube.shape[1:]) if False else dm_cube[np.newaxis].squeeze(0),
        df_d_nsh = dm_nsh_4d,
    )
    # prep_density_fields expects (nsims, ninp, D, D, D) and (nsims, ninp, nax, nax, nax)
    # Our dm_cube is already (nsubvol, ninp, D_pad, D_pad, D_pad)
    dens_dict = {
        'dm_cube': dm_cube.astype(np.float16),
        'dm_nsh':  dm_nsh.astype(np.float16),
    }

    batch = {**halo_dict, **dens_dict}

    # add cosmo to halo group (prep_halo_catalog already embeds it, but store flat too)
    sim_ids    = [isim]    * nsubvol
    subvol_ids = idx.tolist()

    return batch, sim_ids, subvol_ids


# ── main shard builder ────────────────────────────────────────────────────────

def build_shard(
    shard_rank: int,
    sim_ids_all: list,
    cfg: dict,
    out_path: str,
) -> None:
    """Build one HDF5 shard for the given list of simulation IDs."""
    sc = cfg['sim_settings']
    dc = cfg['data_settings']

    nb          = int(sc['nb'])                             # 8
    nax         = int(sc['ns_h']) // nb                    # 16
    nvox        = nax ** 3                                  # 4096
    n_pad       = derive_padding(cfg)                       # 4
    D_pad       = nax + 2 * n_pad                          # 24
    ninp        = count_ninp(cfg)                          # 4
    Nmax        = int(sc['Nmax'])                          # 4
    nsubvol     = int(sc['nsubvol_per_ji'])                # 128
    n_total     = len(sim_ids_all) * nsubvol

    lh_cosmo    = np.loadtxt(dc['lh_cosmo_file'])          # (2000, 5)

    print(
        f'[shard {shard_rank}]  {len(sim_ids_all)} sims × {nsubvol} subvols '
        f'= {n_total} rows  →  {out_path}',
        flush=True,
    )

    with create_shard(out_path, n_total, nvox, ninp, D_pad, Nmax, cfg) as f:
        row = 0
        mu_sig_written = False

        for isim in tqdm(sim_ids_all, desc=f'shard {shard_rank}', leave=True):
            try:
                cosmo_vals = lh_cosmo[isim]
                batch, sids, svids = process_sim_to_batch(
                    isim, cfg, nb, nax, n_pad, D_pad, nsubvol, Nmax, cosmo_vals,
                )
                write_rows(f, row, batch, sids, svids)

                # write mu_all / sig_all once from the first successful sim
                if not mu_sig_written and 'mu_all' in batch:
                    mg = f['metadata']
                    mg.create_dataset('mu_all',  data=batch['mu_all'])
                    mg.create_dataset('sig_all', data=batch['sig_all'])
                    mu_sig_written = True

                row += nsubvol
                f.flush()   # flush incrementally so partial shards are readable

            except Exception:
                print(
                    f'[shard {shard_rank}] sim {isim} FAILED — skipping:\n'
                    + traceback.format_exc(),
                    flush=True,
                )

    print(f'[shard {shard_rank}]  done  ({row} rows written)', flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Build per-GPU HDF5 training/validation shards.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',     required=True, help='TRAIN_CHARM_JOINT.yaml path')
    p.add_argument('--shard_rank', type=int, default=None,
                   help='Which training shard to build (0..n_shards-1). '
                        'Overridden by SLURM_ARRAY_TASK_ID if not set.')
    p.add_argument('--split',      default='train', choices=['train', 'val'],
                   help='"train" builds a training shard; "val" builds the val shard. '
                        'Slurm task id >= n_shards triggers val automatically.')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    sc   = cfg['sim_settings']
    dc   = cfg['data_settings']

    n_shards       = int(dc['n_shards'])                   # 8
    nsims_train    = int(sc.get('nsims_train', 1800))
    nsims_val      = int(sc.get('nsims_val',   100))
    shard_dir      = dc['shard_dir']

    # Determine shard rank from CLI or Slurm
    rank = args.shard_rank
    if rank is None:
        rank = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # Determine split: Slurm task id == n_shards → val shard
    split = args.split
    if rank >= n_shards:
        split = 'val'

    if split == 'val':
        sim_ids  = list(range(nsims_train, nsims_train + nsims_val))
        out_path = os.path.join(shard_dir, 'CHARM_val_shard.h5')
        shard_id = 'val'
    else:
        # Interleaved assignment: shard r gets sims r, r+n_shards, r+2*n_shards, ...
        sim_ids  = list(range(rank, nsims_train, n_shards))
        out_path = os.path.join(shard_dir, f'CHARM_train_shard_{rank}.h5')
        shard_id = str(rank)

    build_shard(shard_id, sim_ids, cfg, out_path)


if __name__ == '__main__':
    main()
