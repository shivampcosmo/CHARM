#!/usr/bin/env python
"""
process_halos_quijote_v2.py
---------------------------
Per-simulation halo catalog processing for CHARM training data.

For each Quijote LH simulation:
  1. Reads Rockstar halo catalog (z=0.5 by default)
  2. Filters to parent halos and applies Mmin mass cut
  3. Computes concentration residual  delta_c = c_sim - c_Diemer19
  4. Loads pre-computed FastPM velocity field and interpolates at halo positions
     to obtain the velocity residual  dv = v_FastPM - v_true
  5. Paints all halo properties onto the 128³ NGP grid in a single pass
  6. Sorts halos within each voxel by descending mass
  7. Saves a compact per-sim HDF5 used by build_training_shards.py

Stored arrays (all on the 128³ halo grid, nMax_h slots per voxel):
  N_halos        (128, 128, 128)            int16
  M_halos        (128, 128, 128, nMax_h)    float32  — log10 mass [Msun/h], sorted desc
  pos_halos      (128, 128, 128, nMax_h, 3) float32  — sub-voxel offsets in voxel units
                                                        range [-0.5, 0.5] per component
  c_halos_sim    (128, 128, 128, nMax_h)    float32  — Rockstar concentration
  c_halos_diff   (128, 128, 128, nMax_h)    float32  — c_sim - c_Diemer19
  v_halos_true   (128, 128, 128, nMax_h, 3) float32  — true halo velocity [km/s]
  v_halos_diff   (128, 128, 128, nMax_h, 3) float32  — v_FastPM - v_true [km/s]

Usage — single sim:
    python prep_data/process_halos_quijote_v2.py \\
        --config run_configs/TRAIN_CHARM_JOINT.yaml --isim 0

Usage — Slurm array (one task per sim):
    sbatch --array=0-1999 prep_data/run_process_halos.sh
"""

import argparse
import os
import sys
import pickle
import traceback

import numpy as np
import h5py
import MAS_library as MASL
from scipy.interpolate import RegularGridInterpolator

# Colossus imports — set inside process_one_sim after cosmology is known
from colossus.cosmology import cosmology as colossus_cosmo
from colossus.halo import mass_so
from colossus.halo import concentration as colossus_conc

# Make charm package importable when run from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))
from config_loader import load_config  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────


def setup_ngp_funcs(ngp_funcs_dir: str):
    """Import the compiled ngp_funcs Cython extension."""
    abs_dir = os.path.abspath(ngp_funcs_dir)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)
    try:
        from ngp_funcs import NGP_xyz_prop  # noqa: PLC0415
        return NGP_xyz_prop
    except ImportError as e:
        raise ImportError(
            f"Could not import NGP_xyz_prop from {abs_dir}. "
            "Run 'python setup.py build_ext --inplace' in that directory."
        ) from e


def read_rockstar_200c(snapdir: str, snapnum: int):
    """
    Read a Rockstar *_pid.list catalog.

    Returns
    -------
    pos   : (N, 3) float32  — positions in Mpc/h
    vel   : (N, 3) float32  — peculiar velocities in km/s
    mass  : (N,)   float32  — M200c in Msun/h
    Rs    : (N,)   float32  — scale radius in kpc/h (comoving)
    """
    fpath = os.path.join(snapdir, f'out_{snapnum}_pid.list')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Rockstar catalog not found: {fpath}")

    # Parse header columns from the first comment line containing 'ID'
    header = None
    with open(fpath) as f:
        for line in f:
            if line.startswith('#') and 'ID' in line:
                header = line.lstrip('#').split()
                break
    if header is None:
        raise ValueError(f"Could not parse header from {fpath}")

    data = np.loadtxt(fpath, comments='#')

    # Filter to parent halos only (PID == -1)
    pid_col = data[:, -1]
    data = data[pid_col == -1]

    col = header.index
    pos  = data[:, col('X') : col('Z') + 1].astype(np.float32)   # Mpc/h
    vel  = data[:, col('VX') : col('VZ') + 1].astype(np.float32) # km/s
    mass = data[:, col('M200c')].astype(np.float32)               # Msun/h
    Rs   = data[:, col('Rs')].astype(np.float32)                  # kpc/h comoving

    return pos, vel, mass, Rs


def load_fastpm_velocity(fastpm_dir: str, isim: int, grid: int, z_snap: str) -> np.ndarray:
    """
    Load the pre-computed FastPM velocity field for simulation isim.

    Returns
    -------
    vel_field : (3, grid, grid, grid) float32, in km/s
    """
    fname = os.path.join(
        fastpm_dir, str(isim),
        f'velocity_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk'
    )
    if not os.path.exists(fname):
        raise FileNotFoundError(f"FastPM velocity not found: {fname}")
    with open(fname, 'rb') as f:
        vel_field = pickle.load(f)['velocity_cic_unpad_combined']  # stored in km/s / 1000
    return (np.asarray(vel_field) * 1000.0).astype(np.float32)    # convert to km/s


# ── core processing ───────────────────────────────────────────────────────────

def process_one_sim(isim: int, cfg: dict, NGP_xyz_prop, isfid: bool = False) -> None:
    """Process one simulation and write its halo HDF5 file."""
    sc = cfg['sim_settings']
    dc = cfg['data_settings']

    grid    = int(sc['ns_h'])                # 128
    BoxSize = float(dc['BoxSize'])           # 1000. Mpc/h
    nb      = int(sc['nb'])                  # 8  →  nax_h = 16
    snapnum = int(dc['snapnum'])             # 3  (z = 0.5)
    z_dict  = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0, -1: 99.0}
    redshift = z_dict[snapnum]
    Mmin_cut = float(dc['Mmin_cut'])         # 5e12 Msun/h
    nMax_h   = int(dc.get('nMax_h_raw', 10))
    z_snap   = str(dc['z_snap'])             # '0.5'

    # ── output path ──────────────────────────────────────────────────────────
    out_dir  = os.path.join(dc['halo_hdf5_dir'], str(isim))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'halos_rockstar_200c_z{redshift}.h5')

    # if os.path.exists(out_path):
    #     print(f'[sim {isim}] already exists, skipping.', flush=True)
    #     return

    # ── cosmology ────────────────────────────────────────────────────────────
    if isfid:
        cosmo_vals = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])
    else:
        cosmo_vals = np.loadtxt(dc['lh_cosmo_file'])[isim]
    Om0, Ob0, h0, ns_cosmo, sigma8 = cosmo_vals
    colossus_cosmo.setCosmology('myCosmo', **{
        'flat': True, 'H0': h0 * 100, 'Om0': Om0, 'Ob0': Ob0,
        'sigma8': sigma8, 'ns': ns_cosmo,
    })

    # ── read Rockstar catalog ─────────────────────────────────────────────────
    snapdir = os.path.join(dc['rockstar_dir'], str(isim))
    pos_h, vel_h, mass_h, Rs_h = read_rockstar_200c(snapdir, snapnum)

    # mass cut
    sel = mass_h > Mmin_cut
    pos_h, vel_h, mass_h, Rs_h = pos_h[sel], vel_h[sel], mass_h[sel], Rs_h[sel]
    lgMass = np.log10(mass_h).astype(np.float32)

    # ── concentrations ────────────────────────────────────────────────────────
    # R_200c: comoving Mpc/h from colossus, converted to kpc/h for consistency with Rs
    R200c_kpc = mass_so.M_to_R(mass_h, redshift, '200c') * (1.0 + redshift)
    c_sim  = (R200c_kpc / Rs_h).astype(np.float32)
    c_func = colossus_conc.concentration(
        mass_h, '200c', redshift, model='diemer19'
    ).astype(np.float32)
    c_diff = (c_sim - c_func).astype(np.float32)

    # ── FastPM velocity interpolation ─────────────────────────────────────────
    vel_field = load_fastpm_velocity(dc['fastpm_dir'], isim, grid, z_snap)
    # vel_field: (3, 128, 128, 128) in km/s

    xall   = np.linspace(0.0, BoxSize, grid + 1)
    xarray = 0.5 * (xall[1:] + xall[:-1])  # voxel centres in Mpc/h

    interp = [
        RegularGridInterpolator(
            (xarray, xarray, xarray), vel_field[i],
            method='linear', bounds_error=False, fill_value=None
        )
        for i in range(3)
    ]
    vel_pred = np.stack([interp[i](pos_h) for i in range(3)], axis=-1).astype(np.float32)
    vel_diff = (vel_pred - vel_h).astype(np.float32)   # v_FastPM - v_true  [km/s]

    # ── count halos per voxel (for N_halos grid) ─────────────────────────────
    N_halos_grid = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.NGP(np.float32(pos_h), N_halos_grid, BoxSize)

    # ── paint all properties onto grid: one NGP_xyz_prop call ─────────────────
    # prop layout: [lgM, c_sim, c_diff, vx_true, vy_true, vz_true, vx_diff, vy_diff, vz_diff]
    # NGP_xyz_prop stores positions at slots 0-2 (sub-voxel offset in Mpc/h from the
    # rounded voxel centre), properties from slot 3 onwards.
    props = np.concatenate([
        lgMass[:, None],    # 1
        c_sim[:, None],     # 2
        c_diff[:, None],    # 3
        vel_h,              # 4,5,6
        vel_diff,           # 7,8,9
    ], axis=-1).astype(np.float32)   # (N_halos, 9)

    # gridM: (128, 128, 128, nMax_h, 3 + 9 = 12)
    gridM = np.zeros((grid, grid, grid, nMax_h, 3 + props.shape[1]), dtype=np.float32)
    NGP_xyz_prop(np.float32(pos_h), props, gridM, BoxSize)

    # ── sort halos by descending mass (slot 3 = lgM) ─────────────────────────
    argsort_M = np.flip(np.argsort(gridM[..., 3], axis=-1), axis=-1)
    gridM_s   = np.take_along_axis(gridM, argsort_M[..., np.newaxis], axis=-2)

    # Extract individual arrays
    # Slots 0-2: sub-voxel offset from rounded voxel centre in Mpc/h.
    # Convert to voxel units [−0.5, 0.5] by multiplying by grid/BoxSize.
    pos_vox_offset = (gridM_s[..., 0:3] * (grid / BoxSize)).astype(np.float32)
    M_halos    = gridM_s[..., 3]    # log10 mass
    c_sim_g    = gridM_s[..., 4]    # simulated concentration
    c_diff_g   = gridM_s[..., 5]    # delta concentration
    v_true_g   = gridM_s[..., 6:9]  # true velocity (km/s)
    v_diff_g   = gridM_s[..., 9:12] # velocity residual (km/s)

    # ── write HDF5 ────────────────────────────────────────────────────────────
    N_halos_int = N_halos_grid.astype(np.int16)
    with h5py.File(out_path, 'w') as f:
        f.attrs['isim']      = isim
        f.attrs['grid']      = grid
        f.attrs['BoxSize']   = BoxSize
        f.attrs['redshift']  = redshift
        f.attrs['Mmin_cut']  = Mmin_cut
        f.attrs['nMax_h']    = nMax_h
        f.attrs['mass_type'] = 'rockstar_200c'
        f.attrs['n_halos_total'] = int(np.sum(N_halos_int))

        kw = dict(compression='lzf')
        f.create_dataset('N_halos',      data=N_halos_int,     **kw)
        f.create_dataset('M_halos',      data=M_halos,         **kw)
        f.create_dataset('pos_halos',    data=pos_vox_offset,  **kw)
        f.create_dataset('c_halos_sim',  data=c_sim_g,         **kw)
        f.create_dataset('c_halos_diff', data=c_diff_g,        **kw)
        f.create_dataset('v_halos_true', data=v_true_g,        **kw)
        f.create_dataset('v_halos_diff', data=v_diff_g,        **kw)

    print(
        f'[sim {isim:4d}]  N_halos_total={int(np.sum(N_halos_int)):6d}  '
        f'saved → {out_path}',
        flush=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Process Rockstar halo catalogs to per-sim HDF5 files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',      required=True,  help='Path to TRAIN_CHARM_JOINT.yaml')
    p.add_argument('--isim',        type=int, default=None,
                   help='Process a single simulation index.')
    p.add_argument('--isim_start',  type=int, default=None,
                   help='Start of a range of simulations to process.')
    p.add_argument('--isim_end',    type=int, default=None,
                   help='End (exclusive) of the range.')
    p.add_argument('--isfid',       action='store_true',
                   help='Process the fiducial simulation instead of LH.')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    dc   = cfg['data_settings']

    # Resolve ngp_funcs_dir relative to repo root (one level above prep_data/)
    repo_root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ngp_funcs_dir = os.path.join(repo_root, dc.get('ngp_funcs_dir', 'notebooks/testing'))
    NGP_xyz_prop = setup_ngp_funcs(ngp_funcs_dir)

    # Determine which simulations to process
    if args.isim is not None:
        sim_list = [args.isim]
    elif args.isim_start is not None and args.isim_end is not None:
        sim_list = list(range(args.isim_start, args.isim_end))
    else:
        # Slurm array mode: SLURM_ARRAY_TASK_ID selects the sim index
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        sim_list = [task_id]

    for isim in sim_list:
        try:
            process_one_sim(isim, cfg, NGP_xyz_prop, isfid=args.isfid)
        except Exception:
            print(f'[sim {isim}] FAILED:\n{traceback.format_exc()}', flush=True)


if __name__ == '__main__':
    main()
