#!/usr/bin/env python
"""
run_inference_v2.py
-------------------
Inference script for the jointly-trained CHARM model.

Loads a checkpoint, processes the DM density+velocity field for a single
simulation, runs model.sample(), reconstructs the full 1 Gpc/h mock halo
catalog, and saves it to disk as an .npz file.

Output arrays (all halos in the mock catalog):
    pos_mock   (N, 3)  float32  — 3D positions in Mpc/h, periodic BC applied
    lgM_mock   (N,)    float32  — log10 halo mass [Msun/h]
    vel_mock   (N, 3)  float32  — true velocity in km/s (v_DM - v_diff_pred)
    conc_mock  (N,)    float32  — concentration (raw c_sim convention)

Usage:
    cd /mnt/ceph/users/spandey/CHARM_v2/CHARM

    python charm/inferers/run_inference_v2.py \\
        --config run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml \\
        --sim_id 0

    # Override checkpoint explicitly:
    python charm/inferers/run_inference_v2.py \\
        --config run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml \\
        --sim_id 0 \\
        --checkpoint ../model_checkpoints/CHARM_JOINT_trial_Mmin1e14/charm_joint_best_val.pth

    # Override output directory:
    python charm/inferers/run_inference_v2.py \\
        --config run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml \\
        --sim_id 0 \\
        --output_dir ../inference_results/trial_Mmin1e14
"""

import argparse
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import RegularGridInterpolator

# ── make charm package importable ────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Insert repo root so 'from charm.utils import ...' works as an absolute import,
# and insert charm/ so direct 'import config_loader' works too.
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config
from run_charm_joint_v2vel_ddp import build_model
from inferers.calibrate_binary_prior import predict as _calib_predict


def estimate_target_prior_from_cosmology(cosmo_vec, calibrator_path: str
                                         ) -> tuple[float, dict]:
    """
    Predict the inference-time per-voxel occupancy fraction π_target
    from cosmology alone, using the calibrator built by
    calibrate_binary_prior.py.

    The calibrator predicts log10(N_occ_voxels) — the number of grid
    voxels that contain ≥1 halo.  Dividing by ns_h³ gives the correct
    π_target for the Bayesian binary-prior correction.

    Returns (pi_target, info_dict).  info_dict carries the predicted
    log10(N_occ_voxels), expected occupied-voxel count, and ns_h³ —
    purely for logging.
    """
    cal = np.load(calibrator_path, allow_pickle=False)
    degree = int(cal['poly_degree']) if 'poly_degree' in cal else 2
    log_n_pred = _calib_predict(
        np.asarray(cosmo_vec, dtype=np.float64),
        coeffs        = cal['coeffs'],
        feature_means = cal['feature_means'],
        feature_stds  = cal['feature_stds'],
        degree        = degree,
    )
    n_occ_pred = float(10.0 ** log_n_pred)   # N_occ_voxels (voxels with ≥1 halo)
    ns_h       = int(cal['ns_h'])
    ns_cube    = ns_h ** 3
    pi         = n_occ_pred / ns_cube         # = N_occ_voxels / ns_h³  (correct π_target)
    return pi, {
        'log10_N_occ_pred':  log_n_pred,
        'N_occ_pred':        n_occ_pred,
        'ns_h':              ns_h,
        'ns_cube':           ns_cube,
        'calibrator_r2':     float(cal['r2']),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML config (e.g. run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml)')
    p.add_argument('--sim_id', type=int, required=True,
                   help='Simulation ID to run inference on.')
    p.add_argument('--checkpoint', default=None,
                   help='Path to .pth checkpoint. Defaults to '
                        '<checkpoint_dir>/charm_joint_best_val.pth.')
    p.add_argument('--output_dir', default=None,
                   help='Directory for output .npz catalog. Defaults to '
                        '<checkpoint_dir>/inference/')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Torch device.')
    p.add_argument('--binary_target_prior', type=float, default=None,
                   help='True voxel occupancy fraction at this Mmin/cosmology. '
                        'Required when the binary head was trained with '
                        'subsample / alpha modes (π_train=0.5); the trained '
                        'pw_occ is renormalised to π_target via Bayes. '
                        'If left unset, it is auto-estimated from cosmology '
                        'using the calibrator file (see '
                        '--binary_prior_calibrator). Pass an explicit value '
                        'to override the auto-estimate.')
    p.add_argument('--binary_prior_calibrator', default=None,
                   help='Path to a .npz file produced by '
                        'calibrate_binary_prior.py mapping cosmology → '
                        'expected total halo count. Defaults to '
                        '<checkpoint_dir>/binary_prior_calibrator.npz. '
                        'Used to auto-estimate π_target from cosmology when '
                        '--binary_target_prior is not given.')
    p.add_argument('--binary_train_prior', type=float, default=None,
                   help='Override the trained-prior on the loaded model. '
                        'Use this when running inference on a checkpoint '
                        'that was trained with a different binary_loss_mode '
                        'than the config currently specifies (e.g. loading '
                        'a subsample checkpoint while the config is set to '
                        'focal). 0.5 for subsample / alpha; null for none '
                        '/ focal.')
    return p.parse_args()


# ── sub-volume helpers (same as build_training_shards.py) ─────────────────────

def subvols_unpadded(arr: np.ndarray, nb: int, nax: int) -> np.ndarray:
    """Split (nb*nax, nb*nax, nb*nax[,...]) → (nb³, nax, nax, nax[,...])."""
    extra = arr.shape[3:]
    arr2  = arr.reshape(nb, nax, nb, nax, nb, nax, *extra)
    ndim_extra = len(extra)
    perm = (0, 2, 4, 1, 3, 5) + tuple(range(6, 6 + ndim_extra))
    return arr2.transpose(perm).reshape(nb ** 3, nax, nax, nax, *extra)


def subvols_padded(arr: np.ndarray, nb: int, nax: int, n_pad: int) -> np.ndarray:
    """Extract (nb³, nax+2*n_pad, ...) padded sub-cubes from a pre-padded array."""
    D_pad = nax + 2 * n_pad
    s = arr.strides
    if arr.ndim == 3:
        view = as_strided(
            arr,
            shape  =(nb, nb, nb, D_pad, D_pad, D_pad),
            strides=(nax*s[0], nax*s[1], nax*s[2], s[0], s[1], s[2]),
        )
        return view.reshape(nb**3, D_pad, D_pad, D_pad)
    elif arr.ndim == 4:
        C = arr.shape[0]
        view = as_strided(
            arr,
            shape  =(nb, nb, nb, C, D_pad, D_pad, D_pad),
            strides=(nax*s[1], nax*s[2], nax*s[3], s[0], s[1], s[2], s[3]),
        )
        return view.reshape(nb**3, C, D_pad, D_pad, D_pad)
    else:
        raise ValueError(f"Unsupported shape for padded subvol extraction: {arr.shape}")


# ── data loading ──────────────────────────────────────────────────────────────

def load_fastpm(fastpm_dir: str, isim: int, grid: int, z_snap: str):
    """Return density (grid³) float32 and velocity (3, grid³) float32 in km/s."""
    base = os.path.join(fastpm_dir, str(isim),
                        f'density_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk')
    with open(base, 'rb') as fh:
        rho = pickle.load(fh)['density_cic_unpad_combined'].astype(np.float32)

    vbase = os.path.join(fastpm_dir, str(isim),
                         f'velocity_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk')
    with open(vbase, 'rb') as fh:
        vel_raw = pickle.load(fh)['velocity_cic_unpad_combined']
    vel = (np.asarray(vel_raw) * 1000.0).astype(np.float32)  # stored ÷1000 → km/s

    return rho, vel


def load_cosmology(lh_cosmo_file: str, isim: int) -> np.ndarray:
    """Load 5-parameter cosmology [Omega_m, Omega_b, h, n_s, sigma_8] for isim."""
    cosmo_all = np.loadtxt(lh_cosmo_file)
    return cosmo_all[isim].astype(np.float32)


# ── build CNN input tensors ───────────────────────────────────────────────────

def build_cond_tensors(rho, vel, cosmo_vals, nb, nax, n_pad,
                       rho_pad=None, vel_pad=None):
    """
    Build the three conditioning tensors for CHARM_Model.sample():

        cond_x      (1, nsubs, ninp, D_pad, D_pad, D_pad)
        cond_x_nsh  (1, nsubs*nvox, ninp)
        cond_cosmo  (1, nsubs*nvox, ncosmo)

    Parameters
    ----------
    rho        : (grid, grid, grid)  float32  CIC overdensity
    vel        : (3, grid, grid, grid)  float32  velocity km/s
    cosmo_vals : (5,)  cosmology parameters
    nb         : int — sub-cubes per side (8)
    nax        : int — voxels per sub-cube side (16)
    n_pad      : int — padding voxels per side (4)
    rho_pad    : (grid+2*n_pad, ...) float32  pre-padded density (optional).
                 If provided, the internal wrap-padding step is skipped so that
                 callers tiling a larger volume can supply correct cross-chunk
                 context rather than letting each chunk wrap its own boundaries.
    vel_pad    : (3, grid+2*n_pad, ...) float32  pre-padded velocity (optional).
    """
    nsubs = nb ** 3          # 512
    nvox  = nax ** 3         # 4096
    D_pad = nax + 2 * n_pad  # 24

    # ── padded sub-cubes for CNN ──────────────────────────────────────────────
    if rho_pad is None:
        rho_pad = np.pad(rho, n_pad, 'wrap')                         # (136,136,136)
    if vel_pad is None:
        vel_pad = np.pad(vel, [(0,0)] + [(n_pad,n_pad)]*3, 'wrap')  # (3,136,136,136)

    rho_sub = subvols_padded(rho_pad, nb, nax, n_pad).copy()        # (512, D_pad, D_pad, D_pad)
    vel_sub = subvols_padded(vel_pad, nb, nax, n_pad).copy()        # (512, 3, D_pad, D_pad, D_pad)

    dm_cube = np.concatenate(
        [rho_sub[:, np.newaxis], vel_sub], axis=1
    ).astype(np.float32)                                             # (512, 4, D_pad, D_pad, D_pad)

    # ── unpadded local features for non-shifted conditioning ─────────────────
    rho_unp  = subvols_unpadded(rho, nb, nax)                       # (512, 16, 16, 16)
    vel_unp  = subvols_unpadded(
        vel.transpose(1, 2, 3, 0), nb, nax                          # → (128,128,128,3)
    )                                                                # (512, 16, 16, 16, 3)
    vel_unp  = np.moveaxis(vel_unp, -1, 1)                          # (512, 3, 16, 16, 16)

    dm_nsh_4d = np.concatenate(
        [rho_unp[:, np.newaxis], vel_unp], axis=1
    ).astype(np.float32)                                             # (512, 4, 16, 16, 16)
    dm_nsh = np.moveaxis(dm_nsh_4d, 1, -1).reshape(nsubs, nvox, -1) # (512, 4096, 4)

    # ── cosmology: broadcast over all voxels ──────────────────────────────────
    ncosmo = len(cosmo_vals)
    cosmo_broadcast = np.broadcast_to(
        cosmo_vals[np.newaxis, :], (nsubs * nvox, ncosmo)
    ).copy().astype(np.float32)                                      # (2097152, 5)

    # Add batch dimension (nbatches=1 outer batch)
    cond_x     = torch.from_numpy(dm_cube[np.newaxis])              # (1, 512, 4, 24, 24, 24)
    cond_x_nsh = torch.from_numpy(
        dm_nsh.reshape(1, nsubs * nvox, -1)
    )                                                                # (1, 2097152, 4)
    cond_cosmo = torch.from_numpy(
        cosmo_broadcast[np.newaxis]
    )                                                                # (1, 2097152, 5)

    return cond_x, cond_x_nsh, cond_cosmo


# ── catalog reconstruction ────────────────────────────────────────────────────

def flat_idx_to_global_voxel(nb, nax):
    """
    Pre-compute global (ix, iy, iz) voxel indices for all nb³ × nax³ flat entries.

    Returns three (nb³ × nax³,) int arrays.
    The flat ordering is: sub-cube (jx*nb² + jy*nb + jz) × voxel (vx*nax² + vy*nax + vz).
    """
    nsubs = nb ** 3
    nvox  = nax ** 3
    total = nsubs * nvox

    isub = np.arange(total) // nvox
    ivox = np.arange(total) % nvox

    jx = isub // (nb * nb)
    jy = (isub % (nb * nb)) // nb
    jz = isub % nb

    vx = ivox // (nax * nax)
    vy = (ivox % (nax * nax)) // nax
    vz = ivox % nax

    return (jx * nax + vx).astype(np.int32), \
           (jy * nax + vy).astype(np.int32), \
           (jz * nax + vz).astype(np.int32)


def build_dm_velocity_interpolators(vel: np.ndarray, BoxSize: float):
    """
    Build RegularGridInterpolator for each velocity component.
    vel : (3, ns, ns, ns) float32 in km/s
    Returns list of 3 callables: [interp_vx, interp_vy, interp_vz]
    """
    ns = vel.shape[1]
    cell = BoxSize / ns
    # Grid centres (not edges)
    coords = np.linspace(0.5 * cell, BoxSize - 0.5 * cell, ns, dtype=np.float32)
    interps = []
    for ci in range(3):
        interps.append(
            RegularGridInterpolator(
                (coords, coords, coords),
                vel[ci],
                method='linear',
                bounds_error=False,
                fill_value=None,   # extrapolate by nearest
            )
        )
    return interps


def reconstruct_catalog(sample_out, nb, nax, Nmax,
                         lgMmin, lgMmax,
                         vmin, vmax,
                         cmin, cmax,
                         BoxSize,
                         vel_interps,
                         device='cpu'):
    """
    Convert model.sample() output to physical halo catalog arrays.

    Parameters
    ----------
    sample_out  : dict returned by CHARM_Model.sample() (nbatches=1 outer batch)
    vel_interps : list of 3 RegularGridInterpolator — DM velocity at voxel centres

    Returns
    -------
    pos  (N, 3)  float32  in Mpc/h  (periodic)
    lgM  (N,)    float32  log10(M / [Msun/h])
    vel  (N, 3)  float32  in km/s   (v_DM_interp - v_diff_pred)
    conc (N,)    float32  concentration
    """
    nsubs = nb ** 3
    nvox  = nax ** 3
    cell  = BoxSize / (nb * nax)  # cell size in Mpc/h (= 1000/128 ≈ 7.8125)

    # Convert tensors to float32 numpy arrays
    ntot_flat  = np.asarray(sample_out['ntot'][0],  dtype=np.float32)  # (512*4096,)
    # Clip all NSF outputs to their training bounds before denormalization.
    # Neural spline flows can occasionally sample slightly outside the support
    # region B (a few percent of samples); clipping ensures physical outputs.
    m1_flat    = np.clip(np.asarray(sample_out['m1'][0],    dtype=np.float32), 0.0, 1.0)
    mdiff_flat = np.clip(np.asarray(sample_out['mdiff'][0], dtype=np.float32), 0.0, 1.0)
    vel_flat   = np.clip(np.asarray(sample_out['vel'][0],   dtype=np.float32), -0.5, 0.5)
    conc_flat  = np.clip(np.asarray(sample_out['conc'][0],  dtype=np.float32), 0.0, 1.0)
    pos_flat   = np.clip(np.asarray(sample_out['pos'][0],   dtype=np.float32), -0.5, 0.5)

    # Reconstruct normalised masses: M_norm[i] = M_norm[i-1] - Mdiff_norm[i-1]
    M_norm = np.zeros((len(ntot_flat), Nmax), dtype=np.float32)
    M_norm[:, 0] = m1_flat
    for i in range(1, Nmax):
        M_norm[:, i] = np.clip(M_norm[:, i-1] - mdiff_flat[:, i-1], 0.0, 1.0)
    lgM_grid = M_norm * (lgMmax - lgMmin) + lgMmin  # (nvox_total, Nmax)

    # Global voxel indices for all flat entries
    ix_g, iy_g, iz_g = flat_idx_to_global_voxel(nb, nax)

    # Physical reference positions for the stored sub-voxel offsets.
    # Training positions come from NGP_xyz_prop, which assigns
    # index=int(pos/cell + 0.5) and stores offset=(pos - index*cell)/cell.
    # Use index*cell here to stay consistent with the trained position and
    # velocity-residual targets.
    cx = ix_g * cell
    cy = iy_g * cell
    cz = iz_g * cell

    # Collect all halos
    pos_list  = []
    lgM_list  = []
    vel_list  = []
    conc_list = []

    v_diff_range = vmax - vmin

    for ih in range(Nmax):
        has_halo = ntot_flat > ih    # (nvox_total,) bool
        if not np.any(has_halo):
            continue

        # Sub-voxel position offset in voxel units, ∈ [-0.5, 0.5]
        px = pos_flat[has_halo, ih * 3 + 0]
        py = pos_flat[has_halo, ih * 3 + 1]
        pz = pos_flat[has_halo, ih * 3 + 2]

        # Physical positions with periodic boundary conditions.
        x_h = ((cx[has_halo] + px * cell) % BoxSize).astype(np.float32)
        y_h = ((cy[has_halo] + py * cell) % BoxSize).astype(np.float32)
        z_h = ((cz[has_halo] + pz * cell) % BoxSize).astype(np.float32)

        pos_list.append(np.stack([x_h, y_h, z_h], axis=1))

        # Masses
        lgM_list.append(lgM_grid[has_halo, ih])

        # Velocity residuals: v_norm ∈ [-0.5, 0.5] (normalised by vmax-vmin)
        # v_diff = v_FastPM(at halo pos) - v_true  →  v_true = v_FastPM(at halo pos) - v_diff
        # Interpolate DM velocity at each halo's physical position (not voxel centre)
        # to match the training target: v_diff was computed at the exact halo position.
        pts_halo = np.stack([x_h, y_h, z_h], axis=-1)
        v_dm_x_h = vel_interps[0](pts_halo).astype(np.float32)
        v_dm_y_h = vel_interps[1](pts_halo).astype(np.float32)
        v_dm_z_h = vel_interps[2](pts_halo).astype(np.float32)

        vxn = vel_flat[has_halo, ih * 3 + 0]
        vyn = vel_flat[has_halo, ih * 3 + 1]
        vzn = vel_flat[has_halo, ih * 3 + 2]

        vx_phys = (v_dm_x_h - vxn * v_diff_range).astype(np.float32)
        vy_phys = (v_dm_y_h - vyn * v_diff_range).astype(np.float32)
        vz_phys = (v_dm_z_h - vzn * v_diff_range).astype(np.float32)
        vel_list.append(np.stack([vx_phys, vy_phys, vz_phys], axis=1))

        # Concentrations (raw c_sim): c = c_norm * (cmax - cmin) + cmin
        cn = conc_flat[has_halo, ih]
        conc_list.append((cn * (cmax - cmin) + cmin).astype(np.float32))

    if len(pos_list) == 0:
        pos_mock  = np.zeros((0, 3), dtype=np.float32)
        lgM_mock  = np.zeros(0, dtype=np.float32)
        vel_mock  = np.zeros((0, 3), dtype=np.float32)
        conc_mock = np.zeros(0, dtype=np.float32)
    else:
        pos_mock  = np.concatenate(pos_list,  axis=0)
        lgM_mock  = np.concatenate(lgM_list,  axis=0)
        vel_mock  = np.concatenate(vel_list,  axis=0)
        conc_mock = np.concatenate(conc_list, axis=0)

    return pos_mock, lgM_mock, vel_mock, conc_mock


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    # Strip 'module.' prefix if saved from DDP
    new_state = {}
    for k, v in state.items():
        k = k.replace('module.', '', 1)
        k = k.replace('_orig_mod.', '', 1)
        new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    phase   = ckpt.get('phase_idx', 'unknown')
    step    = ckpt.get('global_step', 'unknown')
    val_min = ckpt.get('val_loss_min', float('nan'))
    print(f'  Loaded checkpoint: phase={phase}, step={step}, '
          f'val_loss_min={val_min:.4f}', flush=True)
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    sc  = cfg['sim_settings']
    dc  = cfg['data_settings']
    tc  = cfg['train_settings']

    nb      = int(sc['nb'])          # 8
    ns_d    = int(sc['ns_d'])        # 128
    ns_h    = int(sc['ns_h'])        # 128
    nax     = ns_h // nb             # 16
    nf      = int(sc['nf'])          # 3
    Nmax    = int(sc['Nmax'])        # 4
    BoxSize = float(dc['BoxSize'])   # 1000.0
    z_snap  = str(dc['z_snap'])      # '0.5'
    z       = float(z_snap)          # 0.5

    lgMmin  = float(sc['lgMmin'])
    lgMmax  = float(sc['lgMmax'])
    vmin    = float(sc['vmin'])
    vmax    = float(sc['vmax'])
    cmin    = float(sc['cmin'])
    cmax    = float(sc['cmax'])

    n_cnn_tot = sum(1 if lt == 'cnn' else 2 for lt in sc['layers_types'])
    n_pad     = (nf - 1) // 2 * n_cnn_tot   # 4 for nf=3, [res,res]

    device = torch.device(args.device)
    print(f'Using device: {device}', flush=True)

    # ── resolve paths ─────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(_REPO_ROOT, tc['checkpoint_dir'])
    ckpt_path = (args.checkpoint
                 if args.checkpoint
                 else os.path.join(ckpt_dir, 'charm_joint_best_val.pth'))

    output_dir = (args.output_dir
                  if args.output_dir
                  else os.path.join(ckpt_dir, 'inference'))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'mock_catalog_sim{args.sim_id:04d}.npz')

    # ── build and load model ──────────────────────────────────────────────────
    print('Building model...', flush=True)
    model = build_model(cfg).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    # ── load FastPM fields ────────────────────────────────────────────────────
    print(f'Loading FastPM fields for sim {args.sim_id}...', flush=True)
    rho, vel = load_fastpm(dc['fastpm_dir'], args.sim_id, ns_d, z_snap)
    # vel: (3, 128, 128, 128) in km/s

    # ── load cosmology ────────────────────────────────────────────────────────
    cosmo_vals = load_cosmology(dc['lh_cosmo_file'], args.sim_id)
    print(f'  Cosmology [Om,Ob,h,ns,s8]: {cosmo_vals}', flush=True)

    # ── build conditioning tensors ────────────────────────────────────────────
    print('Building conditioning tensors...', flush=True)
    cond_x, cond_x_nsh, cond_cosmo = build_cond_tensors(
        rho, vel, cosmo_vals, nb, nax, n_pad
    )
    cond_x     = cond_x.to(device)
    cond_x_nsh = cond_x_nsh.to(device)
    cond_cosmo = cond_cosmo.to(device)

    print(f'  cond_x     shape: {tuple(cond_x.shape)}', flush=True)
    print(f'  cond_x_nsh shape: {tuple(cond_x_nsh.shape)}', flush=True)
    print(f'  cond_cosmo shape: {tuple(cond_cosmo.shape)}', flush=True)

    # ── run inference ─────────────────────────────────────────────────────────
    if args.binary_train_prior is not None:
        print(f'Overriding model.binary_train_prior from {model.binary_train_prior} '
              f'to {args.binary_train_prior}', flush=True)
        model.binary_train_prior = float(args.binary_train_prior)

    btp = args.binary_target_prior
    if model.binary_train_prior is not None and btp is None:
        # Auto-estimate π_target from cosmology using the calibrator.
        cal_path = (args.binary_prior_calibrator
                    or os.path.join(ckpt_dir,
                                    'binary_prior_calibrator.npz'))
        if os.path.exists(cal_path):
            btp, info = estimate_target_prior_from_cosmology(
                cosmo_vals, cal_path)
            print(
                f'Auto-estimated binary_target_prior from cosmology:\n'
                f'  calibrator: {cal_path}  (R²={info["calibrator_r2"]:.4f})\n'
                f'  predicted log10(N_occ_voxels) = {info["log10_N_occ_pred"]:.3f} '
                f'→ N_occ_voxels ≈ {info["N_occ_pred"]:.0f}\n'
                f'  π_target = N_occ / ns_h³ = {info["N_occ_pred"]:.0f} / '
                f'{info["ns_cube"]:,d} = {btp:.3e}',
                flush=True,
            )
        else:
            print(
                f'WARNING: model was trained with subsample / alpha '
                f'(π_train={model.binary_train_prior}) and '
                f'--binary_target_prior was not given. No calibrator found '
                f'at {cal_path}; pw_occ will be uncorrected and the mock '
                f'count will be systematically over-predicted. To fix, run:\n'
                f'    python charm/inferers/calibrate_binary_prior.py --config <CFG>\n'
                f'or pass --binary_target_prior explicitly.',
                flush=True,
            )

    if model.binary_train_prior is not None and btp is not None:
        r = (btp / model.binary_train_prior) * \
            ((1 - model.binary_train_prior) / (1 - btp))
        print(f'Applying binary prior correction: π_train={model.binary_train_prior}, '
              f'π_target={btp:.3e}, odds-ratio r={r:.3e}', flush=True)
    print('Running model.sample()...', flush=True)
    with torch.no_grad():
        sample_out = model.sample(
            cond_x,
            cond_x_nsh,
            cond_cosmo,
            sample_binary = True,
            sample_multi  = True,
            sample_m1     = True,
            sample_mdiff  = True,
            sample_vel    = True,
            sample_conc   = True,
            sample_pos    = True,
            use_truth_masses = False,
            binary_target_prior = btp,
        )

    ntot_flat = np.asarray(sample_out['ntot'][0], dtype=np.float32)
    n_mock_total = int(ntot_flat.sum())
    print(f'  Total mock halos predicted: {n_mock_total}', flush=True)

    # ── DM velocity interpolators ─────────────────────────────────────────────
    print('Building DM velocity interpolators...', flush=True)
    vel_interps = build_dm_velocity_interpolators(vel, BoxSize)

    # ── reconstruct catalog ───────────────────────────────────────────────────
    print('Reconstructing halo catalog...', flush=True)
    pos_mock, lgM_mock, vel_mock, conc_mock = reconstruct_catalog(
        sample_out, nb, nax, Nmax,
        lgMmin, lgMmax,
        vmin, vmax,
        cmin, cmax,
        BoxSize,
        vel_interps,
    )

    print(f'  Catalog size: {len(pos_mock)} halos', flush=True)
    if len(pos_mock) > 0:
        print(f'  lgM range:  [{lgM_mock.min():.2f}, {lgM_mock.max():.2f}]', flush=True)
        print(f'  conc range: [{conc_mock.min():.2f}, {conc_mock.max():.2f}]', flush=True)

    # ── reassemble ntot volume ────────────────────────────────────────────────
    ntot_vol = (ntot_flat
                .reshape(nb, nb, nb, nax, nax, nax)
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(ns_h, ns_h, ns_h)
                .astype(np.int32))

    # ── save ──────────────────────────────────────────────────────────────────
    np.savez(
        out_path,
        pos_mock   = pos_mock,
        lgM_mock   = lgM_mock,
        vel_mock   = vel_mock,
        conc_mock  = conc_mock,
        ntot_vol   = ntot_vol,
        # metadata
        sim_id     = np.int32(args.sim_id),
        lgMmin     = np.float32(lgMmin),
        lgMmax     = np.float32(lgMmax),
        vmin       = np.float32(vmin),
        vmax       = np.float32(vmax),
        cmin       = np.float32(cmin),
        cmax       = np.float32(cmax),
        BoxSize    = np.float32(BoxSize),
        ns_h       = np.int32(ns_h),
        z          = np.float32(z),
        cosmo      = cosmo_vals,
    )
    print(f'Saved mock catalog to {out_path}', flush=True)


if __name__ == '__main__':
    main()
