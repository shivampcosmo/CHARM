#!/usr/bin/env python
"""
plot_inference_v2.py
--------------------
Load mock and true halo catalogs for a given simulation and compare:

  1. Projected 2D density maps (side-by-side mock vs truth)
  2. Halo mass function (dN/dlgM)
  3. Real-space Pk — unweighted and mass-weighted
  4. Redshift-space Pk — unweighted and mass-weighted (along z-axis)
  5. Velocity component PDFs
  6. Concentration–mass (c-M) relation

Saves a multi-panel figure to the output directory.

Usage:
    cd /mnt/ceph/users/spandey/CHARM_v2/CHARM

    # Derive true-halo directory automatically from config:
    python charm/plot_inference_v2.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v0.yaml \\
        --mock   ../model_checkpoints/CHARM_JOINT_v0/inference/mock_catalog_sim0003.npz \\
        --sim_id 3

    # Override true-halo directory explicitly:
    python charm/plot_inference_v2.py \\
        --mock   ../model_checkpoints/CHARM_JOINT_v0/inference/mock_catalog_sim0003.npz \\
        --sim_id 3 \\
        --true_halo_dir ../data/halos_Mmin5e12
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mock', required=True,
                   help='Path to mock catalog .npz from run_inference_v2.py')
    p.add_argument('--sim_id', type=int, required=True,
                   help='Simulation ID (used to find true HDF5 file).')
    p.add_argument('--config', default=None,
                   help='Path to YAML training config. Used to auto-derive the '
                        'true-halo directory (data_settings.halo_hdf5_dir) when '
                        '--true_halo_dir is not given.')
    p.add_argument('--true_halo_dir', default=None,
                   help='Directory containing per-sim true halo HDF5 files. '
                        'Takes precedence over --config. One of --config or '
                        '--true_halo_dir must be provided.')
    p.add_argument('--output_dir', default=None,
                   help='Output directory for figures. Defaults to same directory as --mock.')
    p.add_argument('--z_snap', default='0.5',
                   help='Redshift string in HDF5 filename.')
    return p.parse_args()


# ── load catalogs ─────────────────────────────────────────────────────────────

def load_mock(path: str):
    d = np.load(path)
    meta = {k: d[k].item() for k in ['sim_id', 'lgMmin', 'lgMmax', 'BoxSize', 'ns_h', 'z']}
    meta['vmin']   = float(d['vmin'])
    meta['vmax']   = float(d['vmax'])
    meta['cmin']   = float(d['cmin'])
    meta['cmax']   = float(d['cmax'])
    meta['cosmo']  = d['cosmo']
    return {
        'pos':  d['pos_mock'],    # (N, 3) Mpc/h
        'lgM':  d['lgM_mock'],    # (N,)
        'vel':  d['vel_mock'],    # (N, 3) km/s
        'conc': d['conc_mock'],   # (N,)
        'ntot_vol': d['ntot_vol'],  # (128,128,128)
    }, meta


def load_true(halo_dir: str, sim_id: int, z_snap: str = '0.5'):
    """Load true halo catalog from per-sim HDF5. Returns pos (Mpc/h), lgM, vel, conc."""
    fpath = os.path.join(halo_dir, str(sim_id), f'halos_rockstar_200c_z{z_snap}.h5')
    with h5py.File(fpath, 'r') as f:
        N_halos  = f['N_halos'][:]      # (128,128,128) int16 — original NGP assignment
        M_halos  = f['M_halos'][:]      # (128,128,128,6) float32 log10 mass
        pos_off  = f['pos_halos'][:]    # (128,128,128,6,3) sub-voxel offsets in voxel units
        v_true   = f['v_halos_true'][:] # (128,128,128,6,3) km/s
        c_sim    = f['c_halos_sim'][:]  # (128,128,128,6)
        BoxSize  = float(f.attrs['BoxSize'])
        grid     = int(f.attrs['grid'])

    cell = BoxSize / grid  # Mpc/h per voxel (≈ 7.8125)

    # Build absolute positions from voxel indices + sub-voxel offsets
    ix, iy, iz = np.meshgrid(np.arange(grid), np.arange(grid), np.arange(grid),
                              indexing='ij')   # (128,128,128)
    cx = (ix + 0.5) * cell   # voxel centre x in Mpc/h
    cy = (iy + 0.5) * cell
    cz = (iz + 0.5) * cell

    pos_list  = []
    lgM_list  = []
    vel_list  = []
    conc_list = []
    nMax_h = M_halos.shape[-1]   # up to 6 halos per voxel

    for ih in range(nMax_h):
        mask = N_halos > ih    # (128,128,128)
        if not np.any(mask):
            continue

        # Sub-voxel offsets ∈ [-0.5, 0.5] in voxel units
        px = pos_off[mask, ih, 0]
        py = pos_off[mask, ih, 1]
        pz = pos_off[mask, ih, 2]

        x_h = ((cx[mask] + px * cell) % BoxSize).astype(np.float32)
        y_h = ((cy[mask] + py * cell) % BoxSize).astype(np.float32)
        z_h = ((cz[mask] + pz * cell) % BoxSize).astype(np.float32)
        pos_list.append(np.stack([x_h, y_h, z_h], axis=1))

        lgM_list.append(M_halos[mask, ih].astype(np.float32))
        vel_list.append(v_true[mask, ih, :].astype(np.float32))
        conc_list.append(c_sim[mask, ih].astype(np.float32))

    pos  = np.concatenate(pos_list,  axis=0)
    lgM  = np.concatenate(lgM_list,  axis=0)
    vel  = np.concatenate(vel_list,  axis=0)
    conc = np.concatenate(conc_list, axis=0)

    return {'pos': pos, 'lgM': lgM, 'vel': vel, 'conc': conc,
            'ntot_vol': N_halos.astype(np.int32)}, BoxSize


# ── statistics ────────────────────────────────────────────────────────────────

def halo_mass_function(lgM: np.ndarray, lgMmin: float, lgMmax: float,
                       n_bins: int = 20, BoxSize: float = 1000.0):
    """Return (bin_centres, dN/dlgM) per (Mpc/h)³."""
    bins   = np.linspace(lgMmin, lgMmax, n_bins + 1)
    counts, _ = np.histogram(lgM, bins=bins)
    dlgM   = bins[1] - bins[0]
    V      = BoxSize ** 3
    dn_dlgM = counts / (V * dlgM)
    return 0.5 * (bins[:-1] + bins[1:]), dn_dlgM


def _density_field(pos: np.ndarray, weights, Ng: int, BoxSize: float,
                   MAS: str = 'TSC') -> np.ndarray:
    """Paint positions to a (Ng,Ng,Ng) overdensity grid, normalised to mean=1."""
    import MAS_library as MASL

    pos32  = np.ascontiguousarray(pos.astype(np.float32))
    Lbox32 = np.float32(BoxSize)
    delta  = np.zeros((Ng, Ng, Ng), dtype=np.float32)

    if weights is None:
        MASL.MA(pos32, delta, Lbox32, MAS)
    else:
        w32 = np.ascontiguousarray(weights.astype(np.float32))
        MASL.MA(pos32, delta, Lbox32, MAS, W=w32)

    mean = delta.mean(dtype=np.float64)
    if mean <= 0:
        return None
    return (delta / mean - 1.0).astype(np.float32)


def power_spectrum(pos: np.ndarray, weights: np.ndarray,
                   Ng: int, BoxSize: float, kmax: float = 0.4,
                   MAS: str = 'TSC', axis: int = 0):
    """
    Compute P(k) monopole using Pylians with TSC (or other) mass assignment.

    Returns (k [h/Mpc], Pk [(Mpc/h)^3]) truncated to k <= kmax.
    """
    import Pk_library as PKL

    delta = _density_field(pos, weights, Ng, BoxSize, MAS)
    if delta is None:
        return np.zeros(1), np.zeros(1)

    Pk_obj = PKL.Pk(delta, np.float32(BoxSize), axis=axis, MAS=MAS,
                    threads=1, verbose=False)
    k = Pk_obj.k3D
    P = Pk_obj.Pk[:, 0]
    sel = (k > 0) & (k <= kmax) & np.isfinite(P) & (P > 0)
    return k[sel], P[sel]


def power_spectrum_multipoles(pos: np.ndarray, weights,
                               Ng: int, BoxSize: float, kmax: float = 0.4,
                               MAS: str = 'TSC', axis: int = 2):
    """
    Compute P0, P2, P4 multipoles using Pylians.

    `axis` must match the line-of-sight direction used when applying RSD.
    Returns (k, P0, P2, P4) truncated to k <= kmax.
    P0 > 0; P2 and P4 can be negative.
    """
    import Pk_library as PKL

    delta = _density_field(pos, weights, Ng, BoxSize, MAS)
    if delta is None:
        z = np.zeros(1)
        return z, z, z, z

    Pk_obj = PKL.Pk(delta, np.float32(BoxSize), axis=axis, MAS=MAS,
                    threads=1, verbose=False)
    k  = Pk_obj.k3D
    P0 = Pk_obj.Pk[:, 0]
    P2 = Pk_obj.Pk[:, 1]
    P4 = Pk_obj.Pk[:, 2]
    sel = (k > 0) & (k <= kmax) & np.isfinite(P0)
    return k[sel], P0[sel], P2[sel], P4[sel]


def apply_rsd(pos: np.ndarray, vel: np.ndarray,
              BoxSize: float, z: float,
              cosmo: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Apply redshift-space distortions along `axis`.

    s_parallel = r_parallel + v_parallel * (1+z) / (H0 * E(z))
    where H0*E(z) is in km/s/(Mpc/h) with H0=100 h km/s/Mpc/h → H0*E=100*E km/s/(Mpc/h).

    Returns periodic position array with axis `axis` distorted.
    """
    Omega_m = float(cosmo[0])
    Ez = np.sqrt(Omega_m * (1.0 + z) ** 3 + (1.0 - Omega_m))
    H_eff = 100.0 * Ez    # km/s / (Mpc/h)

    pos_rsd = pos.copy()
    pos_rsd[:, axis] = (pos[:, axis]
                        + vel[:, axis] * (1.0 + z) / H_eff) % BoxSize
    return pos_rsd


def projected_map(pos: np.ndarray, BoxSize: float,
                  Np: int = 256, depth_frac: float = 0.2) -> np.ndarray:
    """
    Project halos onto the xy-plane using halos within a depth slab.

    depth_frac : fraction of box along z to include
    Returns 2D histogram (Np, Np).
    """
    depth = BoxSize * depth_frac
    mask  = pos[:, 2] < depth
    if mask.sum() == 0:
        return np.zeros((Np, Np), dtype=np.float64)
    p2d = pos[mask][:, :2]
    H, _, _ = np.histogram2d(
        p2d[:, 0], p2d[:, 1],
        bins=Np, range=[[0, BoxSize], [0, BoxSize]],
    )
    return H


# ── plotting ──────────────────────────────────────────────────────────────────

def make_all_plots(mock, true_cat, meta, output_dir):
    BoxSize = meta['BoxSize']
    lgMmin  = meta['lgMmin']
    lgMmax  = meta['lgMmax']
    z       = meta['z']
    cosmo   = meta['cosmo']
    sim_id  = meta['sim_id']
    Ng      = 384   # grid for Pk (Pylians TSC)
    K_MAX   = 0.4   # h/Mpc — show out to at least this scale

    # ── colours / labels ──────────────────────────────────────────────────────
    C_MOCK  = '#E07B54'
    C_TRUE  = '#5B8DB8'

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f'CHARM inference — sim {sim_id:04d}  '
                 f'($N_{{mock}}={len(mock["pos"])}$, $N_{{true}}={len(true_cat["pos"])}$)',
                 fontsize=13, y=0.99)

    gs  = fig.add_gridspec(4, 4, hspace=0.38, wspace=0.32)

    # ── Row 0: projected 2D density maps ──────────────────────────────────────
    # Coarse grid for the visualization so large-scale structure is legible.
    depth_frac = 0.20
    Hp = 128
    map_mock  = projected_map(mock['pos'],     BoxSize, Hp, depth_frac)
    map_true  = projected_map(true_cat['pos'], BoxSize, Hp, depth_frac)

    vmax_map = max(map_mock.max(), map_true.max(), 1.0)
    norm_map = LogNorm(vmin=0.5, vmax=vmax_map, clip=True)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(map_mock.T + 0.1, origin='lower', norm=norm_map, cmap='inferno',
              extent=[0, BoxSize, 0, BoxSize], aspect='auto',
              interpolation='gaussian')
    ax.set_title(f'Mock 2D projection (grid {Hp}², slab {depth_frac:.2f}L)',
                 fontsize=10)
    ax.set_xlabel('x [Mpc/h]'); ax.set_ylabel('y [Mpc/h]')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(map_true.T + 0.1, origin='lower', norm=norm_map, cmap='inferno',
              extent=[0, BoxSize, 0, BoxSize], aspect='auto',
              interpolation='gaussian')
    ax.set_title(f'True 2D projection (grid {Hp}², slab {depth_frac:.2f}L)',
                 fontsize=10)
    ax.set_xlabel('x [Mpc/h]'); ax.set_ylabel('y [Mpc/h]')

    # Ratio map
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_map = np.where(map_true > 0, map_mock / (map_true + 0.01), 0.0)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(ratio_map.T, origin='lower', cmap='RdBu_r', vmin=0.0, vmax=2.0,
                   extent=[0, BoxSize, 0, BoxSize], aspect='auto')
    ax.set_title('Mock / True ratio', fontsize=10)
    ax.set_xlabel('x [Mpc/h]'); ax.set_ylabel('y [Mpc/h]')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # N_tot per voxel histogram (log y)
    ntot_mock = mock['ntot_vol'].ravel()
    ntot_true = true_cat.get('ntot_vol', None)
    ax = fig.add_subplot(gs[0, 3])
    max_n = max(int(ntot_mock.max()),
                int(ntot_true.max()) if ntot_true is not None else 0)
    bins_n = np.arange(0, max_n + 2) - 0.5
    ax.hist(ntot_mock, bins=bins_n, density=True, alpha=0.7, color=C_MOCK,
            label=f'Mock  ({len(mock["pos"])} halos)')
    if ntot_true is not None:
        ax.hist(ntot_true.ravel(), bins=bins_n, density=True, alpha=0.7,
                color=C_TRUE, label=f'True  ({len(true_cat["pos"])} halos)')
    ax.set_yscale('log')
    ax.set_xlabel('$N_{tot}$ per voxel'); ax.set_ylabel('P')
    ax.set_title('Per-voxel halo count', fontsize=10)
    ax.legend(fontsize=8)

    # ── Row 1: HMF (+ratio) + Pk real space ──────────────────────────────────
    # HMF panel: stack HMF on top with ratio (mock/true) underneath.
    sub_hmf = gs[1, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_hmf  = fig.add_subplot(sub_hmf[0])
    ax_hr   = fig.add_subplot(sub_hmf[1], sharex=ax_hmf)
    xm, ym = halo_mass_function(mock['lgM'],     lgMmin, lgMmax, BoxSize=BoxSize)
    xt, yt = halo_mass_function(true_cat['lgM'], lgMmin, lgMmax, BoxSize=BoxSize)
    ax_hmf.semilogy(xm, ym, '-o', color=C_MOCK, ms=4, lw=1.5, label='Mock')
    ax_hmf.semilogy(xt, yt, '-s', color=C_TRUE, ms=4, lw=1.5, label='True')
    ax_hmf.set_ylabel('dn/d$\\log M$  [(Mpc/h)$^{-3}$]')
    ax_hmf.set_title('Halo mass function', fontsize=10)
    ax_hmf.legend(fontsize=9)
    plt.setp(ax_hmf.get_xticklabels(), visible=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        hmf_ratio = np.where(yt > 0, ym / yt, np.nan)
    ax_hr.plot(xm, hmf_ratio, '-o', color='k', ms=3, lw=1.2)
    ax_hr.axhline(1.0, ls='--', color='gray', lw=1.0)
    ax_hr.fill_between(xm, 0.9, 1.1, alpha=0.15, color='gray')
    ax_hr.set_ylim(0.5, 1.5)
    ax_hr.set_xlabel('$\\log_{10} M$ [$M_\\odot/h$]')
    ax_hr.set_ylabel('mock/true')

    # Real-space Pk (unweighted) — main panel + ratio subplot
    sub_rs = gs[1, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_rs  = fig.add_subplot(sub_rs[0])
    ax_rsr = fig.add_subplot(sub_rs[1], sharex=ax_rs)
    k_m, Pk_m = power_spectrum(mock['pos'],     None, Ng, BoxSize, kmax=K_MAX)
    k_t, Pk_t = power_spectrum(true_cat['pos'], None, Ng, BoxSize, kmax=K_MAX)
    ax_rs.loglog(k_m, Pk_m, '-', color=C_MOCK, lw=1.5, label='Mock')
    ax_rs.loglog(k_t, Pk_t, '-', color=C_TRUE, lw=1.5, label='True')
    ax_rs.set_ylabel('$P(k)$ [(Mpc/h)$^3$]')
    ax_rs.set_title(f'Real-space Pk (unweighted, TSC {Ng}$^3$)', fontsize=10)
    ax_rs.legend(fontsize=9)
    plt.setp(ax_rs.get_xticklabels(), visible=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        _rs_ratio = np.where(Pk_t > 0, Pk_m / np.interp(k_m, k_t, Pk_t), np.nan)
    ax_rsr.semilogx(k_m, _rs_ratio, '-', color='k', lw=1.2)
    ax_rsr.axhline(1.0, ls='--', color='gray', lw=1.0)
    ax_rsr.fill_between(k_m, 0.9, 1.1, alpha=0.15, color='gray')
    ax_rsr.set_ylim(0.5, 1.5)
    ax_rsr.set_xlabel('$k$ [h/Mpc]')
    ax_rsr.set_ylabel('mock/true')

    # Real-space Pk (mass-weighted) — main panel + ratio subplot
    M_mock_lin = 10.0 ** mock['lgM']
    M_true_lin = 10.0 ** true_cat['lgM']
    sub_mw = gs[1, 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_mw  = fig.add_subplot(sub_mw[0])
    ax_mwr = fig.add_subplot(sub_mw[1], sharex=ax_mw)
    k_mw, Pk_mw = power_spectrum(mock['pos'],     M_mock_lin/1e14, Ng, BoxSize, kmax=K_MAX)
    k_tw, Pk_tw = power_spectrum(true_cat['pos'], M_true_lin/1e14, Ng, BoxSize, kmax=K_MAX)
    ax_mw.loglog(k_mw, Pk_mw, '-', color=C_MOCK, lw=1.5, label='Mock')
    ax_mw.loglog(k_tw, Pk_tw, '-', color=C_TRUE, lw=1.5, label='True')
    ax_mw.set_ylabel('$P_M(k)$ [(Mpc/h)$^3$]')
    ax_mw.set_title('Real-space Pk (mass-weighted)', fontsize=10)
    ax_mw.legend(fontsize=9)
    plt.setp(ax_mw.get_xticklabels(), visible=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        _mw_ratio = np.where(Pk_tw > 0, Pk_mw / np.interp(k_mw, k_tw, Pk_tw), np.nan)
    ax_mwr.semilogx(k_mw, _mw_ratio, '-', color='k', lw=1.2)
    ax_mwr.axhline(1.0, ls='--', color='gray', lw=1.0)
    ax_mwr.fill_between(k_mw, 0.9, 1.1, alpha=0.15, color='gray')
    ax_mwr.set_ylim(0.5, 1.5)
    ax_mwr.set_xlabel('$k$ [h/Mpc]')
    ax_mwr.set_ylabel('mock/true')

    # Combined Pk ratio comparison (unweighted + mass-weighted on same axes)
    ax = fig.add_subplot(gs[1, 3])
    if k_m.size and k_t.size:
        ax.semilogx(k_m, _rs_ratio, '-', color='k', lw=1.5, label='Unweighted')
    if k_mw.size and k_tw.size:
        ax.semilogx(k_mw, _mw_ratio, '--', color='purple', lw=1.5, label='Mass-weighted')
    ax.axhline(1.0, ls='--', color='gray', lw=1.0)
    ax.fill_between(k_m if k_m.size else k_mw, 0.9, 1.1, alpha=0.15, color='gray')
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('$k$ [h/Mpc]')
    ax.set_ylabel('$P_{mock}/P_{true}$')
    ax.set_title('Real-space Pk ratios', fontsize=10)
    ax.legend(fontsize=9)

    # ── Row 2: RSD Pk multipoles (P0, P2/P0, P4/P0) ─────────────────────────
    # RSD applied along z-axis (axis=2); multipoles computed with the same axis
    # so that the LOS decomposition is consistent.
    pos_rsd_m = apply_rsd(mock['pos'],     mock['vel'],     BoxSize, z, cosmo, axis=2)
    pos_rsd_t = apply_rsd(true_cat['pos'], true_cat['vel'], BoxSize, z, cosmo, axis=2)

    k_rs_m,  P0_m,  P2_m,  P4_m  = power_spectrum_multipoles(
        pos_rsd_m, None, Ng, BoxSize, kmax=K_MAX, axis=2)
    k_rs_t,  P0_t,  P2_t,  P4_t  = power_spectrum_multipoles(
        pos_rsd_t, None, Ng, BoxSize, kmax=K_MAX, axis=2)
    k_rs_mw, P0_mw, _, _ = power_spectrum_multipoles(
        pos_rsd_m, M_mock_lin, Ng, BoxSize, kmax=K_MAX, axis=2)
    k_rs_tw, P0_tw, _, _ = power_spectrum_multipoles(
        pos_rsd_t, M_true_lin, Ng, BoxSize, kmax=K_MAX, axis=2)

    def _multipole_over_monopole(Pell, P0):
        """Safe dimensionless multipole ratio Pell/P0."""
        p0_abs = np.abs(P0)
        floor = 1e-6 * np.nanmax(p0_abs) if np.any(np.isfinite(p0_abs)) else 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((p0_abs > floor) & np.isfinite(Pell), Pell / P0, np.nan)

    P2oP0_m = _multipole_over_monopole(P2_m, P0_m)
    P2oP0_t = _multipole_over_monopole(P2_t, P0_t)
    P4oP0_m = _multipole_over_monopole(P4_m, P0_m)
    P4oP0_t = _multipole_over_monopole(P4_t, P0_t)

    def _pk_ratio_safe(Pm, Pt, k_m, k_t):
        """Element-wise mock/true ratio, NaN where |true| is negligibly small."""
        Pt_interp = np.interp(k_m, k_t, Pt)
        scale = np.maximum(np.abs(Pt_interp), 1e-3 * np.abs(Pt).max())
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(scale > 0, Pm / Pt_interp, np.nan)

    def _add_ratio_subplot(gs_cell, k_m, k_t, Pm, Pt, title, ylabel,
                           use_loglog=True):
        sub = gs_cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(sub[0])
        ax_rat  = fig.add_subplot(sub[1], sharex=ax_main)
        if use_loglog:
            ax_main.loglog(k_m, Pm, '-', color=C_MOCK, lw=1.5, label='Mock')
            ax_main.loglog(k_t, Pt, '-', color=C_TRUE, lw=1.5, label='True')
        else:
            ax_main.semilogx(k_m, Pm, '-', color=C_MOCK, lw=1.5, label='Mock')
            ax_main.semilogx(k_t, Pt, '-', color=C_TRUE, lw=1.5, label='True')
            ax_main.axhline(0, ls=':', color='gray', lw=0.8)
        ax_main.set_ylabel(ylabel)
        ax_main.set_title(title, fontsize=10)
        ax_main.legend(fontsize=9)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ratio = _pk_ratio_safe(Pm, Pt, k_m, k_t)
        ax_rat.semilogx(k_m, ratio, '-', color='k', lw=1.2)
        ax_rat.axhline(1.0, ls='--', color='gray', lw=1.0)
        ax_rat.fill_between(k_m, 0.9, 1.1, alpha=0.15, color='gray')
        ax_rat.set_ylim(0.5, 1.5)
        ax_rat.set_xlabel('$k$ [h/Mpc]')
        ax_rat.set_ylabel('mock/true')
        return ax_main, ax_rat

    _add_ratio_subplot(
        gs[2, 0], k_rs_m, k_rs_t, P0_m, P0_t,
        title='RSD $P_0$ — monopole (unweighted)',
        ylabel='$P_0^s(k)$ [(Mpc/h)$^3$]',
        use_loglog=True,
    )
    _add_ratio_subplot(
        gs[2, 1], k_rs_m, k_rs_t, P2oP0_m, P2oP0_t,
        title='RSD $P_2/P_0$ — quadrupole ratio (unweighted)',
        ylabel='$P_2^s(k) / P_0^s(k)$',
        use_loglog=False,
    )
    _add_ratio_subplot(
        gs[2, 2], k_rs_m, k_rs_t, P4oP0_m, P4oP0_t,
        title='RSD $P_4/P_0$ — hexadecapole ratio (unweighted)',
        ylabel='$P_4^s(k) / P_0^s(k)$',
        use_loglog=False,
    )
    _add_ratio_subplot(
        gs[2, 3], k_rs_mw, k_rs_tw, P0_mw, P0_tw,
        title='RSD $P_0$ — monopole (mass-weighted)',
        ylabel='$P_{M,0}^s(k)$ [(Mpc/h)$^3$]',
        use_loglog=True,
    )

    # ── Row 3: Velocity PDFs + c-M relation ──────────────────────────────────
    # Use meta vmin/vmax with a small buffer so bins cover training range exactly.
    v_lo   = meta['vmin'] * 1.15
    v_hi   = meta['vmax'] * 1.15
    v_bins = np.linspace(v_lo, v_hi, 60)
    v_mid  = 0.5 * (v_bins[:-1] + v_bins[1:])

    ax = fig.add_subplot(gs[3, 0])
    pdf_max = 0.0
    for ci, lab in enumerate(['$v_x$', '$v_y$', '$v_z$']):
        vm_h, _ = np.histogram(mock['vel'][:, ci],     bins=v_bins, density=True)
        vt_h, _ = np.histogram(true_cat['vel'][:, ci], bins=v_bins, density=True)
        pdf_max = max(pdf_max, vm_h.max(), vt_h.max())
        ax.semilogy(v_mid, vm_h + 1e-12, ls='-',  color=C_MOCK,
                    alpha=0.5 + 0.2*ci, lw=1.2, label=f'Mock {lab}')
        ax.semilogy(v_mid, vt_h + 1e-12, ls='--', color=C_TRUE,
                    alpha=0.5 + 0.2*ci, lw=1.2, label=f'True {lab}')
    if pdf_max > 0:
        ax.set_ylim(pdf_max * 1e-2, pdf_max * 1.5)
    ax.set_xlabel('Velocity [km/s]'); ax.set_ylabel('PDF')
    ax.set_title('Velocity PDF', fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # Velocity dispersion vs mass
    ax = fig.add_subplot(gs[3, 1])
    lgM_bins  = np.linspace(meta['lgMmin'], meta['lgMmax'], 10)
    lgM_mid   = 0.5 * (lgM_bins[:-1] + lgM_bins[1:])
    for cat, color, label in [(mock, C_MOCK, 'Mock'), (true_cat, C_TRUE, 'True')]:
        vdisp = []
        for j in range(len(lgM_bins) - 1):
            m = (cat['lgM'] >= lgM_bins[j]) & (cat['lgM'] < lgM_bins[j+1])
            if m.sum() > 2:
                vdisp.append(np.std(cat['vel'][m]))
            else:
                vdisp.append(np.nan)
        ax.plot(lgM_mid, vdisp, '-o', color=color, ms=4, lw=1.5, label=label)
    ax.set_xlabel('$\\log_{10} M$')
    ax.set_ylabel('Velocity dispersion [km/s]')
    ax.set_title('Velocity dispersion vs mass', fontsize=10)
    ax.legend(fontsize=9)

    # c-M relation
    ax = fig.add_subplot(gs[3, 2])
    lgM_bins2  = np.linspace(meta['lgMmin'], meta['lgMmax'], 10)
    lgM_mid2   = 0.5 * (lgM_bins2[:-1] + lgM_bins2[1:])
    c_med_all = []
    for cat, color, label in [(mock, C_MOCK, 'Mock'), (true_cat, C_TRUE, 'True')]:
        c_med, c_err = [], []
        for j in range(len(lgM_bins2) - 1):
            m = (cat['lgM'] >= lgM_bins2[j]) & (cat['lgM'] < lgM_bins2[j+1])
            if m.sum() > 2:
                c_med.append(np.median(cat['conc'][m]))
                c_err.append(np.std(cat['conc'][m]) / np.sqrt(m.sum()))
            else:
                c_med.append(np.nan)
                c_err.append(np.nan)
        ax.errorbar(lgM_mid2, c_med, yerr=c_err, fmt='-o', color=color,
                    ms=4, lw=1.5, capsize=2, label=label)
        c_med_all.extend([v for v in c_med if not np.isnan(v)])
    ax.set_xlabel('$\\log_{10} M$')
    ax.set_ylabel('Concentration $c_{200c}$')
    ax.set_title('c–M relation', fontsize=10)
    ax.legend(fontsize=9)
    # Clip y-axis to the true-data percentile range so NSF tail samples
    # in the mock don't inflate the scale.
    if c_med_all:
        _c_true_med = [np.median(true_cat['conc'][
            (true_cat['lgM'] >= lgM_bins2[j]) & (true_cat['lgM'] < lgM_bins2[j+1])
        ]) for j in range(len(lgM_bins2)-1)
            if ((true_cat['lgM'] >= lgM_bins2[j]) & (true_cat['lgM'] < lgM_bins2[j+1])).sum() > 2]
        if _c_true_med:
            _ylo = max(0, np.nanmin(_c_true_med) * 0.5)
            _yhi = np.nanmax(_c_true_med) * 2.0
            ax.set_ylim(_ylo, _yhi)

    # Concentration PDF
    # Use data-adaptive bins: normalization bounds [cmin, cmax] can be much
    # wider than the actual concentration range, causing the mock histogram
    # (which may have NSF tail samples near cmax) to look flat & weird while
    # the true histogram (concentrated in the lower part) looks "ok".
    # Fix: derive bin edges from the joint 0.5–99.5 percentile range.
    ax = fig.add_subplot(gs[3, 3])
    _c_all = np.concatenate([mock['conc'], true_cat['conc']])
    c_lo = np.percentile(_c_all, 0.5)
    c_hi = np.percentile(_c_all, 99.5)
    c_bins = np.linspace(c_lo, c_hi, 30)
    ax.hist(mock['conc'],     bins=c_bins, density=True, alpha=0.6,
            color=C_MOCK, label=f'Mock (med={np.nanmedian(mock["conc"]):.1f})')
    ax.hist(true_cat['conc'], bins=c_bins, density=True, alpha=0.6,
            color=C_TRUE, label=f'True (med={np.nanmedian(true_cat["conc"]):.1f})')
    ax.set_xlabel('Concentration $c_{200c}$')
    ax.set_ylabel('PDF')
    ax.set_title(f'Concentration distribution\n'
                 f'[norm. bounds: {meta["cmin"]:.0f}–{meta["cmax"]:.0f}]',
                 fontsize=10)
    ax.legend(fontsize=9)

    # ── save ─────────────────────────────────────────────────────────────────
    out_fname = os.path.join(output_dir, f'inference_stats_sim{sim_id:04d}.pdf')
    fig.savefig(out_fname, bbox_inches='tight', dpi=150)
    print(f'Saved figure: {out_fname}', flush=True)

    # Also save a PNG for quick viewing
    png_fname = out_fname.replace('.pdf', '.png')
    fig.savefig(png_fname, bbox_inches='tight', dpi=150)
    print(f'Saved figure: {png_fname}', flush=True)

    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── load mock catalog ─────────────────────────────────────────────────────
    print(f'Loading mock catalog: {args.mock}', flush=True)
    mock, meta = load_mock(args.mock)
    print(f'  Mock halos: {len(mock["pos"])}', flush=True)

    # ── locate true halo directory ────────────────────────────────────────────
    if args.true_halo_dir:
        halo_dir = args.true_halo_dir
    elif args.config:
        cfg      = load_config(args.config)
        dc       = cfg['data_settings']
        halo_dir = dc.get('halo_hdf5_dir') or os.path.join(
            _REPO_ROOT, '..', 'data',
            f'halos_Mmin{dc.get("Mmin_cut_str", "")}',
        )
        if not os.path.isdir(halo_dir):
            raise FileNotFoundError(
                f'Could not find halo directory derived from config: {halo_dir}\n'
                'Pass --true_halo_dir explicitly.'
            )
    else:
        raise ValueError(
            'Provide either --config (to auto-derive the true-halo directory) '
            'or --true_halo_dir explicitly.'
        )

    # ── load true catalog ─────────────────────────────────────────────────────
    print(f'Loading true catalog from: {halo_dir}/{args.sim_id}/', flush=True)
    true_cat, _ = load_true(halo_dir, args.sim_id, args.z_snap)
    print(f'  True halos: {len(true_cat["pos"])}', flush=True)

    # ── output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(args.mock))
    os.makedirs(output_dir, exist_ok=True)

    # ── make plots ────────────────────────────────────────────────────────────
    print('Generating plots...', flush=True)
    make_all_plots(mock, true_cat, meta, output_dir)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
