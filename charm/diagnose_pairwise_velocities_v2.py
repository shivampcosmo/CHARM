#!/usr/bin/env python
"""
Diagnose small-scale pairwise velocity differences between mock and truth.

The key statistic for small-scale RSD quadrupoles is not the one-point velocity
PDF, but pairwise LOS velocity structure.  This script compares:

  sigma_12,los(r) = std[v_los(j) - v_los(i)]
  v_12,rad(r)     = mean[(v(j) - v(i)) . r_hat]

for true and mock catalogs, using periodic separations and a sampled set of
anchor halos.  It also optionally repeats the same diagnostic for the velocity
residual v_PM(pos) - v_halo, which is the target learned by the velocity head.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config
from plot_inference_v2 import load_mock, load_true


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML training config.')
    p.add_argument('--sim_id', type=int, required=True,
                   help='Simulation id to diagnose.')
    p.add_argument('--mock', default=None,
                   help='Path to mock catalog .npz. Defaults to '
                        '<checkpoint_dir>/inference/mock_catalog_simXXXX.npz.')
    p.add_argument('--mock_dir', default=None,
                   help='Directory containing mock catalogs. Ignored if --mock is set.')
    p.add_argument('--true_halo_dir', default=None,
                   help='Directory containing true halo HDF5 files. Defaults to config.')
    p.add_argument('--output_dir', default=None,
                   help='Output directory. Defaults to mock catalog directory.')
    p.add_argument('--z_snap', default=None,
                   help='Redshift string. Defaults to data_settings.z_snap.')
    p.add_argument('--los_axis', type=int, choices=(0, 1, 2), default=2,
                   help='Line-of-sight axis used for RSD.')
    p.add_argument('--rmin', type=float, default=1.0,
                   help='Minimum pair separation [Mpc/h].')
    p.add_argument('--rmax', type=float, default=50.0,
                   help='Maximum pair separation [Mpc/h].')
    p.add_argument('--nbins', type=int, default=16,
                   help='Number of log-spaced separation bins.')
    p.add_argument('--max_halos', type=int, default=150000,
                   help='Maximum halos per catalog after optional random subsampling.')
    p.add_argument('--n_anchors', type=int, default=25000,
                   help='Number of anchor halos used for pair sampling.')
    p.add_argument('--max_neighbors', type=int, default=256,
                   help='Maximum neighbors sampled per anchor within rmax.')
    p.add_argument('--seed', type=int, default=1234,
                   help='Random seed for halo/neighbor subsampling.')
    p.add_argument('--mass_min', type=float, default=None,
                   help='Optional minimum log10 halo mass for the diagnostic.')
    p.add_argument('--skip_residuals', action='store_true',
                   help='Only diagnose total velocities; skip v_PM - v_halo residuals.')
    return p.parse_args()


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(_REPO_ROOT, path))


def resolve_paths(cfg: dict, args):
    ckpt_dir = repo_path(cfg['train_settings']['checkpoint_dir'])
    mock_dir = os.path.abspath(args.mock_dir or os.path.join(ckpt_dir, 'inference'))
    mock_path = os.path.abspath(
        args.mock or os.path.join(mock_dir, f'mock_catalog_sim{args.sim_id:04d}.npz'))
    true_halo_dir = repo_path(args.true_halo_dir or cfg['data_settings']['halo_hdf5_dir'])
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(mock_path))
    return mock_path, true_halo_dir, output_dir


def load_fastpm_velocity(fastpm_dir: str, sim_id: int, grid: int, z_snap: str):
    fname = os.path.join(
        fastpm_dir,
        str(sim_id),
        f'velocity_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk',
    )
    if not os.path.exists(fname):
        raise FileNotFoundError(f'FastPM velocity file not found: {fname}')
    with open(fname, 'rb') as f:
        vel = pickle.load(f)['velocity_cic_unpad_combined']
    return (np.asarray(vel) * 1000.0).astype(np.float32)


def build_velocity_interpolators(vel: np.ndarray, BoxSize: float):
    ns = vel.shape[1]
    cell = BoxSize / ns
    coords = np.linspace(0.5 * cell, BoxSize - 0.5 * cell, ns, dtype=np.float32)
    return [
        RegularGridInterpolator(
            (coords, coords, coords),
            vel[i],
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
        for i in range(3)
    ]


def pm_velocity_at(pos: np.ndarray, interps) -> np.ndarray:
    return np.stack([interp(pos).astype(np.float32) for interp in interps], axis=1)


def maybe_subsample(cat: dict, max_halos: int, seed: int, mass_min: float | None):
    rng = np.random.default_rng(seed)
    mask = np.ones(len(cat['pos']), dtype=bool)
    if mass_min is not None:
        mask &= cat['lgM'] >= mass_min
    idx = np.flatnonzero(mask)
    if idx.size > max_halos:
        idx = rng.choice(idx, size=max_halos, replace=False)
    idx = np.sort(idx)
    return {
        'pos': np.asarray(cat['pos'][idx], dtype=np.float64),
        'vel': np.asarray(cat['vel'][idx], dtype=np.float64),
        'lgM': np.asarray(cat['lgM'][idx], dtype=np.float64),
    }


def init_accumulators(nbins: int):
    return {
        'count': np.zeros(nbins, dtype=np.int64),
        'sum_dv_los': np.zeros(nbins, dtype=np.float64),
        'sum_dv_los2': np.zeros(nbins, dtype=np.float64),
        'sum_abs_dv_los': np.zeros(nbins, dtype=np.float64),
        'sum_vrad': np.zeros(nbins, dtype=np.float64),
        'sum_vrad2': np.zeros(nbins, dtype=np.float64),
    }


def accumulate(acc, bin_idx, dv_los, v_rad):
    for ib in np.unique(bin_idx):
        m = bin_idx == ib
        acc['count'][ib] += int(m.sum())
        x = dv_los[m]
        y = v_rad[m]
        acc['sum_dv_los'][ib] += float(x.sum())
        acc['sum_dv_los2'][ib] += float((x * x).sum())
        acc['sum_abs_dv_los'][ib] += float(np.abs(x).sum())
        acc['sum_vrad'][ib] += float(y.sum())
        acc['sum_vrad2'][ib] += float((y * y).sum())


def finalize(acc, bins):
    count = acc['count'].astype(np.float64)
    out = {'r': np.sqrt(bins[:-1] * bins[1:]), 'count': acc['count']}
    ok = count > 0
    for key in ('dv_los', 'vrad'):
        s = acc[f'sum_{key}']
        s2 = acc[f'sum_{key}2']
        mean = np.full_like(count, np.nan, dtype=np.float64)
        sig = np.full_like(count, np.nan, dtype=np.float64)
        mean[ok] = s[ok] / count[ok]
        var = np.full_like(count, np.nan, dtype=np.float64)
        var[ok] = np.maximum(s2[ok] / count[ok] - mean[ok] ** 2, 0.0)
        sig[ok] = np.sqrt(var[ok])
        out[f'mean_{key}'] = mean
        out[f'sigma_{key}'] = sig
    out['mean_abs_dv_los'] = np.full_like(count, np.nan, dtype=np.float64)
    out['mean_abs_dv_los'][ok] = acc['sum_abs_dv_los'][ok] / count[ok]
    return out


def pairwise_velocity_stats(pos: np.ndarray, vel: np.ndarray, BoxSize: float,
                            bins: np.ndarray, los_axis: int,
                            n_anchors: int, max_neighbors: int, seed: int):
    rng = np.random.default_rng(seed)
    pos = np.mod(np.asarray(pos, dtype=np.float64), BoxSize)
    vel = np.asarray(vel, dtype=np.float64)
    tree = cKDTree(pos, boxsize=BoxSize)
    rmax = float(bins[-1])
    anchors = np.arange(len(pos))
    if anchors.size > n_anchors:
        anchors = rng.choice(anchors, size=n_anchors, replace=False)

    acc = init_accumulators(len(bins) - 1)
    for ia in anchors:
        neigh = np.asarray(tree.query_ball_point(pos[ia], rmax), dtype=np.int64)
        neigh = neigh[neigh != ia]
        if neigh.size == 0:
            continue
        if neigh.size > max_neighbors:
            neigh = rng.choice(neigh, size=max_neighbors, replace=False)

        rvec = pos[neigh] - pos[ia]
        rvec -= BoxSize * np.round(rvec / BoxSize)
        r = np.linalg.norm(rvec, axis=1)
        valid = (r >= bins[0]) & (r < bins[-1]) & np.isfinite(r)
        if not np.any(valid):
            continue
        rvec = rvec[valid]
        r = r[valid]
        neigh = neigh[valid]

        dv = vel[neigh] - vel[ia]
        dv_los = dv[:, los_axis]
        rhat = rvec / r[:, None]
        v_rad = np.sum(dv * rhat, axis=1)
        bin_idx = np.searchsorted(bins, r, side='right') - 1
        accumulate(acc, bin_idx, dv_los, v_rad)

    return finalize(acc, bins)


def safe_ratio(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def save_stats_npz(path, stats):
    payload = {}
    for prefix, item in stats.items():
        for key, val in item.items():
            payload[f'{prefix}_{key}'] = val
    np.savez(path, **payload)


def plot_stats(stats, output_path, title):
    rows = 2 if 'mock_resid' in stats else 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.3 * rows),
                             constrained_layout=True, squeeze=False)

    for row, field in enumerate(['vel', 'resid'][:rows]):
        true = stats[f'true_{field}']
        mock = stats[f'mock_{field}']
        r = true['r']

        ax = axes[row, 0]
        ax.loglog(r, true['sigma_dv_los'], '-o', ms=3, label='True')
        ax.loglog(r, mock['sigma_dv_los'], '-o', ms=3, label='Mock')
        ax.set_xlabel(r'$r$ [$h^{-1}{\rm Mpc}$]')
        ax.set_ylabel(r'$\sigma[\Delta v_{\rm los}]$ [km/s]')
        ax.set_title(f'{field}: pairwise LOS dispersion')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend()

        ax = axes[row, 1]
        ax.semilogx(r, safe_ratio(mock['sigma_dv_los'], true['sigma_dv_los']),
                    '-o', ms=3, color='k')
        ax.axhline(1.0, ls='--', color='0.3')
        ax.axhline(0.9, ls=':', color='0.4')
        ax.axhline(1.1, ls=':', color='0.4')
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel(r'$r$ [$h^{-1}{\rm Mpc}$]')
        ax.set_ylabel('mock / true')
        ax.set_title(f'{field}: LOS dispersion ratio')
        ax.grid(True, which='both', alpha=0.25)

        ax = axes[row, 2]
        ax.semilogx(r, true['mean_vrad'], '-o', ms=3, label='True')
        ax.semilogx(r, mock['mean_vrad'], '-o', ms=3, label='Mock')
        ax.axhline(0.0, ls='--', color='0.3')
        ax.set_xlabel(r'$r$ [$h^{-1}{\rm Mpc}$]')
        ax.set_ylabel(r'$\langle \Delta{\bf v}\cdot\hat{\bf r}\rangle$ [km/s]')
        ax.set_title(f'{field}: mean radial pairwise velocity')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.savefig(output_path + '.png', dpi=170, bbox_inches='tight')
    fig.savefig(output_path + '.pdf', dpi=170, bbox_inches='tight')
    plt.close(fig)


def print_small_scale_summary(stats, rcut=10.0):
    for field in ['vel', 'resid']:
        if f'true_{field}' not in stats:
            continue
        t = stats[f'true_{field}']
        m = stats[f'mock_{field}']
        sel = t['r'] <= rcut
        if not np.any(sel):
            continue
        ratio = safe_ratio(m['sigma_dv_los'][sel], t['sigma_dv_los'][sel])
        diff_vrad = m['mean_vrad'][sel] - t['mean_vrad'][sel]
        print(
            f'{field:>5s}  r<={rcut:g} Mpc/h: '
            f'median sigma_los ratio={np.nanmedian(ratio):.3f}, '
            f'median mean_vrad(mock-true)={np.nanmedian(diff_vrad):+.1f} km/s',
            flush=True,
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sc = cfg['sim_settings']
    dc = cfg['data_settings']
    z_snap = str(args.z_snap or dc['z_snap'])
    mock_path, true_halo_dir, output_dir = resolve_paths(cfg, args)

    print(f'Loading mock: {mock_path}', flush=True)
    mock, meta = load_mock(mock_path)
    print(f'Loading truth: {true_halo_dir}/{args.sim_id}/', flush=True)
    true_cat, _ = load_true(true_halo_dir, args.sim_id, z_snap)
    BoxSize = float(meta['BoxSize'])

    mock_s = maybe_subsample(mock, args.max_halos, args.seed + 1, args.mass_min)
    true_s = maybe_subsample(true_cat, args.max_halos, args.seed + 2, args.mass_min)
    print(f'Using {len(mock_s["pos"])} mock halos and '
          f'{len(true_s["pos"])} true halos.', flush=True)

    bins = np.geomspace(args.rmin, args.rmax, args.nbins + 1)
    stats = {
        'mock_vel': pairwise_velocity_stats(
            mock_s['pos'], mock_s['vel'], BoxSize, bins, args.los_axis,
            args.n_anchors, args.max_neighbors, args.seed + 10),
        'true_vel': pairwise_velocity_stats(
            true_s['pos'], true_s['vel'], BoxSize, bins, args.los_axis,
            args.n_anchors, args.max_neighbors, args.seed + 20),
    }

    if not args.skip_residuals:
        print('Computing PM-velocity residual diagnostics...', flush=True)
        vel_pm = load_fastpm_velocity(dc['fastpm_dir'], args.sim_id,
                                      int(sc['ns_d']), z_snap)
        interps = build_velocity_interpolators(vel_pm, BoxSize)
        mock_resid = pm_velocity_at(mock_s['pos'], interps) - mock_s['vel']
        true_resid = pm_velocity_at(true_s['pos'], interps) - true_s['vel']
        stats['mock_resid'] = pairwise_velocity_stats(
            mock_s['pos'], mock_resid, BoxSize, bins, args.los_axis,
            args.n_anchors, args.max_neighbors, args.seed + 30)
        stats['true_resid'] = pairwise_velocity_stats(
            true_s['pos'], true_resid, BoxSize, bins, args.los_axis,
            args.n_anchors, args.max_neighbors, args.seed + 40)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f'_lgMmin{args.mass_min:.2f}' if args.mass_min is not None else ''
    out_base = os.path.join(
        output_dir, f'pairwise_velocity_diagnostics_sim{args.sim_id:04d}{suffix}')
    save_stats_npz(out_base + '.npz', stats)
    plot_stats(stats, out_base, f'Pairwise velocity diagnostics: sim {args.sim_id:04d}')
    print_small_scale_summary(stats)
    print(f'Saved: {out_base}.png', flush=True)
    print(f'Saved: {out_base}.pdf', flush=True)
    print(f'Saved: {out_base}.npz', flush=True)


if __name__ == '__main__':
    main()
