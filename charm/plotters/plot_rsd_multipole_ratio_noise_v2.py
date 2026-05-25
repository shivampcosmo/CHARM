#!/usr/bin/env python
"""
Plot aggregate RSD multipole ratios from saved CHARM mock catalogs.

This script is the plotting/noising half of the modular RSD workflow:
  1. run_inference_catalog_range_v2.py writes clean mock_catalog_simXXXX.npz
     files with no extra RSD-position noise.
  2. this script loads those mocks, loads truth, applies requested anisotropic
     Gaussian noise to redshift-space positions, computes unweighted and
     mass-weighted P0/P2/P4 ratios, equilateral B0(k,k,k) bispectrum ratios,
     and HMF ratios for each simulation, then plots individual simulations plus
     mean and 16-84% bands.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config
from plotters.plot_inference_v2 import (
    _density_field,
    apply_rsd,
    halo_mass_function,
    load_mock,
    load_true,
)


PANELS = OrderedDict([
    ('p0_unweighted', {
        'title': r'Unweighted RSD $P_0$',
        'ylabel': r'$P_{0,\rm mock}/P_{0,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('p2_unweighted', {
        'title': r'Unweighted RSD $P_2$',
        'ylabel': r'$P_{2,\rm mock}/P_{2,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('p4_unweighted', {
        'title': r'Unweighted RSD $P_4$',
        'ylabel': r'$P_{4,\rm mock}/P_{4,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('p0_weighted', {
        'title': r'Mass-weighted RSD $P_0$',
        'ylabel': r'$P^{w}_{0,\rm mock}/P^{w}_{0,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('p2_weighted', {
        'title': r'Mass-weighted RSD $P_2$',
        'ylabel': r'$P^{w}_{2,\rm mock}/P^{w}_{2,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('p4_weighted', {
        'title': r'Mass-weighted RSD $P_4$',
        'ylabel': r'$P^{w}_{4,\rm mock}/P^{w}_{4,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('bk_unweighted', {
        'title': r'Unweighted RSD $B_0(k,k,k)$',
        'ylabel': r'$B_{0,\rm mock}/B_{0,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('bk_weighted', {
        'title': r'Mass-weighted RSD $B_0(k,k,k)$',
        'ylabel': r'$B^{w}_{0,\rm mock}/B^{w}_{0,\rm true}$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'xscale': 'log',
    }),
    ('hmf', {
        'title': r'Halo mass function',
        'ylabel': r'$(dn/d\log M)_{\rm mock}/(dn/d\log M)_{\rm true}$',
        'xlabel': r'$\log_{10} M$ [$M_\odot/h$]',
        'xscale': 'linear',
    }),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML training config.')
    p.add_argument('--mock_dir', default=None,
                   help='Directory with mock_catalog_simXXXX.npz files. Defaults '
                        'to <checkpoint_dir>/inference/.')
    p.add_argument('--true_halo_dir', default=None,
                   help='Directory with true halo HDF5 files. Defaults to config.')
    p.add_argument('--sim_start', type=int, default=None,
                   help='First simulation id. Defaults to nsims_train + nsims_val.')
    p.add_argument('--sim_end', type=int, default=None,
                   help='Exclusive upper simulation id. Defaults to sim_start + nsims_test.')
    p.add_argument('--sim_ids', nargs='*', type=int, default=None,
                   help='Explicit simulation ids. Overrides --sim_start/--sim_end.')
    p.add_argument('--output_dir', default=None,
                   help='Directory for figures and arrays. Defaults to --mock_dir.')
    p.add_argument('--output_name', default='rsd_multipole_ratio_noise_summary',
                   help='Output basename without extension.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing .png/.pdf/.npz outputs. Without this, '
                        'the script exits before processing if any target output exists.')
    p.add_argument('--workers', type=int, default=25,
                   help='Number of simulations to process concurrently.')
    p.add_argument('--ng', type=int, default=384,
                   help='Grid size for Pylians multipoles.')
    p.add_argument('--kmax', type=float, default=0.4,
                   help='Maximum k for multipole curves.')
    p.add_argument('--mas', default='TSC',
                   choices=('NGP', 'CIC', 'TSC', 'PCS'),
                   help='Mass-assignment scheme for Pylians density fields.')
    p.add_argument('--mass_weight_pivot', type=float, default=1.0e14,
                   help='Pivot mass for weighted statistics [Msun/h].')
    p.add_argument('--mass_weight_alpha', type=float, default=0.7,
                   help='Exponent for weighted statistics: w=(M/pivot)^alpha.')
    p.add_argument('--hmf_bins', type=int, default=20,
                   help='Number of log-mass bins for the HMF ratio.')
    p.add_argument('--bk_ng', type=int, default=None,
                   help='Grid size for Pylians bispectrum. Defaults to --ng.')
    p.add_argument('--bk_kmin', type=float, default=None,
                   help='Minimum k for equilateral bispectrum ratios. Defaults '
                        'to max(0.04, 4*k_fundamental).')
    p.add_argument('--bk_kmax', type=float, default=None,
                   help='Maximum k for equilateral bispectrum ratios. Defaults '
                        'to --kmax, clipped below the grid Nyquist frequency.')
    p.add_argument('--bk_num_bins', type=int, default=10,
                   help='Number of log-spaced equilateral bispectrum k values.')
    p.add_argument('--bk_threads', type=int, default=10,
                   help='OpenMP threads used inside each Pylians Bk call.')
    p.add_argument('--z_snap', default=None,
                   help='Redshift string in true HDF5 filename. Defaults to config.')
    p.add_argument('--los_axis', type=int, choices=(0, 1, 2), default=2,
                   help='Line-of-sight axis for RSD and multipoles.')
    p.add_argument('--mock_rsd_noise_sigma_los', type=float, default=2.0,
                   help='Gaussian RSD-position noise sigma along LOS for mock [Mpc/h].')
    p.add_argument('--mock_rsd_noise_sigma_transverse', type=float, default=0.0,
                   help='Gaussian RSD-position noise sigma in each transverse '
                        'axis for mock [Mpc/h].')
    p.add_argument('--truth_rsd_noise_sigma_los', type=float, default=0.0,
                   help='Gaussian RSD-position noise sigma along LOS for truth [Mpc/h].')
    p.add_argument('--truth_rsd_noise_sigma_transverse', type=float, default=0.0,
                   help='Gaussian RSD-position noise sigma in each transverse '
                        'axis for truth [Mpc/h].')
    p.add_argument('--noise_seed', type=int, default=None,
                   help='Base random seed for RSD-position noise. Defaults to sim_id per sim.')
    p.add_argument('--ylim', nargs=2, type=float, default=(0.5, 1.5),
                   metavar=('YMIN', 'YMAX'),
                   help='Y-limits for all ratio panels.')
    p.add_argument('--auto_ylim', action='store_true',
                   help='Use data-driven y-limits per panel instead of --ylim.')
    return p.parse_args()


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def default_sim_ids(cfg: dict, sim_start: int | None,
                    sim_end: int | None) -> list[int]:
    sc = cfg['sim_settings']
    if sim_start is None:
        sim_start = int(sc.get('nsims_train', 1800)) + int(sc.get('nsims_val', 100))
    if sim_end is None:
        sim_end = sim_start + int(sc.get('nsims_test', 100))
    if sim_end <= sim_start:
        raise ValueError(f'Empty sim range: [{sim_start}, {sim_end})')
    return list(range(sim_start, sim_end))


def resolve_paths(cfg: dict, args):
    ckpt_dir = os.path.join(_REPO_ROOT, cfg['train_settings']['checkpoint_dir'])
    mock_dir = os.path.abspath(args.mock_dir or os.path.join(ckpt_dir, 'inference'))
    true_halo_dir = os.path.abspath(repo_path(
        args.true_halo_dir or cfg['data_settings']['halo_hdf5_dir']))
    output_dir = os.path.abspath(args.output_dir or mock_dir)
    return mock_dir, true_halo_dir, output_dir


def validate_sigma(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f'{name} must be non-negative; got {value}')
    return value


def resolve_noise_config(args) -> dict[str, float]:
    return {
        'mock_los': validate_sigma(
            'mock_rsd_noise_sigma_los', args.mock_rsd_noise_sigma_los),
        'mock_transverse': validate_sigma(
            'mock_rsd_noise_sigma_transverse',
            args.mock_rsd_noise_sigma_transverse),
        'truth_los': validate_sigma(
            'truth_rsd_noise_sigma_los', args.truth_rsd_noise_sigma_los),
        'truth_transverse': validate_sigma(
            'truth_rsd_noise_sigma_transverse',
            args.truth_rsd_noise_sigma_transverse),
    }


def apply_position_noise(pos_rsd: np.ndarray, sigma_los: float,
                         sigma_transverse: float, BoxSize: float,
                         rng: np.random.Generator, los_axis: int) -> np.ndarray:
    pos_noisy = np.asarray(pos_rsd, dtype=np.float64).copy()
    if pos_noisy.size == 0:
        return pos_noisy.astype(np.float32)
    noise = np.zeros_like(pos_noisy, dtype=np.float64)
    if sigma_los > 0.0:
        noise[:, los_axis] = rng.normal(0.0, sigma_los, size=pos_noisy.shape[0])
    if sigma_transverse > 0.0:
        transverse_axes = [axis for axis in range(3) if axis != los_axis]
        noise[:, transverse_axes] = rng.normal(
            0.0, sigma_transverse,
            size=(pos_noisy.shape[0], len(transverse_axes)),
        )
    return np.mod(pos_noisy + noise, BoxSize).astype(np.float32)


def mass_weights(lgM: np.ndarray, pivot: float, alpha: float) -> np.ndarray:
    pivot = float(pivot)
    alpha = float(alpha)
    if pivot <= 0.0:
        raise ValueError(f'mass_weight_pivot must be positive; got {pivot}')
    if not np.isfinite(alpha):
        raise ValueError(f'mass_weight_alpha must be finite; got {alpha}')

    lgM = np.asarray(lgM, dtype=np.float64)
    weights = np.power(np.power(10.0, lgM) / pivot, alpha)
    weights[~np.isfinite(weights)] = 0.0
    return weights.astype(np.float32)


def safe_ratio(num: np.ndarray, den: np.ndarray,
               floor_frac: float = 1e-4) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    finite = den[np.isfinite(den)]
    if finite.size == 0:
        return np.full_like(num, np.nan, dtype=np.float64)
    floor = floor_frac * np.nanmax(np.abs(finite))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(den) > floor, num / den, np.nan)
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def interp_ratio(k_num, p_num, k_den, p_den, floor_frac: float = 1e-4):
    k_num = np.asarray(k_num, dtype=np.float64)
    p_num = np.asarray(p_num, dtype=np.float64)
    k_den = np.asarray(k_den, dtype=np.float64)
    p_den = np.asarray(p_den, dtype=np.float64)
    if k_num.size == 0:
        return k_num, np.asarray([], dtype=np.float64)
    if k_den.size == 0:
        return k_num, np.full(k_num.shape, np.nan, dtype=np.float64)
    p_den_i = np.interp(k_num, k_den, p_den, left=np.nan, right=np.nan)
    return k_num, safe_ratio(p_num, p_den_i, floor_frac=floor_frac)


def multipoles_from_delta(delta: np.ndarray | None, BoxSize: float, kmax: float,
                          MAS: str, axis: int):
    """Compute Pylians P0/P2/P4 from an already painted overdensity field."""
    if delta is None:
        z = np.asarray([], dtype=np.float64)
        return z, z, z, z

    import Pk_library as PKL

    Pk_obj = PKL.Pk(delta, np.float32(BoxSize), axis=axis, MAS=MAS,
                    threads=10, verbose=False)
    k = np.asarray(Pk_obj.k3D, dtype=np.float64)
    P0 = np.asarray(Pk_obj.Pk[:, 0], dtype=np.float64)
    P2 = np.asarray(Pk_obj.Pk[:, 1], dtype=np.float64)
    P4 = np.asarray(Pk_obj.Pk[:, 2], dtype=np.float64)
    sel = (k > 0) & (k <= kmax) & np.isfinite(P0)
    return k[sel], P0[sel], P2[sel], P4[sel]


def bispectrum_k_values(BoxSize: float, Ng: int, kmin: float | None,
                        kmax: float | None, fallback_kmax: float,
                        n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError(f'bk_num_bins must be positive; got {n_bins}')
    if Ng <= 0:
        raise ValueError(f'bk_ng must be positive; got {Ng}')

    k_fund = 2.0 * np.pi / float(BoxSize)
    kmin_eff = max(4.0 * k_fund, 0.04 if kmin is None else float(kmin))
    kmax_eff = float(fallback_kmax if kmax is None else kmax)
    nyquist = np.pi * float(Ng) / float(BoxSize)
    kmax_eff = min(kmax_eff, 0.85 * nyquist)
    if kmin_eff <= 0.0 or kmax_eff <= kmin_eff:
        raise ValueError(
            f'Invalid bispectrum k range: kmin={kmin_eff:g}, '
            f'kmax={kmax_eff:g}, Ng={Ng}, BoxSize={BoxSize:g}'
        )
    if n_bins == 1:
        return np.asarray([kmin_eff], dtype=np.float64)
    return np.geomspace(kmin_eff, kmax_eff, n_bins).astype(np.float64)


def equilateral_bispectrum_monopole(delta: np.ndarray | None,
                                    BoxSize: float,
                                    k_values: np.ndarray,
                                    MAS: str,
                                    threads: int) -> np.ndarray:
    """
    Return Pylians orientation-averaged B(k,k,k) from an RSD density field.

    Pylians parameterizes triangles by k1, k2, and the angle between them.
    For the convention used by PKL.Bk, equilateral triangles have
    theta=2*pi/3, giving k3=k when k1=k2=k.
    """
    k_values = np.asarray(k_values, dtype=np.float64)
    if delta is None or k_values.size == 0:
        return np.full(k_values.shape, np.nan, dtype=np.float64)

    import Pk_library as PKL

    theta = np.asarray([2.0 * np.pi / 3.0], dtype=np.float32)
    out = np.full(k_values.shape, np.nan, dtype=np.float64)
    threads = max(1, int(threads))
    with open(os.devnull, 'w') as devnull:
        for i, kval in enumerate(k_values):
            try:
                with contextlib.redirect_stdout(devnull):
                    bbk = PKL.Bk(
                        delta,
                        np.float32(BoxSize),
                        np.float32(kval),
                        np.float32(kval),
                        theta,
                        MAS=MAS,
                        threads=threads,
                    )
                b_arr = np.asarray(bbk.B, dtype=np.float64).reshape(-1)
                if b_arr.size:
                    out[i] = b_arr[0]
            except (FloatingPointError, ZeroDivisionError, ValueError):
                out[i] = np.nan
    return out


def compute_one(payload):
    (sim_id, mock_dir, true_halo_dir, z_snap, ng, kmax, los_axis,
     noise_cfg, noise_seed_base, lh_cosmo_file, MAS, hmf_bins,
     mass_weight_pivot, mass_weight_alpha, bk_ng, bk_kmin, bk_kmax,
     bk_num_bins, bk_threads) = payload

    mock_path = os.path.join(mock_dir, f'mock_catalog_sim{sim_id:04d}.npz')
    if not os.path.exists(mock_path):
        raise FileNotFoundError(f'Missing mock catalog for sim {sim_id}: {mock_path}')

    mock, meta = load_mock(mock_path)
    true_cat, _ = load_true(true_halo_dir, sim_id, z_snap)
    BoxSize = float(meta['BoxSize'])
    z = float(meta['z'])
    lgMmin = float(meta['lgMmin'])
    lgMmax = float(meta['lgMmax'])
    cosmo = np.loadtxt(lh_cosmo_file)[sim_id].astype(np.float32)

    pos_rsd_m = apply_rsd(mock['pos'], mock['vel'], BoxSize, z, cosmo,
                          axis=los_axis)
    pos_rsd_t = apply_rsd(true_cat['pos'], true_cat['vel'], BoxSize, z, cosmo,
                          axis=los_axis)

    seed = int(sim_id if noise_seed_base is None else noise_seed_base + sim_id)
    rng_mock = np.random.default_rng(seed + 101)
    rng_truth = np.random.default_rng(seed + 202)
    pos_rsd_m = apply_position_noise(
        pos_rsd_m, noise_cfg['mock_los'], noise_cfg['mock_transverse'],
        BoxSize, rng_mock, los_axis)
    pos_rsd_t = apply_position_noise(
        pos_rsd_t, noise_cfg['truth_los'], noise_cfg['truth_transverse'],
        BoxSize, rng_truth, los_axis)

    k_bk = bispectrum_k_values(
        BoxSize, bk_ng, bk_kmin, bk_kmax, kmax, bk_num_bins)

    stats = {}

    def _add_power_and_bispectrum(kind: str, weights_m, weights_t):
        delta_m = _density_field(pos_rsd_m, weights_m, ng, BoxSize, MAS=MAS)
        delta_t = _density_field(pos_rsd_t, weights_t, ng, BoxSize, MAS=MAS)

        k_m, p0_m, p2_m, p4_m = multipoles_from_delta(
            delta_m, BoxSize, kmax, MAS, los_axis)
        k_t, p0_t, p2_t, p4_t = multipoles_from_delta(
            delta_t, BoxSize, kmax, MAS, los_axis)

        stats[f'p0_{kind}'] = interp_ratio(k_m, p0_m, k_t, p0_t)
        stats[f'p2_{kind}'] = interp_ratio(k_m, p2_m, k_t, p2_t)
        stats[f'p4_{kind}'] = interp_ratio(k_m, p4_m, k_t, p4_t)

        if bk_ng == ng:
            delta_m_bk = delta_m
            delta_t_bk = delta_t
        else:
            delta_m_bk = _density_field(
                pos_rsd_m, weights_m, bk_ng, BoxSize, MAS=MAS)
            delta_t_bk = _density_field(
                pos_rsd_t, weights_t, bk_ng, BoxSize, MAS=MAS)

        b_m = equilateral_bispectrum_monopole(
            delta_m_bk, BoxSize, k_bk, MAS, bk_threads)
        b_t = equilateral_bispectrum_monopole(
            delta_t_bk, BoxSize, k_bk, MAS, bk_threads)
        stats[f'bk_{kind}'] = (
            k_bk,
            safe_ratio(b_m, b_t, floor_frac=1e-5),
        )

    _add_power_and_bispectrum('unweighted', None, None)

    weights_m = mass_weights(mock['lgM'], mass_weight_pivot, mass_weight_alpha)
    weights_t = mass_weights(true_cat['lgM'], mass_weight_pivot, mass_weight_alpha)
    _add_power_and_bispectrum('weighted', weights_m, weights_t)

    x_hmf_m, y_hmf_m = halo_mass_function(
        mock['lgM'], lgMmin, lgMmax, n_bins=hmf_bins, BoxSize=BoxSize)
    x_hmf_t, y_hmf_t = halo_mass_function(
        true_cat['lgM'], lgMmin, lgMmax, n_bins=hmf_bins, BoxSize=BoxSize)
    stats['hmf'] = interp_ratio(
        x_hmf_m, y_hmf_m, x_hmf_t, y_hmf_t, floor_frac=0.0)

    return sim_id, stats


def stack_results(results: list[tuple[int, dict]]):
    stacked = OrderedDict()
    for key in PANELS:
        entries = []
        x_ref = None
        sim_ids = []
        for sim_id, stats in results:
            x, ratio = stats[key]
            x = np.asarray(x, dtype=np.float64)
            ratio = np.asarray(ratio, dtype=np.float64)
            entries.append((sim_id, x, ratio))
            sim_ids.append(sim_id)
            if x_ref is None and x.size > 0:
                x_ref = x
        if x_ref is None:
            x_ref = np.asarray([], dtype=np.float64)

        rows = []
        for _, x, ratio in entries:
            if x_ref.size == 0:
                ratio = np.asarray([], dtype=np.float64)
            elif x.size == 0:
                ratio = np.full(x_ref.shape, np.nan, dtype=np.float64)
            elif x.size != x_ref.size or not np.allclose(x, x_ref, equal_nan=True):
                ratio = np.interp(x_ref, x, ratio, left=np.nan, right=np.nan)
            rows.append(ratio)
        stacked[key] = (x_ref, np.vstack(rows), np.asarray(sim_ids))
    return stacked


def nan_summary(rows: np.ndarray):
    if rows.size == 0 or rows.shape[-1] == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(rows, axis=0)
        p16 = np.nanpercentile(rows, 16, axis=0)
        p84 = np.nanpercentile(rows, 84, axis=0)
    return mean, p16, p84


def set_auto_ylim(ax, rows, mean, p16, p84):
    vals = np.concatenate([
        rows[np.isfinite(rows)],
        mean[np.isfinite(mean)],
        p16[np.isfinite(p16)],
        p84[np.isfinite(p84)],
        np.asarray([0.9, 1.0, 1.1]),
    ])
    if vals.size == 0:
        ax.set_ylim(0.5, 1.5)
        return
    lo, hi = np.nanpercentile(vals, [1, 99])
    pad = 0.08 * max(hi - lo, 0.2)
    ax.set_ylim(lo - pad, hi + pad)


def make_plot(stacked: OrderedDict, sim_ids: list[int], output_dir: str,
              output_name: str, noise_cfg: dict[str, float],
              los_axis: int, ylim, auto_ylim: bool,
              mass_weight_pivot: float, mass_weight_alpha: float,
              bk_ng: int, bk_num_bins: int):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

    fig, axes = plt.subplots(3, 3, figsize=(17.2, 13.4),
                             constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (key, spec) in zip(axes_flat, PANELS.items()):
        x, rows, _ = stacked[key]
        mean, p16, p84 = nan_summary(rows)

        ax.axhspan(0.9, 1.1, color='0.88', alpha=0.65, zorder=0,
                   label=r'$\pm 10\%$')
        if x.size:
            for row in rows:
                if spec['xscale'] == 'log':
                    ax.semilogx(x, row, color='0.42', alpha=0.14, lw=0.75,
                                zorder=1)
                else:
                    ax.plot(x, row, color='0.42', alpha=0.14, lw=0.75,
                            zorder=1)
            ax.fill_between(x, p16, p84, color='0.45', alpha=0.24,
                            linewidth=0, zorder=2,
                            label='16th-84th percentile')
            if spec['xscale'] == 'log':
                ax.semilogx(x, mean, color='k', lw=2.2, zorder=3,
                            label='mean')
            else:
                ax.plot(x, mean, color='k', lw=2.2, zorder=3, label='mean')
        ax.axhline(1.0, color='k', ls='--', lw=1.7, zorder=4)
        ax.axhline(0.9, color='0.25', ls='--', lw=0.8, zorder=4)
        ax.axhline(1.1, color='0.25', ls='--', lw=0.8, zorder=4)

        ax.set_title(spec['title'])
        ax.set_xlabel(spec['xlabel'])
        ax.set_ylabel(spec['ylabel'])
        ax.grid(True, which='major', color='0.88', lw=0.8)
        ax.grid(True, which='minor', color='0.93', lw=0.45)
        ax.tick_params(axis='both', which='major', length=5.5, width=1.0,
                       labelsize=10, direction='out')
        ax.tick_params(axis='both', which='minor', length=3.2, width=0.8,
                       direction='out')
        if auto_ylim:
            set_auto_ylim(ax, rows, mean, p16, p84)
        else:
            ax.set_ylim(*ylim)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.035))

    axis_name = 'xyz'[los_axis]
    fig.suptitle(
        f'RSD power, equilateral bispectrum, and HMF ratios: {len(sim_ids)} simulations '
        f'({min(sim_ids):04d}-{max(sim_ids):04d}) | '
        f'mock LOS/T={noise_cfg["mock_los"]:g}/{noise_cfg["mock_transverse"]:g} Mpc/h, '
        f'truth LOS/T={noise_cfg["truth_los"]:g}/{noise_cfg["truth_transverse"]:g} Mpc/h | '
        f'LOS={axis_name} | '
        f'$w=(M/{mass_weight_pivot:.1e})^{{{mass_weight_alpha:g}}}$ | '
        f'$B_0$: {bk_num_bins} equilateral bins, Ng={bk_ng}',
        y=1.055,
        fontsize=14,
        fontweight='bold',
    )

    pdf_path = os.path.join(output_dir, f'{output_name}.pdf')
    png_path = os.path.join(output_dir, f'{output_name}.png')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=180)
    fig.savefig(png_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    return pdf_path, png_path


def output_paths(output_dir: str, output_name: str) -> tuple[str, str, str]:
    return (
        os.path.join(output_dir, f'{output_name}.pdf'),
        os.path.join(output_dir, f'{output_name}.png'),
        os.path.join(output_dir, f'{output_name}.npz'),
    )


def format_noise_value(value: float) -> str:
    text = f'{float(value):.6g}'
    return text.replace('-', 'm').replace('+', '').replace('.', 'p')


def add_noise_suffix(output_name: str, noise_cfg: dict[str, float],
                     los_axis: int) -> str:
    return (
        f'{output_name}'
        f'_mockLOS{format_noise_value(noise_cfg["mock_los"])}'
        f'_mockT{format_noise_value(noise_cfg["mock_transverse"])}'
        f'_truthLOS{format_noise_value(noise_cfg["truth_los"])}'
        f'_truthT{format_noise_value(noise_cfg["truth_transverse"])}'
        f'_los{"xyz"[los_axis]}'
    )


def save_arrays(path: str, stacked: OrderedDict, sim_ids: list[int],
                noise_cfg: dict[str, float], los_axis: int,
                ng: int, kmax: float, MAS: str, hmf_bins: int,
                mass_weight_pivot: float, mass_weight_alpha: float,
                bk_ng: int, bk_kmin: float | None, bk_kmax: float | None,
                bk_num_bins: int):
    payload = {
        'sim_ids': np.asarray(sim_ids, dtype=np.int32),
        'mock_rsd_noise_sigma_los': np.float32(noise_cfg['mock_los']),
        'mock_rsd_noise_sigma_transverse': np.float32(noise_cfg['mock_transverse']),
        'truth_rsd_noise_sigma_los': np.float32(noise_cfg['truth_los']),
        'truth_rsd_noise_sigma_transverse': np.float32(noise_cfg['truth_transverse']),
        'los_axis': np.int32(los_axis),
        'ng': np.int32(ng),
        'kmax': np.float32(kmax),
        'mas': np.asarray(MAS),
        'hmf_bins': np.int32(hmf_bins),
        'mass_weight_pivot': np.float64(mass_weight_pivot),
        'mass_weight_alpha': np.float64(mass_weight_alpha),
        'bk_ng': np.int32(bk_ng),
        'bk_kmin_arg': np.nan if bk_kmin is None else np.float64(bk_kmin),
        'bk_kmax_arg': np.nan if bk_kmax is None else np.float64(bk_kmax),
        'bk_num_bins': np.int32(bk_num_bins),
        'bispectrum_shape': np.asarray('equilateral'),
    }
    for key in PANELS:
        x, rows, _ = stacked[key]
        mean, p16, p84 = nan_summary(rows)
        payload[f'{key}_x'] = x
        if key == 'hmf':
            payload[f'{key}_lgM'] = x
        else:
            payload[f'{key}_k'] = x
        payload[f'{key}_rows'] = rows
        payload[f'{key}_mean'] = mean
        payload[f'{key}_p16'] = p16
        payload[f'{key}_p84'] = p84
    np.savez(path, **payload)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dc = cfg['data_settings']
    sim_ids = args.sim_ids if args.sim_ids else default_sim_ids(
        cfg, args.sim_start, args.sim_end)
    mock_dir, true_halo_dir, output_dir = resolve_paths(cfg, args)
    z_snap = str(args.z_snap or dc['z_snap'])
    noise_cfg = resolve_noise_config(args)
    lh_cosmo_file = repo_path(dc['lh_cosmo_file'])
    ng = int(args.ng)
    bk_ng = int(args.bk_ng or args.ng)
    kmax = float(args.kmax)
    if ng <= 0:
        raise ValueError(f'ng must be positive; got {ng}')
    if bk_ng <= 0:
        raise ValueError(f'bk_ng must be positive; got {bk_ng}')
    if kmax <= 0.0:
        raise ValueError(f'kmax must be positive; got {kmax}')
    if int(args.hmf_bins) <= 0:
        raise ValueError(f'hmf_bins must be positive; got {args.hmf_bins}')
    if int(args.bk_num_bins) <= 0:
        raise ValueError(f'bk_num_bins must be positive; got {args.bk_num_bins}')
    if float(args.mass_weight_pivot) <= 0.0:
        raise ValueError(
            f'mass_weight_pivot must be positive; got {args.mass_weight_pivot}')
    os.makedirs(output_dir, exist_ok=True)
    output_name = add_noise_suffix(args.output_name, noise_cfg, int(args.los_axis))

    targets = output_paths(output_dir, output_name)
    existing = [path for path in targets if os.path.exists(path)]
    if existing and not args.overwrite:
        existing_str = '\n  '.join(existing)
        raise SystemExit(
            'Output file(s) already exist. Use --overwrite to replace them:\n'
            f'  {existing_str}'
        )

    payloads = []
    for sim_id in sim_ids:
        payloads.append((
            sim_id,
            mock_dir,
            true_halo_dir,
            z_snap,
            ng,
            kmax,
            int(args.los_axis),
            noise_cfg,
            args.noise_seed,
            lh_cosmo_file,
            args.mas,
            int(args.hmf_bins),
            float(args.mass_weight_pivot),
            float(args.mass_weight_alpha),
            bk_ng,
            args.bk_kmin,
            args.bk_kmax,
            int(args.bk_num_bins),
            int(args.bk_threads),
        ))

    print(f'Mock catalog directory: {mock_dir}', flush=True)
    print(f'True halo directory: {true_halo_dir}', flush=True)
    print(
        'RSD position noise [Mpc/h]: '
        f'mock LOS={noise_cfg["mock_los"]:g}, '
        f'mock transverse={noise_cfg["mock_transverse"]:g}; '
        f'truth LOS={noise_cfg["truth_los"]:g}, '
        f'truth transverse={noise_cfg["truth_transverse"]:g}',
        flush=True,
    )
    print(
        f'Statistics: P multipoles Ng={ng}, MAS={args.mas}, kmax={kmax:g}; '
        f'equilateral B0 Ng={bk_ng}, bins={int(args.bk_num_bins)}, '
        f'w=(M/{float(args.mass_weight_pivot):.1e})^{float(args.mass_weight_alpha):g}',
        flush=True,
    )
    print(f'Processing {len(sim_ids)} simulations with {args.workers} worker(s).',
          flush=True)

    results = []
    n_workers = max(1, min(int(args.workers), len(payloads)))
    if n_workers == 1:
        for payload in payloads:
            sim_id, stats = compute_one(payload)
            results.append((sim_id, stats))
            print(f'[{sim_id:04d}] ratios ready', flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_sim = {ex.submit(compute_one, payload): payload[0]
                          for payload in payloads}
            for fut in futures.as_completed(fut_to_sim):
                sim_id, stats = fut.result()
                results.append((sim_id, stats))
                print(f'[{sim_id:04d}] ratios ready', flush=True)

    results.sort(key=lambda item: item[0])
    plotted_sim_ids = [sim_id for sim_id, _ in results]
    stacked = stack_results(results)

    pdf_path, png_path = make_plot(
        stacked, plotted_sim_ids, output_dir, output_name, noise_cfg,
        int(args.los_axis), tuple(args.ylim), bool(args.auto_ylim),
        float(args.mass_weight_pivot), float(args.mass_weight_alpha),
        bk_ng, int(args.bk_num_bins),
    )
    npz_path = targets[2]
    save_arrays(
        npz_path, stacked, plotted_sim_ids, noise_cfg, int(args.los_axis),
        ng, kmax, args.mas, int(args.hmf_bins),
        float(args.mass_weight_pivot), float(args.mass_weight_alpha),
        bk_ng, args.bk_kmin, args.bk_kmax, int(args.bk_num_bins),
    )
    print(f'Saved figure: {pdf_path}', flush=True)
    print(f'Saved figure: {png_path}', flush=True)
    print(f'Saved arrays:  {npz_path}', flush=True)


if __name__ == '__main__':
    main()
