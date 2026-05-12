#!/usr/bin/env python
"""
Aggregate mock/true ratio diagnostics over many CHARM inference catalogs.

This is intentionally separate from plot_inference_v2.py. It reuses that
script's catalog loading and statistic helpers, then makes one summary figure:
gray curves for individual simulations, black mean curve, gray 16th-84th
percentile band, and reference lines at 1.0 and +/-10%.  It also writes
cosmology-coloured variants where each simulation curve is coloured by
Omega_m or sigma_8.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config
from plot_inference_v2 import (
    apply_rsd,
    halo_mass_function,
    load_mock,
    load_true,
    power_spectrum,
    power_spectrum_multipoles,
)


CACHE_VERSION = 3


STAT_SPECS = OrderedDict([
    ('ntot_pdf', {
        'title': r'Per-voxel count PDF',
        'xlabel': r'$N_{\rm tot}$',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('hmf', {
        'title': 'Halo mass function',
        'xlabel': r'$\log_{10} M$ [$M_\odot/h$]',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('pk_real', {
        'title': r'Real-space $P(k)$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'ylabel': r'$P_{\rm mock}/P_{\rm true}$',
        'xscale': 'log',
    }),
    ('pk_real_mass', {
        'title': r'Real-space mass-weighted $P(k)$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'ylabel': r'$P_{M,\rm mock}/P_{M,\rm true}$',
        'xscale': 'log',
    }),
    ('rsd_p0', {
        'title': r'RSD monopole $P_0$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'ylabel': 'mock / true',
        'xscale': 'log',
    }),
    ('rsd_p2', {
        'title': r'RSD quadrupole $P_2$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'ylabel': r'$P_{2,\rm mock}/P_{2,\rm true}$',
        'xscale': 'log',
    }),
    ('rsd_p4p0', {
        'title': r'RSD hexadecapole ratio $P_4/P_0$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'ylabel': 'mock / true',
        'xscale': 'log',
    }),
    ('rsd_p0_mass', {
        'title': r'RSD mass-weighted monopole $P_0$',
        'xlabel': r'$k$ [$h/{\rm Mpc}$]',
        'ylabel': 'mock / true',
        'xscale': 'log',
    }),
    ('vel_pdf_x', {
        'title': r'Velocity PDF: $v_x$',
        'xlabel': 'velocity [km/s]',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('vel_pdf_y', {
        'title': r'Velocity PDF: $v_y$',
        'xlabel': 'velocity [km/s]',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('vel_pdf_z', {
        'title': r'Velocity PDF: $v_z$',
        'xlabel': 'velocity [km/s]',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('vel_disp_mass', {
        'title': 'Velocity dispersion vs mass',
        'xlabel': r'$\log_{10} M$ [$M_\odot/h$]',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('cm_relation', {
        'title': r'Concentration-mass relation',
        'xlabel': r'$\log_{10} M$ [$M_\odot/h$]',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
    ('conc_pdf', {
        'title': 'Concentration PDF',
        'xlabel': r'$c_{200c}$',
        'ylabel': 'mock / true',
        'xscale': 'linear',
    }),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML training config.')
    p.add_argument('--mock_dir', default=None,
                   help='Directory with mock_catalog_simXXXX.npz files. Defaults '
                        'to <checkpoint_dir>/inference.')
    p.add_argument('--true_halo_dir', default=None,
                   help='Directory with true halo HDF5 files. Defaults to '
                        'data_settings.halo_hdf5_dir.')
    p.add_argument('--sim_start', type=int, default=None,
                   help='First simulation id. Defaults to nsims_train + nsims_val.')
    p.add_argument('--sim_end', type=int, default=None,
                   help='Exclusive upper simulation id. Defaults to sim_start + nsims_test.')
    p.add_argument('--sim_ids', nargs='*', type=int, default=None,
                   help='Explicit simulation ids. Overrides --sim_start/--sim_end.')
    p.add_argument('--output_dir', default=None,
                   help='Directory for figures. Defaults to --mock_dir.')
    p.add_argument('--output_name', default='inference_ratio_summary_test_sims',
                   help='Output basename without extension.')
    p.add_argument('--cache_dir', default=None,
                   help='Per-simulation statistic cache directory. Defaults to '
                        '<output_dir>/ratio_stats_cache.')
    p.add_argument('--no_cache', action='store_true',
                   help='Do not read or write per-simulation cached statistics.')
    p.add_argument('--workers', type=int, default=1,
                   help='Number of simulations to process concurrently.')
    p.add_argument('--ng', type=int, default=384,
                   help='Grid size for Pylians P(k), matching plot_inference_v2.py.')
    p.add_argument('--kmax', type=float, default=0.5,
                   help='Maximum k for P(k) curves.')
    p.add_argument('--z_snap', default=None,
                   help='Redshift string in true HDF5 filename. Defaults to config.')
    p.add_argument('--ylim', nargs=2, type=float, default=(0.5, 1.5),
                   metavar=('YMIN', 'YMAX'),
                   help='Shared y-limits for all ratio panels.')
    p.add_argument('--auto_ylim', action='store_true',
                   help='Use data-driven y-limits per panel instead of --ylim.')
    return p.parse_args()


def default_sim_ids(cfg: dict, sim_start: int | None, sim_end: int | None) -> list[int]:
    sc = cfg['sim_settings']
    if sim_start is None:
        sim_start = int(sc.get('nsims_train', 1800)) + int(sc.get('nsims_val', 100))
    if sim_end is None:
        sim_end = sim_start + int(sc.get('nsims_test', 100))
    if sim_end <= sim_start:
        raise ValueError(f'Empty sim range: [{sim_start}, {sim_end})')
    return list(range(sim_start, sim_end))


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def resolve_paths(cfg: dict, args):
    ckpt_dir = os.path.join(_REPO_ROOT, cfg['train_settings']['checkpoint_dir'])
    mock_dir = os.path.abspath(args.mock_dir or os.path.join(ckpt_dir, 'inference'))
    true_halo_dir = os.path.abspath(repo_path(
        args.true_halo_dir or cfg['data_settings']['halo_hdf5_dir']))
    output_dir = os.path.abspath(args.output_dir or mock_dir)
    cache_dir = os.path.abspath(args.cache_dir or os.path.join(output_dir, 'ratio_stats_cache'))
    return mock_dir, true_halo_dir, output_dir, cache_dir


def load_cosmology_params(cfg: dict, sim_ids: list[int]) -> dict[str, np.ndarray]:
    """Return cosmological parameters aligned with the plotted simulation order."""
    cosmo_path = repo_path(cfg['data_settings']['lh_cosmo_file'])
    cosmo_all = np.loadtxt(cosmo_path)
    sim_ids_arr = np.asarray(sim_ids, dtype=np.int64)
    if sim_ids_arr.size == 0:
        raise ValueError('No simulation ids supplied.')
    if sim_ids_arr.min() < 0 or sim_ids_arr.max() >= len(cosmo_all):
        raise ValueError(
            f'Simulation ids [{sim_ids_arr.min()}, {sim_ids_arr.max()}] '
            f'are outside cosmology table with {len(cosmo_all)} rows: {cosmo_path}'
        )
    cosmo = np.asarray(cosmo_all[sim_ids_arr], dtype=np.float64)
    return {
        'Omega_m': cosmo[:, 0],
        'sigma_8': cosmo[:, 4],
    }


def safe_ratio(num: np.ndarray, den: np.ndarray, floor_frac: float = 1e-8) -> np.ndarray:
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


def interp_ratio(x_num, y_num, x_den, y_den, floor_frac: float = 1e-8):
    x_num = np.asarray(x_num, dtype=np.float64)
    y_num = np.asarray(y_num, dtype=np.float64)
    x_den = np.asarray(x_den, dtype=np.float64)
    y_den = np.asarray(y_den, dtype=np.float64)
    if x_num.size == 0 or x_den.size == 0:
        return x_num, np.full_like(x_num, np.nan, dtype=np.float64)
    y_den_i = np.interp(x_num, x_den, y_den, left=np.nan, right=np.nan)
    return x_num, safe_ratio(y_num, y_den_i, floor_frac=floor_frac)


def multipole_over_monopole(pell: np.ndarray, p0: np.ndarray) -> np.ndarray:
    p0_abs = np.abs(p0)
    floor = 1e-6 * np.nanmax(p0_abs) if np.any(np.isfinite(p0_abs)) else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where((p0_abs > floor) & np.isfinite(pell), pell / p0, np.nan)
    return out


def binned_median(x, y, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        out.append(np.nanmedian(y[m]) if m.sum() > 2 else np.nan)
    return np.asarray(out, dtype=np.float64)


def velocity_dispersion(lgM, vel, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lgM >= lo) & (lgM < hi)
        out.append(np.std(vel[m]) if m.sum() > 2 else np.nan)
    return np.asarray(out, dtype=np.float64)


def hist_pdf(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    width = np.diff(bins)
    total = counts.sum()
    if total == 0:
        return np.full(len(width), np.nan, dtype=np.float64)
    return counts / (total * width)


def count_pdf(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.full(len(bins) - 1, np.nan, dtype=np.float64)
    return counts / total


def cache_path(cache_dir: str, sim_id: int):
    return os.path.join(cache_dir, f'ratio_stats_sim{sim_id:04d}.npz')


def save_cache(path: str, stats: dict):
    payload = {'cache_version': np.int32(CACHE_VERSION)}
    for name, (x, ratio) in stats.items():
        payload[f'{name}_x'] = x
        payload[f'{name}_ratio'] = ratio
    np.savez(path, **payload)


def load_cache(path: str):
    data = np.load(path)
    if 'cache_version' not in data or int(data['cache_version']) != CACHE_VERSION:
        return None
    stats = {}
    for name in STAT_SPECS:
        x_key = f'{name}_x'
        r_key = f'{name}_ratio'
        if x_key not in data or r_key not in data:
            return None
        stats[name] = (data[x_key], data[r_key])
    return stats


def compute_one(payload):
    (sim_id, mock_dir, true_halo_dir, z_snap, ng, kmax, lgMmin, lgMmax,
     vmin, vmax, cmin, cmax, count_max, stat_cache_path, use_cache) = payload

    if use_cache and os.path.exists(stat_cache_path):
        cached = load_cache(stat_cache_path)
        if cached is not None:
            return sim_id, cached

    mock_path = os.path.join(mock_dir, f'mock_catalog_sim{sim_id:04d}.npz')
    if not os.path.exists(mock_path):
        raise FileNotFoundError(f'Missing mock catalog for sim {sim_id}: {mock_path}')

    mock, meta = load_mock(mock_path)
    true_cat, _ = load_true(true_halo_dir, sim_id, z_snap)

    BoxSize = float(meta['BoxSize'])
    z = float(meta['z'])
    cosmo = meta['cosmo']
    stats = {}

    count_bins = np.arange(0, count_max + 2) - 0.5
    count_x = np.arange(0, count_max + 1)
    stats['ntot_pdf'] = (
        count_x,
        safe_ratio(
            count_pdf(mock['ntot_vol'].ravel(), count_bins),
            count_pdf(true_cat['ntot_vol'].ravel(), count_bins),
        ),
    )

    hmf_x_m, hmf_m = halo_mass_function(mock['lgM'], lgMmin, lgMmax, BoxSize=BoxSize)
    hmf_x_t, hmf_t = halo_mass_function(true_cat['lgM'], lgMmin, lgMmax, BoxSize=BoxSize)
    stats['hmf'] = interp_ratio(hmf_x_m, hmf_m, hmf_x_t, hmf_t)

    mass_mock = 10.0 ** mock['lgM']
    mass_true = 10.0 ** true_cat['lgM']

    k_m, pk_m = power_spectrum(mock['pos'], None, ng, BoxSize, kmax=kmax)
    k_t, pk_t = power_spectrum(true_cat['pos'], None, ng, BoxSize, kmax=kmax)
    stats['pk_real'] = interp_ratio(k_m, pk_m, k_t, pk_t)

    k_m, pk_m = power_spectrum(mock['pos'], mass_mock / 1e14, ng, BoxSize, kmax=kmax)
    k_t, pk_t = power_spectrum(true_cat['pos'], mass_true / 1e14, ng, BoxSize, kmax=kmax)
    stats['pk_real_mass'] = interp_ratio(k_m, pk_m, k_t, pk_t)

    pos_rsd_m = apply_rsd(mock['pos'], mock['vel'], BoxSize, z, cosmo, axis=2)
    pos_rsd_t = apply_rsd(true_cat['pos'], true_cat['vel'], BoxSize, z, cosmo, axis=2)

    k_m, p0_m, p2_m, p4_m = power_spectrum_multipoles(
        pos_rsd_m, None, ng, BoxSize, kmax=kmax, axis=2)
    k_t, p0_t, p2_t, p4_t = power_spectrum_multipoles(
        pos_rsd_t, None, ng, BoxSize, kmax=kmax, axis=2)
    stats['rsd_p0'] = interp_ratio(k_m, p0_m, k_t, p0_t)
    stats['rsd_p2'] = interp_ratio(
        k_m, p2_m,
        k_t, p2_t,
        floor_frac=1e-4,
    )
    stats['rsd_p4p0'] = interp_ratio(
        k_m, multipole_over_monopole(p4_m, p0_m),
        k_t, multipole_over_monopole(p4_t, p0_t),
        floor_frac=1e-4,
    )

    k_m, p0_m, _, _ = power_spectrum_multipoles(
        pos_rsd_m, mass_mock, ng, BoxSize, kmax=kmax, axis=2)
    k_t, p0_t, _, _ = power_spectrum_multipoles(
        pos_rsd_t, mass_true, ng, BoxSize, kmax=kmax, axis=2)
    stats['rsd_p0_mass'] = interp_ratio(k_m, p0_m, k_t, p0_t)

    v_bins = np.linspace(vmin * 1.15, vmax * 1.15, 60)
    v_x = 0.5 * (v_bins[:-1] + v_bins[1:])
    for ci, key in enumerate(('vel_pdf_x', 'vel_pdf_y', 'vel_pdf_z')):
        stats[key] = (
            v_x,
            safe_ratio(
                hist_pdf(mock['vel'][:, ci], v_bins),
                hist_pdf(true_cat['vel'][:, ci], v_bins),
            ),
        )

    mass_bins = np.linspace(lgMmin, lgMmax, 10)
    mass_x = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    stats['vel_disp_mass'] = (
        mass_x,
        safe_ratio(
            velocity_dispersion(mock['lgM'], mock['vel'], mass_bins),
            velocity_dispersion(true_cat['lgM'], true_cat['vel'], mass_bins),
        ),
    )
    stats['cm_relation'] = (
        mass_x,
        safe_ratio(
            binned_median(mock['lgM'], mock['conc'], mass_bins),
            binned_median(true_cat['lgM'], true_cat['conc'], mass_bins),
        ),
    )

    c_bins = np.linspace(cmin, cmax, 40)
    c_x = 0.5 * (c_bins[:-1] + c_bins[1:])
    stats['conc_pdf'] = (
        c_x,
        safe_ratio(
            hist_pdf(mock['conc'], c_bins),
            hist_pdf(true_cat['conc'], c_bins),
        ),
    )

    if use_cache:
        os.makedirs(os.path.dirname(stat_cache_path), exist_ok=True)
        save_cache(stat_cache_path, stats)

    return sim_id, stats


def stack_stats(results: list[tuple[int, dict]]):
    stacked = OrderedDict()
    for name in STAT_SPECS:
        x_ref = None
        rows = []
        sim_ids = []
        for sim_id, stats in results:
            x, ratio = stats[name]
            x = np.asarray(x, dtype=np.float64)
            ratio = np.asarray(ratio, dtype=np.float64)
            if x_ref is None:
                x_ref = x
            elif x.size != x_ref.size or not np.allclose(x, x_ref, equal_nan=True):
                ratio = np.interp(x_ref, x, ratio, left=np.nan, right=np.nan)
            rows.append(ratio)
            sim_ids.append(sim_id)
        stacked[name] = (x_ref, np.vstack(rows), np.asarray(sim_ids))
    return stacked


def nan_summary(rows: np.ndarray):
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(rows, axis=0)
        p16 = np.nanpercentile(rows, 16, axis=0)
        p84 = np.nanpercentile(rows, 84, axis=0)
    return mean, p16, p84


def set_auto_ylim(ax, rows, mean, p16, p84, yscale='linear'):
    vals = np.concatenate([
        rows[np.isfinite(rows)],
        mean[np.isfinite(mean)],
        p16[np.isfinite(p16)],
        p84[np.isfinite(p84)],
        np.asarray([0.9, 1.0, 1.1]),
    ])
    if yscale == 'log':
        vals = vals[vals > 0]
    if vals.size == 0:
        ax.set_ylim(0.5, 1.5)
        return
    if yscale == 'log':
        lo, hi = np.nanpercentile(vals, [1, 99])
        ax.set_ylim(max(lo / 1.15, 1e-6), hi * 1.15)
        return
    lo, hi = np.nanpercentile(vals, [1, 99])
    pad = 0.08 * max(hi - lo, 0.2)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


def make_plot(stacked: OrderedDict, sim_ids: list[int], output_dir: str,
              output_name: str, ylim, auto_ylim: bool,
              color_values: np.ndarray | None = None,
              color_label: str | None = None,
              output_suffix: str = '',
              show_summary: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

    n_panels = len(STAT_SPECS)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 4.2 * nrows),
                             constrained_layout=True)
    axes = np.ravel(axes)
    cmap = None
    norm = None
    if color_values is not None:
        color_values = np.asarray(color_values, dtype=np.float64)
        if color_values.shape[0] != len(sim_ids):
            raise ValueError(
                f'color_values has length {color_values.shape[0]}, '
                f'but there are {len(sim_ids)} simulations.'
            )
        finite_c = color_values[np.isfinite(color_values)]
        if finite_c.size == 0:
            raise ValueError(f'No finite values available for {color_label}.')
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=float(finite_c.min()), vmax=float(finite_c.max()))

    for ax, (name, spec) in zip(axes, STAT_SPECS.items()):
        x, rows, _ = stacked[name]
        yscale = spec.get('yscale', 'linear')
        plot_rows = np.where(rows > 0, rows, np.nan) if yscale == 'log' else rows
        mean, p16, p84 = nan_summary(plot_rows)

        for i, row in enumerate(plot_rows):
            if color_values is None:
                color = '0.45'
                alpha = 0.16
                lw = 0.8
            else:
                color = cmap(norm(color_values[i]))
                alpha = 0.38
                lw = 0.95
            ax.plot(x, row, color=color, alpha=alpha, lw=lw,
                    zorder=1, solid_capstyle='round')
        if show_summary:
            ax.fill_between(x, p16, p84, color='0.45', alpha=0.24,
                            linewidth=0, zorder=2, label='16th-84th percentile')
            ax.plot(x, mean, color='k', lw=2.2, zorder=3, label='mean')

        ax.axhline(1.0, color='k', ls='--', lw=1.8, zorder=0)
        ax.axhline(0.9, color='0.25', ls='--', lw=0.8, zorder=0)
        ax.axhline(1.1, color='0.25', ls='--', lw=0.8, zorder=0)
        ax.grid(True, which='major', color='0.88', lw=0.8)
        ax.grid(True, which='minor', color='0.93', lw=0.45)
        ax.set_title(spec['title'])
        ax.set_xlabel(spec['xlabel'])
        ax.set_ylabel(spec['ylabel'])
        ax.set_xscale(spec['xscale'])
        ax.set_yscale(yscale)
        ax.tick_params(axis='both', which='major', length=5.5, width=1.0,
                       labelsize=10, direction='out')
        ax.tick_params(axis='both', which='minor', length=3.2, width=0.8,
                       direction='out')
        for spine in ax.spines.values():
            spine.set_color('0.18')
            spine.set_linewidth(0.9)
        if auto_ylim:
            set_auto_ylim(ax, plot_rows, mean, p16, p84, yscale=yscale)
        else:
            ax.set_ylim(*ylim)

    for ax in axes[n_panels:]:
        ax.axis('off')

    if show_summary:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.01))
    if color_values is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:n_panels].tolist(),
                            shrink=0.82, pad=0.010,
                            fraction=0.018, aspect=45)
        cbar.set_label(color_label or 'cosmology', fontsize=12)
        cbar.ax.tick_params(labelsize=10, length=5.0, width=0.9,
                            direction='out')
        cbar.outline.set_linewidth(0.8)

    title_suffix = f' coloured by {color_label}' if color_values is not None else ''
    fig.suptitle(
        f'CHARM mock/true summary-statistic ratios: '
        f'{len(sim_ids)} simulations ({min(sim_ids):04d}-{max(sim_ids):04d})'
        f'{title_suffix}',
        y=1.035,
        fontsize=15,
        fontweight='bold',
    )

    pdf_path = os.path.join(output_dir, f'{output_name}{output_suffix}.pdf')
    png_path = os.path.join(output_dir, f'{output_name}{output_suffix}.png')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=180)
    fig.savefig(png_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    return pdf_path, png_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sc = cfg['sim_settings']
    dc = cfg['data_settings']

    sim_ids = args.sim_ids if args.sim_ids else default_sim_ids(
        cfg, args.sim_start, args.sim_end)
    mock_dir, true_halo_dir, output_dir, stat_cache_dir = resolve_paths(cfg, args)
    z_snap = str(args.z_snap or dc['z_snap'])

    os.makedirs(output_dir, exist_ok=True)
    if not args.no_cache:
        os.makedirs(stat_cache_dir, exist_ok=True)

    count_max = int(max(sc.get('Nmax', 0), dc.get('nMax_h_raw', 0)))
    payloads = []
    for sim_id in sim_ids:
        payloads.append((
            sim_id,
            mock_dir,
            true_halo_dir,
            z_snap,
            int(args.ng),
            float(args.kmax),
            float(sc['lgMmin']),
            float(sc['lgMmax']),
            float(sc['vmin']),
            float(sc['vmax']),
            float(sc['cmin']),
            float(sc['cmax']),
            count_max,
            cache_path(stat_cache_dir, sim_id),
            not args.no_cache,
        ))

    print(f'Mock catalog directory: {mock_dir}', flush=True)
    print(f'True halo directory: {true_halo_dir}', flush=True)
    print(f'Processing {len(sim_ids)} simulations with {args.workers} worker(s).',
          flush=True)

    results = []
    n_workers = max(1, min(int(args.workers), len(payloads)))
    if n_workers == 1:
        for payload in payloads:
            sim_id, stats = compute_one(payload)
            results.append((sim_id, stats))
            print(f'[{sim_id:04d}] statistics ready', flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_sim = {ex.submit(compute_one, payload): payload[0]
                          for payload in payloads}
            for fut in futures.as_completed(fut_to_sim):
                sim_id, stats = fut.result()
                results.append((sim_id, stats))
                print(f'[{sim_id:04d}] statistics ready', flush=True)

    results.sort(key=lambda item: item[0])
    plotted_sim_ids = [sim_id for sim_id, _ in results]
    stacked = stack_stats(results)
    pdf_path, png_path = make_plot(
        stacked, plotted_sim_ids, output_dir, args.output_name,
        tuple(args.ylim), bool(args.auto_ylim),
    )
    print(f'Saved figure: {pdf_path}', flush=True)
    print(f'Saved figure: {png_path}', flush=True)

    cosmo_params = load_cosmology_params(cfg, plotted_sim_ids)
    color_specs = [
        ('Omega_m', r'$\Omega_m$', '_colored_Omega_m'),
        ('sigma_8', r'$\sigma_8$', '_colored_sigma8'),
    ]
    for key, label, suffix in color_specs:
        pdf_path, png_path = make_plot(
            stacked,
            plotted_sim_ids,
            output_dir,
            args.output_name,
            tuple(args.ylim),
            bool(args.auto_ylim),
            color_values=cosmo_params[key],
            color_label=label,
            output_suffix=suffix,
            show_summary=False,
        )
        print(f'Saved figure: {pdf_path}', flush=True)
        print(f'Saved figure: {png_path}', flush=True)


if __name__ == '__main__':
    main()
