#!/usr/bin/env python
"""
Plot true and mock halos around the highest-density FastPM voxels.

For one simulation, this script:
  1. Loads the full 128^3 FastPM density field for the 1000 Mpc/h box.
  2. Applies wrap padding with radius 4.
  3. Finds the three highest-density voxels in the unpadded field.
  4. Extracts the 9x9x9 periodic voxel window around each peak.
  5. Selects true and mock halos inside those same voxels.
  6. Plots projected density, true halos, and mock halos in three rows.
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
from matplotlib.colors import Normalize

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config
from plotters.plot_inference_v2 import load_mock, load_true


AXIS_NAMES = ('x', 'y', 'z')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML training config.')
    p.add_argument('--sim_id', type=int, required=True,
                   help='Simulation id to visualize.')
    p.add_argument('--mock', default=None,
                   help='Path to mock catalog .npz. Defaults to '
                        '<checkpoint_dir>/inference/mock_catalog_simXXXX.npz.')
    p.add_argument('--mock_dir', default=None,
                   help='Directory containing mock_catalog_simXXXX.npz. Ignored if '
                        '--mock is passed.')
    p.add_argument('--true_halo_dir', default=None,
                   help='Directory containing true halo HDF5 files. Defaults to '
                        'data_settings.halo_hdf5_dir.')
    p.add_argument('--fastpm_dir', default=None,
                   help='Directory containing FastPM density files. Defaults to '
                        'data_settings.fastpm_dir.')
    p.add_argument('--output_dir', default=None,
                   help='Directory for output figures. Defaults to the mock catalog '
                        'directory.')
    p.add_argument('--output_name', default=None,
                   help='Output basename without extension. Defaults to '
                        'high_density_regions_simXXXX.')
    p.add_argument('--z_snap', default=None,
                   help='Redshift string in FastPM/HDF5 filenames. Defaults to config.')
    p.add_argument('--projection_axis', type=int, choices=(0, 1, 2), default=2,
                   help='Axis to project over. Default 2 makes x-y panels.')
    p.add_argument('--n_peaks', type=int, default=3,
                   help='Number of top-density regions to show.')
    p.add_argument('--pad', type=int, default=4,
                   help='Wrap-padding/window radius in voxels. pad=4 gives 9x9x9.')
    p.add_argument('--smin', type=float, default=18.0,
                   help='Marker size for halos at lgMmin.')
    p.add_argument('--smax', type=float, default=220.0,
                   help='Marker size for halos at lgMmax.')
    return p.parse_args()


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(_REPO_ROOT, path))


def resolve_paths(cfg: dict, args):
    tc = cfg['train_settings']
    dc = cfg['data_settings']

    ckpt_dir = repo_path(tc['checkpoint_dir'])
    mock_dir = os.path.abspath(args.mock_dir or os.path.join(ckpt_dir, 'inference'))
    mock_path = os.path.abspath(
        args.mock or os.path.join(mock_dir, f'mock_catalog_sim{args.sim_id:04d}.npz')
    )
    true_halo_dir = repo_path(args.true_halo_dir or dc['halo_hdf5_dir'])
    fastpm_dir = repo_path(args.fastpm_dir or dc['fastpm_dir'])
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(mock_path))
    output_name = args.output_name or f'high_density_regions_sim{args.sim_id:04d}'
    return mock_path, true_halo_dir, fastpm_dir, output_dir, output_name


def load_density_full(fastpm_dir: str, sim_id: int, grid: int, z_snap: str) -> np.ndarray:
    fname = os.path.join(
        fastpm_dir,
        str(sim_id),
        f'density_HR_full_m_res_{grid}_z={z_snap}_nbatch_8_nfilter_3_ncnn_0.pk',
    )
    if not os.path.exists(fname):
        raise FileNotFoundError(f'FastPM density file not found: {fname}')
    with open(fname, 'rb') as f:
        rho = pickle.load(f)['density_cic_unpad_combined']
    rho = np.asarray(rho, dtype=np.float32)
    if rho.shape != (grid, grid, grid):
        raise ValueError(f'Expected density shape {(grid, grid, grid)}, got {rho.shape}')
    return rho


def top_density_voxels(rho: np.ndarray, n_peaks: int) -> np.ndarray:
    flat = rho.ravel()
    if n_peaks > flat.size:
        raise ValueError(f'n_peaks={n_peaks} exceeds number of voxels={flat.size}')
    finite = np.isfinite(flat)
    if finite.sum() < n_peaks:
        raise ValueError('Density field does not contain enough finite voxels.')
    finite_idx = np.flatnonzero(finite)
    vals = flat[finite_idx]
    top_local = np.argpartition(vals, -n_peaks)[-n_peaks:]
    top_idx = finite_idx[top_local]
    order = np.argsort(flat[top_idx])[::-1]
    return np.column_stack(np.unravel_index(top_idx[order], rho.shape)).astype(np.int32)


def density_window(rho_pad: np.ndarray, center: np.ndarray, pad: int) -> np.ndarray:
    ix, iy, iz = center
    width = 2 * pad + 1
    return rho_pad[ix:ix + width, iy:iy + width, iz:iz + width]


def axis_windows(center: np.ndarray, grid: int, pad: int):
    return [((np.arange(c - pad, c + pad + 1) % grid).astype(np.int32))
            for c in center]


def select_halos_in_window(cat: dict, center: np.ndarray, grid: int,
                           BoxSize: float, pad: int):
    pos = np.asarray(cat['pos'], dtype=np.float64) % BoxSize
    lgM = np.asarray(cat['lgM'], dtype=np.float64)
    if pos.size == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), dtype=np.int32)

    cell = BoxSize / grid
    scaled = pos / cell
    vox = np.floor(scaled).astype(np.int32) % grid
    frac = scaled - np.floor(scaled)

    windows = axis_windows(center, grid, pad)
    maps = [{int(v): j for j, v in enumerate(axis_vals)} for axis_vals in windows]

    mask = (
        np.isin(vox[:, 0], windows[0]) &
        np.isin(vox[:, 1], windows[1]) &
        np.isin(vox[:, 2], windows[2])
    )
    if not np.any(mask):
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), dtype=np.int32)

    vox_sel = vox[mask]
    frac_sel = frac[mask]
    local_base = np.empty_like(vox_sel, dtype=np.float64)
    for axis in range(3):
        local_base[:, axis] = [maps[axis][int(v)] for v in vox_sel[:, axis]]
    local_pos = local_base + frac_sel
    return local_pos, lgM[mask], vox_sel


def marker_sizes(lgM: np.ndarray, lgMmin: float, lgMmax: float,
                 smin: float, smax: float) -> np.ndarray:
    if lgM.size == 0:
        return np.zeros(0)
    t = np.clip((lgM - lgMmin) / max(lgMmax - lgMmin, 1e-6), 0.0, 1.0)
    return smin + (smax - smin) * t ** 1.8


def project_density(region: np.ndarray, projection_axis: int) -> tuple[np.ndarray, list[int]]:
    display_axes = [axis for axis in range(3) if axis != projection_axis]
    proj = np.sum(region, axis=projection_axis)
    return proj, display_axes


def setup_local_axes(ax, display_axes, pad: int):
    extent = pad + 0.5
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'local {AXIS_NAMES[display_axes[0]]} [voxels]')
    ax.set_ylabel(f'local {AXIS_NAMES[display_axes[1]]} [voxels]')
    ticks = np.arange(-pad, pad + 1, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticks(np.arange(-pad - 0.5, pad + 1.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-pad - 0.5, pad + 1.0, 1.0), minor=True)
    ax.grid(which='minor', color='0.88', lw=0.45)
    ax.axvline(-0.5, color='white', lw=0.8, alpha=0.8)
    ax.axvline(0.5, color='white', lw=0.8, alpha=0.8)
    ax.axhline(-0.5, color='white', lw=0.8, alpha=0.8)
    ax.axhline(0.5, color='white', lw=0.8, alpha=0.8)


def scatter_halos(ax, local_pos: np.ndarray, lgM: np.ndarray, display_axes: list[int],
                  lgMmin: float, lgMmax: float, smin: float, smax: float,
                  pad: int, color: str, label: str):
    if lgM.size == 0:
        ax.text(0.5, 0.5, 'No halos', ha='center', va='center',
                transform=ax.transAxes, color='0.35', fontsize=11)
        return None
    xy = local_pos[:, display_axes] - (pad + 0.5)
    sizes = marker_sizes(lgM, lgMmin, lgMmax, smin, smax)
    return ax.scatter(
        xy[:, 0], xy[:, 1],
        s=sizes,
        c=color,
        alpha=0.72,
        edgecolors='k',
        linewidths=0.35,
        label=label,
    )


def add_mass_legend(ax, lgMmin: float, lgMmax: float, smin: float, smax: float):
    masses = np.array([
        lgMmin,
        0.5 * (lgMmin + min(lgMmax, 14.8)),
        min(lgMmax, 14.8),
    ])
    handles = [
        ax.scatter([], [], s=marker_sizes(np.array([m]), lgMmin, lgMmax, smin, smax)[0],
                   c='none', edgecolors='k', linewidths=0.5)
        for m in masses
    ]
    labels = [rf'$\log M={m:.1f}$' for m in masses]
    ax.legend(handles, labels, title='Marker size', loc='upper right',
              frameon=True, framealpha=0.92, fontsize=8, title_fontsize=8)


def make_plot(rho: np.ndarray, centers: np.ndarray, mock: dict, true_cat: dict,
              meta: dict, cfg: dict, args, output_dir: str, output_name: str):
    sc = cfg['sim_settings']
    grid = int(sc['ns_d'])
    pad = int(args.pad)
    width = 2 * pad + 1
    BoxSize = float(meta.get('BoxSize', cfg['data_settings']['BoxSize']))
    lgMmin = float(sc['lgMmin'])
    lgMmax = float(sc['lgMmax'])

    rho_pad = np.pad(rho, pad, mode='wrap')
    display_axes = [axis for axis in range(3) if axis != args.projection_axis]

    regions = [density_window(rho_pad, center, pad) for center in centers]
    projections = [project_density(region, args.projection_axis)[0]
                   for region in regions]
    finite_proj = np.concatenate([p[np.isfinite(p)] for p in projections])
    vmin = np.percentile(finite_proj, 2)
    vmax = np.percentile(finite_proj, 99.5)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite_proj))
        vmax = float(np.nanmax(finite_proj))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        len(centers), 3,
        figsize=(13.5, 4.3 * len(centers)),
        constrained_layout=True,
        squeeze=False,
    )

    density_im = None
    extent = [-pad - 0.5, pad + 0.5, -pad - 0.5, pad + 0.5]
    for row, center in enumerate(centers):
        peak_val = rho[tuple(center)]
        proj = projections[row]

        ax = axes[row, 0]
        density_im = ax.imshow(
            proj.T,
            origin='lower',
            extent=extent,
            cmap='inferno',
            norm=norm,
            interpolation='nearest',
        )
        setup_local_axes(ax, display_axes, pad)
        ax.set_title(
            f'Peak {row + 1}: projected density\n'
            f'voxel=({center[0]}, {center[1]}, {center[2]}), '
            f'rho={peak_val:.3g}'
        )

        true_local, true_lgM, _ = select_halos_in_window(
            true_cat, center, grid, BoxSize, pad)
        mock_local, mock_lgM, _ = select_halos_in_window(
            mock, center, grid, BoxSize, pad)

        ax = axes[row, 1]
        setup_local_axes(ax, display_axes, pad)
        scatter_halos(
            ax, true_local, true_lgM, display_axes,
            lgMmin, lgMmax, args.smin, args.smax,
            pad,
            color='#2B6CB0',
            label='True',
        )
        ax.set_title(f'True halos in {width}x{width}x{width} voxels '
                     f'($N={len(true_lgM)}$)')

        ax = axes[row, 2]
        setup_local_axes(ax, display_axes, pad)
        scatter_halos(
            ax, mock_local, mock_lgM, display_axes,
            lgMmin, lgMmax, args.smin, args.smax,
            pad,
            color='#DD6B20',
            label='Mock',
        )
        ax.set_title(f'Mock halos in {width}x{width}x{width} voxels '
                     f'($N={len(mock_lgM)}$)')
        if row == 0:
            add_mass_legend(ax, lgMmin, lgMmax, args.smin, args.smax)

    if density_im is not None:
        cbar = fig.colorbar(density_im, ax=axes[:, 0], fraction=0.035, pad=0.02)
        cbar.set_label(
            f'density summed over local {AXIS_NAMES[args.projection_axis]}'
        )

    fig.suptitle(
        f'High-density FastPM regions: sim {int(meta["sim_id"]):04d} '
        f'({BoxSize:.0f} Mpc/h, {grid}^3 grid)',
        fontsize=14,
        fontweight='bold',
    )

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{output_name}.pdf')
    png_path = os.path.join(output_dir, f'{output_name}.png')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=180)
    fig.savefig(png_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    return pdf_path, png_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sc = cfg['sim_settings']
    dc = cfg['data_settings']

    grid = int(sc['ns_d'])
    if grid != int(sc['ns_h']):
        raise ValueError(f'This plot assumes ns_d == ns_h, got {grid} and {sc["ns_h"]}')
    if args.pad != 4:
        print(f'WARNING: --pad {args.pad} requested; the default/requested CHARM '
              f'visualization uses wrap padding of 4.', flush=True)

    z_snap = str(args.z_snap or dc['z_snap'])
    mock_path, true_halo_dir, fastpm_dir, output_dir, output_name = resolve_paths(cfg, args)

    print(f'Loading density from: {fastpm_dir}/{args.sim_id}/', flush=True)
    rho = load_density_full(fastpm_dir, args.sim_id, grid, z_snap)
    centers = top_density_voxels(rho, int(args.n_peaks))
    print('Top density voxel centers:', centers.tolist(), flush=True)

    print(f'Loading mock catalog: {mock_path}', flush=True)
    mock, meta = load_mock(mock_path)
    print(f'  Mock halos: {len(mock["pos"])}', flush=True)

    print(f'Loading true catalog from: {true_halo_dir}/{args.sim_id}/', flush=True)
    true_cat, _ = load_true(true_halo_dir, args.sim_id, z_snap)
    print(f'  True halos: {len(true_cat["pos"])}', flush=True)

    pdf_path, png_path = make_plot(
        rho, centers, mock, true_cat, meta, cfg, args, output_dir, output_name)
    print(f'Saved figure: {pdf_path}', flush=True)
    print(f'Saved figure: {png_path}', flush=True)


if __name__ == '__main__':
    main()
