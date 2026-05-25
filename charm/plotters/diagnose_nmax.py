#!/usr/bin/env python3
"""
diagnose_nmax.py
----------------
Compute the fractional halo-count bias introduced by capping the per-voxel
halo count to Nmax:

    bias(Nmax) = Ntot_cap(Nmax) / Ntot_true  -  1   [<= 0]

Plots bias (%) vs Nmax for the first 100 LH training simulations in a
10 × 10 grid, with horizontal reference lines at −1 % and −0.1 %, and
vertical markers at Nmax = 4 and 8.

Usage
-----
    python charm/plotters/diagnose_nmax.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v0.yaml

    # custom output path:
    python charm/plotters/diagnose_nmax.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v0.yaml \\
        --output  diagnostics/nmax_bias.png \\
        --nsims   100 \\
        --nmax_max 16
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))
from config_loader import load_config


# ── helpers ───────────────────────────────────────────────────────────────────

def _halo_path(halo_dir: str, sim_id: int, mass_type: str, z_snap: str) -> str:
    return os.path.join(
        halo_dir, str(sim_id),
        f'halos_{mass_type}_z{z_snap}.h5',
    )


def bias_curve(n_per_voxel: np.ndarray,
               nmax_values: np.ndarray) -> np.ndarray:
    """
    Return bias (%) = (Ntot_cap / Ntot_true - 1) * 100 for each Nmax in
    nmax_values.  n_per_voxel : flat integer array of per-voxel counts.
    """
    ntot_true = float(n_per_voxel.sum())
    if ntot_true == 0:
        return np.zeros(len(nmax_values))
    biases = np.empty(len(nmax_values))
    for i, nmax in enumerate(nmax_values):
        ntot_cap = float(np.minimum(n_per_voxel, nmax).sum())
        biases[i] = (ntot_cap / ntot_true - 1.0) * 100.0
    return biases


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Training YAML config (same one used for training).')
    p.add_argument('--output', default=None,
                   help='Output PNG path.  Defaults to '
                        '<checkpoint_dir>/diagnostics/nmax_bias.png.')
    p.add_argument('--nsims', type=int, default=100,
                   help='Number of training sims to include.')
    p.add_argument('--nmax_max', type=int, default=16,
                   help='Maximum Nmax to evaluate on the x-axis.')
    p.add_argument('--halo_dir', default=None,
                   help='Override per-sim halo HDF5 directory.')
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    sc     = cfg['sim_settings']
    dc     = cfg['data_settings']
    tc     = cfg['train_settings']

    mass_type = sc['mass_type']
    z_snap    = str(dc['z_snap'])
    nmax_cfg  = int(sc['Nmax'])   # what the config currently says

    # ── resolve halo directory ────────────────────────────────────────────────
    if args.halo_dir:
        halo_dir = args.halo_dir
    else:
        halo_dir = dc.get('halo_hdf5_dir') or os.path.join(
            _REPO_ROOT, '..', 'data',
            f'halos_Mmin{dc.get("Mmin_cut_str", "")}',
        )
    if not os.path.isdir(halo_dir):
        raise FileNotFoundError(
            f'Halo directory not found: {halo_dir!r}\n'
            'Set data_settings.halo_hdf5_dir in the config or use --halo_dir.'
        )
    print(f'Halo dir : {halo_dir}')

    # ── load cosmology file ───────────────────────────────────────────────────
    cosmo_all = np.loadtxt(dc['lh_cosmo_file'])   # (nsims_total, 5)

    # ── collect bias curves ───────────────────────────────────────────────────
    nmax_values = np.arange(1, args.nmax_max + 1)
    nsims_req   = min(args.nsims, int(sc['nsims_train']))

    results   = []   # list of (sim_id, cosmo_vec, bias_array, ntot_true, vox_max)
    skipped   = 0
    for sim_id in range(nsims_req):
        fp = _halo_path(halo_dir, sim_id, mass_type, z_snap)
        if not os.path.exists(fp):
            skipped += 1
            continue
        with h5py.File(fp, 'r') as f:
            n_per_voxel = f['N_halos'][:].ravel().astype(np.int32)
        ntot_true = int(n_per_voxel.sum())
        vox_max   = int(n_per_voxel.max())
        if ntot_true == 0:
            skipped += 1
            continue
        b = bias_curve(n_per_voxel, nmax_values)
        results.append((sim_id, cosmo_all[sim_id], b, ntot_true, vox_max))
        if len(results) % 20 == 0:
            print(f'  Loaded {len(results)} / {nsims_req} sims …', flush=True)

    nsims_loaded = len(results)
    print(f'Loaded {nsims_loaded} sims  ({skipped} skipped)', flush=True)
    if nsims_loaded == 0:
        raise RuntimeError('No simulations loaded — check halo_dir.')

    # For the summary panel: find Nmax where bias crosses −1% and −0.1%
    # for each sim (first Nmax where |bias| < threshold).
    def _nmax_at_threshold(b_arr, threshold_pct):
        # b_arr is <= 0; find first index where bias >= -threshold_pct
        idx = np.where(b_arr >= -threshold_pct)[0]
        return nmax_values[idx[0]] if len(idx) > 0 else nmax_values[-1] + 1

    nmax_1pct   = np.array([_nmax_at_threshold(r[2], 1.0)  for r in results])
    nmax_01pct  = np.array([_nmax_at_threshold(r[2], 0.1)  for r in results])
    ntot_all    = np.array([r[3]  for r in results])
    vmax_all    = np.array([r[4]  for r in results])

    print(f'\nNmax needed for <1%  bias : '
          f'median={np.median(nmax_1pct):.0f}  '
          f'max={nmax_1pct.max():.0f}', flush=True)
    print(f'Nmax needed for <0.1% bias: '
          f'median={np.median(nmax_01pct):.0f}  '
          f'max={nmax_01pct.max():.0f}', flush=True)
    print(f'True per-voxel max count  : '
          f'median={np.median(vmax_all):.0f}  '
          f'max={vmax_all.max():.0f}', flush=True)

    # ── build figure ──────────────────────────────────────────────────────────
    ncols     = 10
    nrows     = 10
    nsubs     = min(nsims_loaded, ncols * nrows)   # up to 100 subplots

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.0, nrows * 1.8),
                             sharex=True, sharey=True)
    axes_flat = axes.ravel()

    # Colour-code by total halo count (log scale)
    log_ntot = np.log10(ntot_all[:nsubs].clip(1))
    vmin_c, vmax_c = log_ntot.min(), log_ntot.max()
    cmap = plt.cm.plasma

    for i in range(nsubs):
        ax  = axes_flat[i]
        sid, cosmo, b, ntot, vmax = results[i]

        col = cmap((log_ntot[i] - vmin_c) / max(vmax_c - vmin_c, 1e-6))
        ax.plot(nmax_values, b, color=col, lw=1.2)

        # Reference thresholds
        ax.axhline(-1.0, color='firebrick', lw=0.8, ls='--', alpha=0.8)
        ax.axhline(-0.1, color='darkorange', lw=0.8, ls=':',  alpha=0.8)

        # Nmax markers
        ax.axvline(4, color='steelblue', lw=0.7, ls='--', alpha=0.6)
        ax.axvline(nmax_cfg, color='forestgreen', lw=0.7, ls='--', alpha=0.6)

        ax.set_xlim(1, args.nmax_max)
        ax.set_ylim(-50, 2)
        ax.set_xticks([1, 4, 8, 12, 16][:sum(v <= args.nmax_max for v in [1,4,8,12,16])])
        ax.tick_params(labelsize=5)
        ax.set_title(f'sim {sid}  N={ntot//1000:.0f}k', fontsize=5, pad=2)

        # Mark the point where each threshold is first satisfied
        ax.axvline(nmax_1pct[i],  color='firebrick',  lw=0.6, ls='-', alpha=0.5)
        ax.axvline(nmax_01pct[i], color='darkorange',  lw=0.6, ls='-', alpha=0.5)

    # Hide unused panels
    for i in range(nsubs, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Shared axis labels
    fig.text(0.5, 0.01, r'$N_{\rm max}$ (cap per voxel)',
             ha='center', va='bottom', fontsize=11)
    fig.text(0.01, 0.5,
             r'$(N_{\rm tot,cap}\,/\,N_{\rm tot,true} - 1)\times 100\,[\%]$',
             ha='left', va='center', rotation=90, fontsize=11)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='firebrick',   ls='--', lw=1.2, label='−1 % threshold'),
        Line2D([0], [0], color='darkorange',  ls=':',  lw=1.2, label='−0.1 % threshold'),
        Line2D([0], [0], color='steelblue',   ls='--', lw=1.2, label='Nmax = 4'),
        Line2D([0], [0], color='forestgreen', ls='--', lw=1.2,
               label=f'Nmax = {nmax_cfg} (config)'),
        Line2D([0], [0], color='firebrick',   ls='-',  lw=1.0,
               label='Nmax where |bias| < 1%'),
        Line2D([0], [0], color='darkorange',  ls='-',  lw=1.0,
               label='Nmax where |bias| < 0.1%'),
    ]
    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=7.5, framealpha=0.9)

    # Colourbar for log10(Ntot)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=matplotlib.colors.Normalize(vmin=vmin_c, vmax=vmax_c))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01, aspect=40)
    cbar.set_label(r'$\log_{10}(N_{\rm tot,true})$', fontsize=9)

    # ── summary inset (histogram of Nmax needed) ──────────────────────────────
    ax_inset = fig.add_axes([0.13, 0.91, 0.28, 0.07])
    bins = np.arange(0.5, args.nmax_max + 1.5)
    ax_inset.hist(nmax_1pct,  bins=bins, color='firebrick', alpha=0.6,
                  label=f'<1%  (med={np.median(nmax_1pct):.0f})')
    ax_inset.hist(nmax_01pct, bins=bins, color='darkorange', alpha=0.6,
                  label=f'<0.1% (med={np.median(nmax_01pct):.0f})')
    ax_inset.set_xlabel('Nmax needed', fontsize=7)
    ax_inset.set_ylabel('# sims', fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.legend(fontsize=6, framealpha=0.8)
    ax_inset.set_title('Distribution of required Nmax', fontsize=7)

    fig.suptitle(
        f'Halo-count bias from per-voxel cap  '
        f'(Mmin={dc["Mmin_cut_str"]}, z={z_snap}, {nsubs} sims)',
        fontsize=12, y=1.0,
    )
    plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.98])

    # ── save ─────────────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        diag_dir = os.path.join(
            _REPO_ROOT, tc['checkpoint_dir'], 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        out_path = os.path.join(diag_dir, 'nmax_bias.png')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}', flush=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
