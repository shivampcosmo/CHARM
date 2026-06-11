#!/usr/bin/env python
"""
Run CHARM inference for one simulation and plot only redshift-space
power-spectrum multipole ratios.

This script is intentionally separate from run_inference_v2.py.  It reuses the
same model loading, conditioning, sampling, and catalog reconstruction helpers,
then compares mock and truth only through RSD P0/P2/P4 ratios.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config
from plotters.plot_inference_v2 import apply_rsd, load_true, power_spectrum_multipoles
from run_charm_joint_ddp import build_model
from inferers.run_inference_v2 import (
    build_cond_tensors,
    build_dm_velocity_interpolators,
    estimate_target_prior_from_cosmology,
    load_checkpoint,
    load_cosmology,
    load_fastpm,
    reconstruct_catalog,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML training config.')
    p.add_argument('--sim_id', type=int, required=True,
                   help='Simulation ID to run inference on.')
    p.add_argument('--checkpoint', default=None,
                   help='Path to .pth checkpoint. Defaults to '
                        '<checkpoint_dir>/charm_joint_best_val.pth.')
    p.add_argument('--output_dir', default=None,
                   help='Directory for output plots/npz. Defaults to '
                        '<checkpoint_dir>/inference_rsd_ratios/.')
    p.add_argument('--output_name', default=None,
                   help='Output basename without extension. Defaults to '
                        'rsd_multipole_ratios_simXXXX.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Torch device.')
    p.add_argument('--true_halo_dir', default=None,
                   help='Directory containing true halo HDF5 files. Defaults to config.')
    p.add_argument('--z_snap', default=None,
                   help='Redshift string. Defaults to data_settings.z_snap.')
    p.add_argument('--ng', type=int, default=384,
                   help='Grid size for Pylians multipoles.')
    p.add_argument('--kmax', type=float, default=0.4,
                   help='Maximum k for plotted multipoles.')
    p.add_argument('--los_axis', type=int, choices=(0, 1, 2), default=2,
                   help='Line-of-sight axis for RSD and multipoles.')
    p.add_argument('--mock_rsd_noise_sigma', type=float, default=2.0,
                   help='Legacy shortcut Gaussian position noise sigma for mock '
                        'RSD positions [Mpc/h]. Used for any unspecified '
                        'mock directional sigmas according to --noise_axes.')
    p.add_argument('--truth_rsd_noise_sigma', type=float, default=0.0,
                   help='Legacy shortcut Gaussian position noise sigma for truth '
                        'RSD positions [Mpc/h]. Used for any unspecified '
                        'truth directional sigmas according to --noise_axes.')
    p.add_argument('--mock_rsd_noise_sigma_los', type=float, default=None,
                   help='Gaussian RSD-position noise sigma along LOS for mock [Mpc/h]. '
                        'Defaults to --mock_rsd_noise_sigma.')
    p.add_argument('--mock_rsd_noise_sigma_transverse', type=float, default=None,
                   help='Gaussian RSD-position noise sigma in each transverse '
                        'axis for mock [Mpc/h]. Defaults to 0 for --noise_axes los '
                        'or --mock_rsd_noise_sigma for --noise_axes all.')
    p.add_argument('--truth_rsd_noise_sigma_los', type=float, default=None,
                   help='Gaussian RSD-position noise sigma along LOS for truth [Mpc/h]. '
                        'Defaults to --truth_rsd_noise_sigma.')
    p.add_argument('--truth_rsd_noise_sigma_transverse', type=float, default=None,
                   help='Gaussian RSD-position noise sigma in each transverse '
                        'axis for truth [Mpc/h]. Defaults to 0 for --noise_axes los '
                        'or --truth_rsd_noise_sigma for --noise_axes all.')
    p.add_argument('--noise_axes', choices=('los', 'all'), default='los',
                   help='Legacy shortcut used only for unspecified directional '
                        'sigmas: los means transverse defaults to 0; all means '
                        'LOS and transverse both default to the legacy sigma.')
    p.add_argument('--noise_seed', type=int, default=None,
                   help='Random seed for RSD-position noise. Defaults to sim_id.')
    p.add_argument('--ylim', nargs=2, type=float, default=(0.5, 1.5),
                   metavar=('YMIN', 'YMAX'),
                   help='Y-limits for all ratio panels.')
    p.add_argument('--auto_ylim', action='store_true',
                   help='Use data-driven y-limits instead of --ylim.')
    p.add_argument('--binary_target_prior', type=float, default=None,
                   help='True voxel occupancy fraction for Bayesian binary-prior correction. '
                        'If unset and needed, auto-estimated from the calibrator.')
    p.add_argument('--binary_prior_calibrator', default=None,
                   help='Path to binary_prior_calibrator.npz. Defaults to '
                        '<checkpoint_dir>/binary_prior_calibrator.npz.')
    p.add_argument('--binary_train_prior', type=float, default=None,
                   help='Override loaded model.binary_train_prior.')
    return p.parse_args()


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _validate_noise_sigma(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f'{name} must be non-negative; got {value}')
    return value


def resolve_directional_noise_args(args) -> dict[str, float]:
    """Resolve legacy and per-direction RSD-position noise arguments."""
    mock_legacy = _validate_noise_sigma(
        'mock_rsd_noise_sigma', args.mock_rsd_noise_sigma)
    truth_legacy = _validate_noise_sigma(
        'truth_rsd_noise_sigma', args.truth_rsd_noise_sigma)
    legacy_trans_scale = 1.0 if args.noise_axes == 'all' else 0.0

    def choose(name: str, explicit, fallback: float) -> float:
        return _validate_noise_sigma(
            name, fallback if explicit is None else explicit)

    return {
        'mock_los': choose(
            'mock_rsd_noise_sigma_los',
            args.mock_rsd_noise_sigma_los,
            mock_legacy,
        ),
        'mock_transverse': choose(
            'mock_rsd_noise_sigma_transverse',
            args.mock_rsd_noise_sigma_transverse,
            legacy_trans_scale * mock_legacy,
        ),
        'truth_los': choose(
            'truth_rsd_noise_sigma_los',
            args.truth_rsd_noise_sigma_los,
            truth_legacy,
        ),
        'truth_transverse': choose(
            'truth_rsd_noise_sigma_transverse',
            args.truth_rsd_noise_sigma_transverse,
            legacy_trans_scale * truth_legacy,
        ),
    }


def apply_position_noise(pos_rsd: np.ndarray, sigma_los: float,
                         sigma_transverse: float, BoxSize: float,
                         rng: np.random.Generator, los_axis: int,
                         return_noise: bool = False):
    """Add periodic anisotropic Gaussian noise to redshift-space positions."""
    pos_noisy = np.asarray(pos_rsd, dtype=np.float64).copy()
    sigma_los = _validate_noise_sigma('sigma_los', sigma_los)
    sigma_transverse = _validate_noise_sigma(
        'sigma_transverse', sigma_transverse)
    noise = np.zeros_like(pos_noisy, dtype=np.float64)
    if pos_noisy.size > 0:
        if sigma_los > 0.0:
            noise[:, los_axis] = rng.normal(
                0.0, sigma_los, size=pos_noisy.shape[0])
        if sigma_transverse > 0.0:
            transverse_axes = [ax for ax in range(3) if ax != los_axis]
            noise[:, transverse_axes] = rng.normal(
                0.0, sigma_transverse,
                size=(pos_noisy.shape[0], len(transverse_axes)),
            )
    pos_out = np.mod(pos_noisy + noise, BoxSize).astype(np.float32)
    if return_noise:
        return pos_out, noise.astype(np.float32)
    return pos_out


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


def safe_ratio(num: np.ndarray, den: np.ndarray, floor_frac: float = 1e-4) -> np.ndarray:
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
    p_den_i = np.interp(k_num, k_den, p_den, left=np.nan, right=np.nan)
    return k_num, safe_ratio(p_num, p_den_i, floor_frac=floor_frac)


def auto_ylim_from_ratios(ratios: list[np.ndarray]) -> tuple[float, float]:
    chunks = [r[np.isfinite(r)] for r in ratios if np.any(np.isfinite(r))]
    if not chunks:
        return 0.5, 1.5
    vals = np.concatenate(chunks)
    vals = np.concatenate([vals, np.asarray([0.9, 1.0, 1.1])])
    lo, hi = np.nanpercentile(vals, [1, 99])
    pad = 0.1 * max(hi - lo, 0.2)
    return max(0.0, lo - pad), hi + pad


def compute_rsd_multipole_ratios(mock_cat: dict, true_cat: dict, BoxSize: float,
                                 z: float, cosmo: np.ndarray, ng: int,
                                 kmax: float, los_axis: int,
                                 mock_noise_los: float,
                                 mock_noise_transverse: float,
                                 truth_noise_los: float,
                                 truth_noise_transverse: float,
                                 noise_seed: int):
    pos_rsd_m = apply_rsd(mock_cat['pos'], mock_cat['vel'], BoxSize, z, cosmo,
                          axis=los_axis)
    pos_rsd_t = apply_rsd(true_cat['pos'], true_cat['vel'], BoxSize, z, cosmo,
                          axis=los_axis)

    rng_mock = np.random.default_rng(noise_seed + 101)
    rng_truth = np.random.default_rng(noise_seed + 202)
    pos_rsd_m = apply_position_noise(
        pos_rsd_m, mock_noise_los, mock_noise_transverse,
        BoxSize, rng_mock, los_axis)
    pos_rsd_t = apply_position_noise(
        pos_rsd_t, truth_noise_los, truth_noise_transverse,
        BoxSize, rng_truth, los_axis)

    k_m, p0_m, p2_m, p4_m = power_spectrum_multipoles(
        pos_rsd_m, None, ng, BoxSize, kmax=kmax, axis=los_axis)
    k_t, p0_t, p2_t, p4_t = power_spectrum_multipoles(
        pos_rsd_t, None, ng, BoxSize, kmax=kmax, axis=los_axis)

    return {
        'p0': interp_ratio(k_m, p0_m, k_t, p0_t),
        'p2': interp_ratio(k_m, p2_m, k_t, p2_t),
        'p4': interp_ratio(k_m, p4_m, k_t, p4_t),
        'mock_poles': (k_m, p0_m, p2_m, p4_m),
        'true_poles': (k_t, p0_t, p2_t, p4_t),
    }


def plot_ratios(ratios: dict, out_base: str, sim_id: int,
                noise_cfg: dict[str, float],
                los_axis: int, ylim, auto_ylim: bool):
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 1.0,
    })

    panels = [
        ('p0', r'Monopole $P_0$'),
        ('p2', r'Quadrupole $P_2$'),
        ('p4', r'Hexadecapole $P_4$'),
    ]
    ratio_arrays = [ratios[key][1] for key, _ in panels]
    if auto_ylim:
        ylim = auto_ylim_from_ratios(ratio_arrays)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4),
                             constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        k, ratio = ratios[key]
        ax.axhspan(0.9, 1.1, color='0.82', alpha=0.55, zorder=0,
                   label=r'$\pm 10\%$')
        ax.axhline(1.0, color='k', ls='--', lw=1.8, zorder=1)
        ax.semilogx(k, ratio, color='#1f4e79', lw=2.2, zorder=2)
        ax.set_title(title)
        ax.set_xlabel(r'$k$ [$h/{\rm Mpc}$]')
        ax.set_ylabel(r'$P_{\ell,\rm mock}/P_{\ell,\rm true}$')
        ax.set_ylim(*ylim)
        ax.grid(True, which='major', color='0.88', lw=0.8)
        ax.grid(True, which='minor', color='0.93', lw=0.45)
        ax.tick_params(axis='both', which='major', length=5.5, width=1.0)
        ax.tick_params(axis='both', which='minor', length=3.2, width=0.8)

    axes[0].legend(frameon=False, loc='best')
    axis_name = 'xyz'[los_axis]
    fig.suptitle(
        f'RSD multipole ratios: sim {sim_id:04d} | '
        f'mock noise LOS/T={noise_cfg["mock_los"]:g}/'
        f'{noise_cfg["mock_transverse"]:g} Mpc/h, '
        f'truth noise LOS/T={noise_cfg["truth_los"]:g}/'
        f'{noise_cfg["truth_transverse"]:g} Mpc/h | '
        f'LOS={axis_name}',
        fontsize=14,
        fontweight='bold',
    )

    png_path = out_base + '.png'
    pdf_path = out_base + '.pdf'
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return pdf_path, png_path


def maybe_apply_binary_prior_correction(args, model, ckpt_dir, cosmo_vals):
    if args.binary_train_prior is not None:
        print(f'Overriding model.binary_train_prior from {model.binary_train_prior} '
              f'to {args.binary_train_prior}', flush=True)
        model.binary_train_prior = float(args.binary_train_prior)

    btp = args.binary_target_prior
    if model.binary_train_prior is not None and btp is None:
        cal_path = (args.binary_prior_calibrator
                    or os.path.join(ckpt_dir, 'binary_prior_calibrator.npz'))
        if os.path.exists(cal_path):
            btp, info = estimate_target_prior_from_cosmology(cosmo_vals, cal_path)
            print(
                f'Auto-estimated binary_target_prior from cosmology: '
                f'pi_target={btp:.3e}, calibrator R2={info["calibrator_r2"]:.4f}',
                flush=True,
            )
        else:
            print(
                f'WARNING: model.binary_train_prior={model.binary_train_prior} '
                f'but no binary_target_prior/calibrator was supplied. '
                f'Binary prior correction will be skipped.',
                flush=True,
            )

    if model.binary_train_prior is not None and btp is not None:
        r = (btp / model.binary_train_prior) * \
            ((1 - model.binary_train_prior) / (1 - btp))
        print(f'Applying binary prior correction: pi_train={model.binary_train_prior}, '
              f'pi_target={btp:.3e}, odds-ratio r={r:.3e}', flush=True)
    return btp


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sc = cfg['sim_settings']
    dc = cfg['data_settings']
    tc = cfg['train_settings']

    nb = int(sc['nb'])
    ns_d = int(sc['ns_d'])
    ns_h = int(sc['ns_h'])
    nax = ns_h // nb
    nf = int(sc['nf'])
    Nmax = int(sc['Nmax'])
    BoxSize = float(dc['BoxSize'])
    z_snap = str(args.z_snap or dc['z_snap'])
    z = float(z_snap)

    lgMmin = float(sc['lgMmin'])
    lgMmax = float(sc['lgMmax'])
    vmin = float(sc['vmin'])
    vmax = float(sc['vmax'])
    cmin = float(sc['cmin'])
    cmax = float(sc['cmax'])

    n_cnn_tot = sum(1 if lt == 'cnn' else 2 for lt in sc['layers_types'])
    n_pad = (nf - 1) // 2 * n_cnn_tot
    device = torch.device(args.device)

    ckpt_dir = os.path.join(_REPO_ROOT, tc['checkpoint_dir'])
    ckpt_path = (args.checkpoint
                 if args.checkpoint
                 else os.path.join(ckpt_dir, 'charm_joint_best_val.pth'))
    output_dir = (args.output_dir
                  if args.output_dir
                  else os.path.join(ckpt_dir, 'inference_rsd_ratios'))
    os.makedirs(output_dir, exist_ok=True)
    noise_cfg = resolve_directional_noise_args(args)
    output_name_base = args.output_name or f'rsd_multipole_ratios_sim{args.sim_id:04d}'
    output_name = add_noise_suffix(output_name_base, noise_cfg, int(args.los_axis))
    out_base = os.path.join(output_dir, output_name)

    true_halo_dir = repo_path(args.true_halo_dir or dc['halo_hdf5_dir'])

    print(f'Using device: {device}', flush=True)
    print('Building model...', flush=True)
    model = build_model(cfg).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    print(f'Loading FastPM fields for sim {args.sim_id}...', flush=True)
    rho, vel = load_fastpm(dc['fastpm_dir'], args.sim_id, ns_d, z_snap)
    cosmo_vals = load_cosmology(dc['lh_cosmo_file'], args.sim_id)
    print(f'  Cosmology [Om,Ob,h,ns,s8]: {cosmo_vals}', flush=True)

    print('Building conditioning tensors...', flush=True)
    cond_x, cond_x_nsh, cond_cosmo = build_cond_tensors(
        rho, vel, cosmo_vals, nb, nax, n_pad
    )
    cond_x = cond_x.to(device)
    cond_x_nsh = cond_x_nsh.to(device)
    cond_cosmo = cond_cosmo.to(device)

    btp = maybe_apply_binary_prior_correction(args, model, ckpt_dir, cosmo_vals)

    print('Running model.sample()...', flush=True)
    with torch.no_grad():
        sample_out = model.sample(
            cond_x,
            cond_x_nsh,
            cond_cosmo,
            sample_binary=True,
            sample_multi=True,
            sample_m1=True,
            sample_mdiff=True,
            sample_vel=True,
            sample_conc=True,
            sample_pos=True,
            use_truth_masses=False,
            binary_target_prior=btp,
        )
    ntot_flat = np.asarray(sample_out['ntot'][0], dtype=np.float32)
    print(f'  Total mock halos predicted: {int(ntot_flat.sum())}', flush=True)

    print('Reconstructing mock catalog...', flush=True)
    vel_interps = build_dm_velocity_interpolators(vel, BoxSize)
    pos_mock, lgM_mock, vel_mock, conc_mock = reconstruct_catalog(
        sample_out, nb, nax, Nmax, lgMmin, lgMmax, vmin, vmax, cmin, cmax,
        BoxSize, vel_interps,
    )
    mock_cat = {
        'pos': pos_mock,
        'lgM': lgM_mock,
        'vel': vel_mock,
        'conc': conc_mock,
    }

    print(f'Loading truth: {true_halo_dir}/{args.sim_id}/', flush=True)
    true_cat, _ = load_true(true_halo_dir, args.sim_id, z_snap)

    noise_seed = int(args.noise_seed if args.noise_seed is not None else args.sim_id)
    print(
        'RSD position noise [Mpc/h]: '
        f'mock LOS={noise_cfg["mock_los"]:g}, '
        f'mock transverse={noise_cfg["mock_transverse"]:g}; '
        f'truth LOS={noise_cfg["truth_los"]:g}, '
        f'truth transverse={noise_cfg["truth_transverse"]:g}',
        flush=True,
    )
    print('Computing RSD multipole ratios...', flush=True)
    ratios = compute_rsd_multipole_ratios(
        mock_cat=mock_cat,
        true_cat=true_cat,
        BoxSize=BoxSize,
        z=z,
        cosmo=cosmo_vals,
        ng=int(args.ng),
        kmax=float(args.kmax),
        los_axis=int(args.los_axis),
        mock_noise_los=noise_cfg['mock_los'],
        mock_noise_transverse=noise_cfg['mock_transverse'],
        truth_noise_los=noise_cfg['truth_los'],
        truth_noise_transverse=noise_cfg['truth_transverse'],
        noise_seed=noise_seed,
    )

    npz_path = out_base + '.npz'
    np.savez(
        npz_path,
        k_p0=ratios['p0'][0],
        ratio_p0=ratios['p0'][1],
        k_p2=ratios['p2'][0],
        ratio_p2=ratios['p2'][1],
        k_p4=ratios['p4'][0],
        ratio_p4=ratios['p4'][1],
        mock_poles_k=ratios['mock_poles'][0],
        mock_p0=ratios['mock_poles'][1],
        mock_p2=ratios['mock_poles'][2],
        mock_p4=ratios['mock_poles'][3],
        true_poles_k=ratios['true_poles'][0],
        true_p0=ratios['true_poles'][1],
        true_p2=ratios['true_poles'][2],
        true_p4=ratios['true_poles'][3],
        sim_id=np.int32(args.sim_id),
        mock_rsd_noise_sigma_los=np.float32(noise_cfg['mock_los']),
        mock_rsd_noise_sigma_transverse=np.float32(noise_cfg['mock_transverse']),
        truth_rsd_noise_sigma_los=np.float32(noise_cfg['truth_los']),
        truth_rsd_noise_sigma_transverse=np.float32(noise_cfg['truth_transverse']),
        legacy_mock_rsd_noise_sigma=np.float32(args.mock_rsd_noise_sigma),
        legacy_truth_rsd_noise_sigma=np.float32(args.truth_rsd_noise_sigma),
        legacy_noise_axes=np.asarray(args.noise_axes),
        los_axis=np.int32(args.los_axis),
        noise_seed=np.int32(noise_seed),
        BoxSize=np.float32(BoxSize),
        z=np.float32(z),
        cosmo=cosmo_vals,
    )

    pdf_path, png_path = plot_ratios(
        ratios, out_base, args.sim_id,
        noise_cfg,
        int(args.los_axis),
        tuple(args.ylim),
        bool(args.auto_ylim),
    )
    print(f'Saved figure: {pdf_path}', flush=True)
    print(f'Saved figure: {png_path}', flush=True)
    print(f'Saved arrays:  {npz_path}', flush=True)


if __name__ == '__main__':
    main()
