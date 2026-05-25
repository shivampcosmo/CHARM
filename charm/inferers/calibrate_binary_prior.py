"""
calibrate_binary_prior.py
-------------------------
Fit a polynomial regressor mapping cosmological parameters
(Ωm, Ωb, h, ns, σ8) → expected number of OCCUPIED VOXELS (voxels with
≥1 halo) for the run's Mmin and box volume, using the training
simulations as ground truth.

Default: degree-3 polynomial with optional Ridge (L2) regularisation.
Feature counts for d=5 cosmology parameters:
  degree 2 →  21 features  (suitable for ~200 training sims)
  degree 3 →  56 features  (recommended for ~1800 training sims, default)
  degree 4 → 126 features  (feasible for very large sets)

The result is saved as a .npz file. run_inference_v2.py auto-loads it
to fill in --binary_target_prior from the simulation's cosmology alone,
removing the need to know the true halo count at inference time.

Why we need this
----------------
Subsample / alpha modes train the binary head under a balanced prior
(π_train = 0.5), so the trained pw_occ is calibrated as
p(occ | x, π=0.5). At inference we Bayesian-correct to the true
per-voxel occupancy fraction π_target = N_occ_voxels / ns_h³, but π
varies by up to ~50× across cosmologies in the latin hypercube, so a
single global value is insufficient.

IMPORTANT — predict N_occ_voxels, not N_halos
----------------------------------------------
The binary head predicts whether each voxel has ≥1 halo (occupancy).
The Bayesian correction requires π_target = N_occ_voxels / ns_h³, the
fraction of voxels that are occupied.  This is NOT equal to
N_halos / ns_h³ because multiple halos can share a voxel.  Across the
Quijote LH the multi-halo factor N_halos / N_occ_voxels ranges from
~1.26 (sparse cosmologies) to ~1.62 (dense cosmologies), and using
N_halos instead of N_occ_voxels inflates the correction odds-ratio r by
up to 2× for the densest simulations, causing 20–40% occupancy errors.

Usage
-----
    # Mmin=5e12, 1800 sims, degree-3 polynomial (default):
    python charm/inferers/calibrate_binary_prior.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v0.yaml

    # Explicit degree and Ridge regularisation:
    python charm/inferers/calibrate_binary_prior.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v0.yaml \\
        --poly_degree 3 --ridge_alpha 0.01

    # Legacy degree-2 (for small training sets):
    python charm/inferers/calibrate_binary_prior.py \\
        --config run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14_balanced.yaml \\
        --poly_degree 2

Saves to <checkpoint_dir>/binary_prior_calibrator.npz by default.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations_with_replacement

import h5py
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config


def _poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Map (N, d) standardised cosmology vectors to full polynomial features
    up to `degree` inclusive, including all cross-terms.

    Feature counts for d=5:
      degree 2 → 21,  degree 3 → 56,  degree 4 → 126

    For degree=2 the ordering is identical to the old _quadratic_features,
    so existing saved .npz calibrators remain compatible.
    """
    n, d = x.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            term = np.ones(n)
            for idx in combo:
                term = term * x[:, idx]
            cols.append(term)
    return np.stack(cols, axis=1)


def _ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge regression: solve (X^T X + alpha*I) c = X^T y."""
    A = X.T @ X
    if alpha > 0.0:
        A[np.arange(A.shape[0]), np.arange(A.shape[0])] += alpha
    return np.linalg.solve(A, X.T @ y)


def _cross_val_rmse(X: np.ndarray, y: np.ndarray,
                    alpha: float, k: int = 5) -> float:
    """k-fold cross-validated RMSE in dex (unbiased error estimate)."""
    n = len(y)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    sq_errors = []
    for fold in folds:
        val_mask = np.zeros(n, dtype=bool)
        val_mask[fold] = True
        c = _ridge_solve(X[~val_mask], y[~val_mask], alpha)
        sq_errors.append(((y[val_mask] - X[val_mask] @ c) ** 2).sum())
    return float(np.sqrt(sum(sq_errors) / n))


def _resolve_halo_dir(cfg: dict) -> str:
    """Resolve the per-sim halo HDF5 directory from the config."""
    dc   = cfg['data_settings']
    cand = dc.get('halo_hdf5_dir')
    if cand and os.path.isdir(cand):
        return cand
    # Try auto-deriving from Mmin_cut_str (same logic as config_loader)
    mmin_str = dc.get('Mmin_cut_str', '')
    if mmin_str:
        fallback = os.path.join(_REPO_ROOT, '..', 'data',
                                f'halos_Mmin{mmin_str}')
        if os.path.isdir(fallback):
            return fallback
        raise FileNotFoundError(
            f"Could not find halo HDF5 directory. Tried:\n"
            f"  {cand!r}\n  {fallback!r}\n"
            "Set 'data_settings.halo_hdf5_dir' explicitly in the config."
        )
    raise FileNotFoundError(
        f"Could not find halo HDF5 directory. Tried: {cand!r}. "
        "Set 'data_settings.halo_hdf5_dir' or 'data_settings.Mmin_cut_str' "
        "in the config."
    )


def _per_sim_halo_path(halo_dir: str, sim_id: int,
                       mass_type: str, z_snap: str) -> str:
    """Layout: <halo_dir>/<sim_id>/halos_<mass_type>_z<z>.h5."""
    return os.path.join(
        halo_dir, str(sim_id),
        f'halos_{mass_type}_z{z_snap}.h5'
    )


def collect_training_data(cfg: dict, halo_dir: str,
                          n_skip_missing_max: int = 5
                          ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Load (cosmology, occupied-voxel count) for every training simulation.

    Returns
    -------
    cosmo_arr : (N, 5)
    log_n_arr : (N,)    log10(N_occ_voxels)  — voxels with ≥1 halo
    sims_used : list of sim ids
    """
    sc = cfg['sim_settings']
    dc = cfg['data_settings']
    nsims_train = sc['nsims_train']
    mass_type   = sc['mass_type']
    z_snap      = dc['z_snap']

    cosmo_full = np.loadtxt(dc['lh_cosmo_file'])

    cosmo_list, n_list, sims = [], [], []
    skipped = 0
    for sim_id in range(nsims_train):
        fp = _per_sim_halo_path(halo_dir, sim_id, mass_type, z_snap)
        if not os.path.exists(fp):
            skipped += 1
            if skipped <= n_skip_missing_max:
                print(f'  [skip] sim {sim_id}: missing {fp}', flush=True)
            continue
        with h5py.File(fp, 'r') as f:
            # Count occupied voxels (voxels with ≥1 halo), NOT total halos.
            # The binary head predicts per-voxel occupancy, so π_target must
            # be N_occ_voxels / ns_h³.  Using N_halos instead inflates the
            # Bayesian correction odds-ratio r by up to 2× for dense sims.
            n = int((f['N_halos'][:] > 0).sum())
        if n <= 0:
            print(f'  [skip] sim {sim_id}: zero occupied voxels', flush=True)
            continue
        cosmo_list.append(cosmo_full[sim_id])
        n_list.append(n)
        sims.append(sim_id)

    if skipped > n_skip_missing_max:
        print(f'  ({skipped - n_skip_missing_max} more missing sims '
              'silently skipped)', flush=True)

    cosmo_arr = np.asarray(cosmo_list, dtype=np.float64)
    log_n_arr = np.log10(np.asarray(n_list, dtype=np.float64))
    return cosmo_arr, log_n_arr, sims


def fit(cosmo: np.ndarray, log_n: np.ndarray,
        degree: int = 3, ridge_alpha: float = 0.01
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Fit a standardised degree-`degree` polynomial Ridge regressor.

    Parameters
    ----------
    cosmo       : (N, 5) raw cosmology array
    log_n       : (N,)   log10(N_halos) targets
    degree      : polynomial degree (default 3)
    ridge_alpha : L2 regularisation strength (default 0.01)

    Returns
    -------
    coeffs        : (n_features,)
    feature_means : (5,)  — standardisation means
    feature_stds  : (5,)  — standardisation stds
    r2            : float — training R² (optimistic upper bound)
    cv_rmse       : float — 5-fold CV RMSE in dex (unbiased)
    """
    means = cosmo.mean(axis=0)
    stds  = cosmo.std(axis=0)
    stds  = np.where(stds < 1e-12, 1.0, stds)
    z     = (cosmo - means) / stds

    X      = _poly_features(z, degree)
    coeffs = _ridge_solve(X, log_n, ridge_alpha)

    pred   = X @ coeffs
    ss_res = float(((log_n - pred) ** 2).sum())
    ss_tot = float(((log_n - log_n.mean()) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cv_rmse = _cross_val_rmse(X, log_n, ridge_alpha)
    return coeffs, means, stds, r2, cv_rmse


def predict(cosmo_vec: np.ndarray,
            coeffs: np.ndarray,
            feature_means: np.ndarray,
            feature_stds: np.ndarray,
            degree: int = 2) -> float:
    """
    Predict log10(N_occ_voxels) for a single (5,) cosmology vector.

    N_occ_voxels = number of grid voxels that contain ≥1 halo.
    Dividing by ns_h³ gives the voxel occupancy fraction π_target used
    in the Bayesian binary-prior correction at inference time.

    `degree` defaults to 2 for backward compatibility with .npz files
    produced before the degree-3 upgrade. New files store 'poly_degree'
    explicitly; callers should read and pass it.
    """
    cosmo_vec = np.asarray(cosmo_vec, dtype=np.float64).reshape(1, 5)
    z = (cosmo_vec - feature_means) / feature_stds
    X = _poly_features(z, degree)
    return float((X @ coeffs)[0])


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to YAML config (the same one used for training).')
    p.add_argument('--output', default=None,
                   help='Path for the .npz calibrator file. Defaults to '
                        '<checkpoint_dir>/binary_prior_calibrator.npz.')
    p.add_argument('--halo_dir', default=None,
                   help='Override directory of per-sim halo HDF5 files. '
                        'Auto-derived from config if not given.')
    p.add_argument('--poly_degree', type=int, default=3,
                   help='Degree of the polynomial feature expansion. '
                        'degree 2→21 features, degree 3→56, degree 4→126. '
                        'Degree 3 is recommended for ≥500 training sims.')
    p.add_argument('--ridge_alpha', type=float, default=0.01,
                   help='L2 Ridge regularisation strength. '
                        '0 = ordinary least squares. '
                        'Small values (0.01–0.1) stabilise the fit without '
                        'meaningfully biasing the coefficients.')
    return p.parse_args()


def main():
    args = parse_args()

    cfg      = load_config(args.config)
    halo_dir = args.halo_dir or _resolve_halo_dir(cfg)
    print(f'Halo dir  : {halo_dir}', flush=True)
    print(f'Poly degree: {args.poly_degree}  '
          f'Ridge alpha: {args.ridge_alpha}', flush=True)

    nsims_train = cfg['sim_settings']['nsims_train']
    print(f'Collecting training data for {nsims_train} sims...', flush=True)
    cosmo, log_n, sims = collect_training_data(cfg, halo_dir)
    print(f'  Used {len(sims)} / {nsims_train} sims.', flush=True)
    print(f'  log10(N_occ_voxels) range: {log_n.min():.3f} .. {log_n.max():.3f} '
          f'  (N_occ: {int(10**log_n.min()):,} .. {int(10**log_n.max()):,})',
          flush=True)

    coeffs, means, stds, r2, cv_rmse = fit(
        cosmo, log_n,
        degree=args.poly_degree,
        ridge_alpha=args.ridge_alpha,
    )

    n_features = len(coeffs)
    print(f'  Fit: {n_features} features, '
          f'train R² = {r2:.5f}, '
          f'5-fold CV RMSE = {cv_rmse:.4f} dex',
          flush=True)

    # Per-sim residual diagnostics on the full training set
    z     = (cosmo - means) / stds
    X_all = _poly_features(z, args.poly_degree)
    pred  = X_all @ coeffs
    resid = log_n - pred
    print(f'  Train residuals: rms = {resid.std():.4f} dex  '
          f'median |err| = {np.median(np.abs(resid)):.4f} dex  '
          f'max |err| = {np.abs(resid).max():.4f} dex',
          flush=True)

    ns_h     = int(cfg['sim_settings']['ns_h'])
    out_path = (args.output
                or os.path.join(cfg['train_settings']['checkpoint_dir'],
                                'binary_prior_calibrator.npz'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        coeffs        = coeffs.astype(np.float64),
        feature_means = means.astype(np.float64),
        feature_stds  = stds.astype(np.float64),
        poly_degree   = np.int64(args.poly_degree),
        ridge_alpha   = np.float64(args.ridge_alpha),
        ns_h          = np.int64(ns_h),
        sims_used     = np.asarray(sims, dtype=np.int64),
        r2            = np.float64(r2),
        cv_rmse       = np.float64(cv_rmse),
        feature_order = np.asarray(['Om', 'Ob', 'h', 'ns', 's8']),
        Mmin_cut      = np.float64(cfg['data_settings']['Mmin_cut']),
        z_snap        = np.asarray(cfg['data_settings']['z_snap']),
    )
    print(f'Saved calibrator → {out_path}', flush=True)


if __name__ == '__main__':
    main()
