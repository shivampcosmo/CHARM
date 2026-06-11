#!/usr/bin/env python
"""
build_density_balanced_training_shards.py
-----------------------------------------
Build CHARM-compatible HDF5 training/validation shards with sub-volumes
selected to be approximately uniform in per-subvolume mean background density.

The HDF5 training schema matches build_training_shards.py, with additional
ignored-by-training metadata that records the density-balanced selection.

Examples
--------
Build one train shard, excluding subvolumes already present in existing shards:

    python prep_data/build_density_balanced_training_shards.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v2vel.yaml \\
        --shard_rank 0

Build from all 512 subvolumes per sim, still selecting 128:

    python prep_data/build_density_balanced_training_shards.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v2vel.yaml \\
        --candidate_mode all --shard_rank 0

Make only diagnostic plots:

    python prep_data/build_density_balanced_training_shards.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v2vel.yaml \\
        --diagnostics_only

Exclude the union of multiple previous shard sets:

    python prep_data/build_density_balanced_training_shards.py \\
        --config run_configs/TRAIN_CHARM_JOINT_v2vel.yaml \\
        --exclude_source metadata \\
        --exclude_shard_dirs ../data/shards_Mmin5e12 \\
            ../data/shards_Mmin5e12_density_balanced_remaining_n64
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import h5py
import numpy as np
from tqdm import tqdm


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "charm"))

from build_training_shards import (  # noqa: E402
    count_ninp,
    create_shard,
    derive_padding,
    load_density_full,
    load_halo_h5,
    load_velocity_full,
    split_halos_to_subvols,
    subvols_padded,
    subvols_unpadded,
    write_rows,
)
from config_loader import load_config  # noqa: E402
from utils_data_prep_v2 import prep_halo_catalog  # noqa: E402


DEFAULT_DIAGNOSTIC_OUT_DIR = (
    "/mnt/ceph/users/spandey/CHARM_v2/CHARM/prep_data/test_output"
)
EXCLUDE_SOURCE_CODE = {"none": 0, "metadata": 1, "rng": 2}
CODE_EXCLUDE_SOURCE = {v: k for k, v in EXCLUDE_SOURCE_CODE.items()}
Z_DICT = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0, -1: 99.0}


class MetadataUnavailable(RuntimeError):
    """Raised when previous shard metadata cannot be read."""


@dataclass
class SelectionInfo:
    sim_id: int
    selected_ids: np.ndarray
    selected_means: np.ndarray
    selected_bins: np.ndarray
    bin_edges: np.ndarray
    all_hist: np.ndarray
    candidate_hist: np.ndarray
    selected_hist: np.ndarray
    excluded_count: int
    candidate_count: int
    active_bin_count: int
    constant_density: bool
    exclude_source: str
    density_min_all: float
    density_max_all: float
    density_min_candidate: float
    density_max_candidate: float


def repo_path(path: str | None) -> str | None:
    """Resolve config paths relative to the CHARM repo root."""
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


def resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    repo_candidate = os.path.join(_REPO_ROOT, path)
    if os.path.exists(repo_candidate):
        return os.path.abspath(repo_candidate)
    return os.path.abspath(path)


def normalize_config_paths(cfg: dict) -> dict:
    """Make file-system paths robust to the caller's current directory."""
    cfg = deepcopy(cfg)
    dc = cfg.get("data_settings", {})
    for key in ("fastpm_dir", "halo_hdf5_dir", "lh_cosmo_file", "shard_dir"):
        if key in dc and isinstance(dc[key], str):
            dc[key] = repo_path(dc[key])
    return cfg


def redshift_from_cfg(cfg: dict) -> float:
    snapnum = int(cfg["data_settings"]["snapnum"])
    if snapnum not in Z_DICT:
        raise ValueError(f"Unsupported snapnum={snapnum}; known values are {sorted(Z_DICT)}")
    return Z_DICT[snapnum]


def compute_subvol_mean_density(rho: np.ndarray, nb: int, nax: int) -> np.ndarray:
    """Return mean density contrast for each unpadded subvolume."""
    rho_sub = subvols_unpadded(rho, nb, nax)
    means = rho_sub.reshape(nb ** 3, -1).mean(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(means)):
        bad = int(np.size(means) - np.isfinite(means).sum())
        raise ValueError(f"Mean-density calculation produced {bad} non-finite values")
    return means.astype(np.float64)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values to bins, clipping under/overflow into edge bins."""
    n_bins = len(edges) - 1
    if n_bins <= 0:
        raise ValueError("Need at least one density bin")
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return np.clip(bins, 0, n_bins - 1).astype(np.int16)


def make_constant_edges(value: float, n_bins: int) -> np.ndarray:
    width = max(abs(float(value)) * 1.0e-6, 1.0e-6)
    return np.linspace(float(value) - width, float(value) + width, n_bins + 1)


def selection_seed_for_sim(cfg: dict, args: argparse.Namespace, isim: int) -> int:
    """Deterministic per-sim selection seed for one subsampling realization."""
    dc = cfg["data_settings"]
    base_seed = (
        int(args.selection_seed)
        if args.selection_seed is not None
        else int(dc.get("subvol_seed", 42))
    )
    return base_seed * 1000003 + int(isim) + 17


def old_rng_exclusion_ids(cfg: dict, isim: int, total_subvols: int) -> Set[int]:
    """Reconstruct the old random selection used by build_training_shards.py."""
    sc = cfg["sim_settings"]
    dc = cfg["data_settings"]
    n_old = min(int(sc.get("nsubvol_per_ji", 128)), total_subvols)
    seed = int(dc.get("subvol_seed", 42)) * 1000 + int(isim)
    rng = np.random.default_rng(seed=seed)
    return set(int(x) for x in rng.choice(total_subvols, n_old, replace=False))


def metadata_shard_paths(exclude_shard_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(exclude_shard_dir, "CHARM_train_shard_*.h5")))
    val_path = os.path.join(exclude_shard_dir, "CHARM_val_shard.h5")
    if os.path.exists(val_path):
        paths.append(val_path)
    return paths


def load_metadata_exclusions(
    exclude_shard_dir: str,
    sim_ids_needed: Iterable[int],
) -> Dict[int, Set[int]]:
    """Load actual previously selected subvolume IDs from shard metadata."""
    needed = set(int(x) for x in sim_ids_needed)
    if not os.path.isdir(exclude_shard_dir):
        raise MetadataUnavailable(f"Exclude shard dir does not exist: {exclude_shard_dir}")

    paths = metadata_shard_paths(exclude_shard_dir)
    if not paths:
        raise MetadataUnavailable(f"No CHARM shard files found in {exclude_shard_dir}")

    out: Dict[int, Set[int]] = {}
    saw_usable_metadata = False
    for path in paths:
        with h5py.File(path, "r", swmr=True) as f:
            if "metadata" not in f:
                raise MetadataUnavailable(f"{path} has no metadata group")
            mg = f["metadata"]
            if "sim_ids" not in mg or "subvol_ids" not in mg:
                raise MetadataUnavailable(
                    f"{path} is missing metadata/sim_ids or metadata/subvol_ids"
                )
            sims = mg["sim_ids"][:].astype(np.int64)
            subvols = mg["subvol_ids"][:].astype(np.int64)
            if sims.shape != subvols.shape:
                raise MetadataUnavailable(
                    f"{path} has mismatched sim_ids/subvol_ids shapes: "
                    f"{sims.shape} vs {subvols.shape}"
                )
            saw_usable_metadata = True
            for sim, subvol in zip(sims, subvols):
                sim_i = int(sim)
                if sim_i in needed:
                    out.setdefault(sim_i, set()).add(int(subvol))

    if not saw_usable_metadata:
        raise MetadataUnavailable(f"No usable metadata found in {exclude_shard_dir}")
    return out


def load_metadata_exclusions_many(
    exclude_shard_dirs: Sequence[str],
    sim_ids_needed: Iterable[int],
    require_all_dirs: bool,
) -> Tuple[Dict[int, Set[int]], List[str]]:
    """
    Load and union previous subvolume IDs from one or more shard directories.

    If require_all_dirs is true, any unreadable/missing shard directory raises.
    Otherwise, readable dirs are used and unreadable dirs are returned.
    """
    merged: Dict[int, Set[int]] = {}
    unavailable: List[str] = []
    for shard_dir in exclude_shard_dirs:
        try:
            part = load_metadata_exclusions(shard_dir, sim_ids_needed)
        except MetadataUnavailable:
            if require_all_dirs:
                raise
            unavailable.append(shard_dir)
            continue
        for sim_id, subvol_ids in part.items():
            merged.setdefault(int(sim_id), set()).update(int(x) for x in subvol_ids)

    if not merged:
        missing = ", ".join(unavailable) if unavailable else ", ".join(exclude_shard_dirs)
        raise MetadataUnavailable(f"No usable metadata found in exclude shard dirs: {missing}")
    return merged, unavailable


def prepare_exclusions(
    cfg: dict,
    sim_ids: Sequence[int],
    candidate_mode: str,
    exclude_source: str,
    exclude_shard_dirs: Sequence[str],
    total_subvols: int,
) -> Tuple[Dict[int, Set[int]], Dict[int, str]]:
    """Return per-simulation excluded IDs and their source label."""
    if candidate_mode == "all":
        return {int(s): set() for s in sim_ids}, {int(s): "none" for s in sim_ids}

    metadata_exclusions: Dict[int, Set[int]] = {}
    if exclude_source in ("auto", "metadata"):
        try:
            metadata_exclusions, _ = load_metadata_exclusions_many(
                exclude_shard_dirs=exclude_shard_dirs,
                sim_ids_needed=sim_ids,
                require_all_dirs=(exclude_source == "metadata"),
            )
        except MetadataUnavailable:
            if exclude_source == "metadata":
                raise
            metadata_exclusions = {}

    exclusions: Dict[int, Set[int]] = {}
    sources: Dict[int, str] = {}
    for isim in sim_ids:
        isim = int(isim)
        if isim in metadata_exclusions and len(metadata_exclusions[isim]) > 0:
            exclusions[isim] = set(metadata_exclusions[isim])
            sources[isim] = "metadata"
        elif exclude_source == "metadata":
            raise MetadataUnavailable(
                f"No previous subvolume metadata found for sim {isim} in "
                f"{', '.join(exclude_shard_dirs)}"
            )
        else:
            exclusions[isim] = old_rng_exclusion_ids(cfg, isim, total_subvols)
            sources[isim] = "rng"

        bad = [x for x in exclusions[isim] if x < 0 or x >= total_subvols]
        if bad:
            raise ValueError(
                f"Sim {isim} has out-of-range excluded subvol IDs, e.g. {bad[:5]}"
            )

    return exclusions, sources


def select_density_balanced_subvolumes(
    mean_density: np.ndarray,
    excluded_ids: Set[int],
    exclude_source: str,
    n_select: int,
    n_density_bins: int,
    seed: int,
    sim_id: int,
) -> SelectionInfo:
    """Select n_select subvolumes as uniformly as possible across density bins."""
    total_subvols = int(mean_density.shape[0])
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    if n_density_bins <= 0:
        raise ValueError(f"n_density_bins must be positive, got {n_density_bins}")

    all_ids = np.arange(total_subvols, dtype=np.int32)
    excluded = np.array(sorted(excluded_ids), dtype=np.int32)
    keep_mask = np.ones(total_subvols, dtype=bool)
    if excluded.size:
        keep_mask[excluded] = False
    candidate_ids = all_ids[keep_mask]
    candidate_means = mean_density[candidate_ids]

    if candidate_ids.size < n_select:
        raise ValueError(
            f"Sim {sim_id}: requested {n_select} subvolumes but only "
            f"{candidate_ids.size} candidates remain after excluding {len(excluded_ids)}"
        )
    if not np.all(np.isfinite(candidate_means)):
        raise ValueError(f"Sim {sim_id}: candidate mean densities contain non-finite values")

    rng = np.random.default_rng(seed=seed)
    lo = float(candidate_means.min())
    hi = float(candidate_means.max())
    constant_density = bool(np.isclose(lo, hi, rtol=0.0, atol=1.0e-12))

    if constant_density:
        edges = make_constant_edges(lo, n_density_bins)
        selected_ids = rng.choice(candidate_ids, n_select, replace=False).astype(np.int32)
        rng.shuffle(selected_ids)
        all_bins = np.zeros(total_subvols, dtype=np.int16)
        candidate_bins = np.zeros(candidate_ids.size, dtype=np.int16)
        selected_bins = np.zeros(n_select, dtype=np.int16)
        active_bin_count = 1
    else:
        edges = np.linspace(lo, hi, n_density_bins + 1)
        all_bins = assign_bins(mean_density, edges)
        candidate_bins = assign_bins(candidate_means, edges)
        available_counts = np.bincount(candidate_bins, minlength=n_density_bins)
        active_bins = np.flatnonzero(available_counts > 0)
        if active_bins.size == 0:
            raise ValueError(f"Sim {sim_id}: no active density bins")

        selected_counts = np.zeros(n_density_bins, dtype=np.int32)
        base = n_select // int(active_bins.size)
        for b in active_bins:
            selected_counts[b] = min(base, available_counts[b])

        remaining = n_select - int(selected_counts.sum())
        while remaining > 0:
            advanced = False
            for b in active_bins:
                if selected_counts[b] < available_counts[b]:
                    selected_counts[b] += 1
                    remaining -= 1
                    advanced = True
                    if remaining == 0:
                        break
            if not advanced:
                raise RuntimeError(
                    f"Sim {sim_id}: unable to allocate remaining density-bin quota"
                )

        selected_parts: List[np.ndarray] = []
        for b in active_bins:
            n_take = int(selected_counts[b])
            if n_take == 0:
                continue
            ids_in_bin = candidate_ids[candidate_bins == b].copy()
            rng.shuffle(ids_in_bin)
            selected_parts.append(ids_in_bin[:n_take])

        selected_ids = np.concatenate(selected_parts).astype(np.int32)
        rng.shuffle(selected_ids)
        selected_bins = assign_bins(mean_density[selected_ids], edges)
        active_bin_count = int(active_bins.size)

    if selected_ids.size != n_select:
        raise RuntimeError(
            f"Sim {sim_id}: selection produced {selected_ids.size}, expected {n_select}"
        )
    if np.unique(selected_ids).size != selected_ids.size:
        raise RuntimeError(f"Sim {sim_id}: selection contains duplicate subvolume IDs")
    if selected_ids.min() < 0 or selected_ids.max() >= total_subvols:
        raise RuntimeError(f"Sim {sim_id}: selection contains out-of-range IDs")
    overlap = set(int(x) for x in selected_ids).intersection(excluded_ids)
    if overlap:
        raise RuntimeError(
            f"Sim {sim_id}: selection overlaps excluded IDs, e.g. {sorted(overlap)[:5]}"
        )

    all_hist = np.bincount(all_bins, minlength=n_density_bins).astype(np.int32)
    candidate_hist = np.bincount(candidate_bins, minlength=n_density_bins).astype(np.int32)
    selected_hist = np.bincount(selected_bins, minlength=n_density_bins).astype(np.int32)

    selected_means = mean_density[selected_ids].astype(np.float64)
    return SelectionInfo(
        sim_id=int(sim_id),
        selected_ids=selected_ids,
        selected_means=selected_means,
        selected_bins=selected_bins.astype(np.int16),
        bin_edges=edges.astype(np.float64),
        all_hist=all_hist,
        candidate_hist=candidate_hist,
        selected_hist=selected_hist,
        excluded_count=int(len(excluded_ids)),
        candidate_count=int(candidate_ids.size),
        active_bin_count=active_bin_count,
        constant_density=constant_density,
        exclude_source=str(exclude_source),
        density_min_all=float(mean_density.min()),
        density_max_all=float(mean_density.max()),
        density_min_candidate=lo,
        density_max_candidate=hi,
    )


def select_random_subvolumes(
    mean_density: np.ndarray,
    excluded_ids: Set[int],
    exclude_source: str,
    n_select: int,
    n_density_bins: int,
    seed: int,
    sim_id: int,
) -> SelectionInfo:
    """Select n_select subvolumes uniformly at random from the candidate set."""
    total_subvols = int(mean_density.shape[0])
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    if n_density_bins <= 0:
        raise ValueError(f"n_density_bins must be positive, got {n_density_bins}")

    all_ids = np.arange(total_subvols, dtype=np.int32)
    excluded = np.array(sorted(excluded_ids), dtype=np.int32)
    keep_mask = np.ones(total_subvols, dtype=bool)
    if excluded.size:
        keep_mask[excluded] = False
    candidate_ids = all_ids[keep_mask]
    candidate_means = mean_density[candidate_ids]

    if candidate_ids.size < n_select:
        raise ValueError(
            f"Sim {sim_id}: requested {n_select} subvolumes but only "
            f"{candidate_ids.size} candidates remain after excluding {len(excluded_ids)}"
        )
    if not np.all(np.isfinite(candidate_means)):
        raise ValueError(f"Sim {sim_id}: candidate mean densities contain non-finite values")

    rng = np.random.default_rng(seed=seed)
    selected_ids = rng.choice(candidate_ids, n_select, replace=False).astype(np.int32)
    rng.shuffle(selected_ids)

    lo = float(candidate_means.min())
    hi = float(candidate_means.max())
    constant_density = bool(np.isclose(lo, hi, rtol=0.0, atol=1.0e-12))
    if constant_density:
        edges = make_constant_edges(lo, n_density_bins)
        all_bins = np.zeros(total_subvols, dtype=np.int16)
        candidate_bins = np.zeros(candidate_ids.size, dtype=np.int16)
        selected_bins = np.zeros(n_select, dtype=np.int16)
        active_bin_count = 1
    else:
        edges = np.linspace(lo, hi, n_density_bins + 1)
        all_bins = assign_bins(mean_density, edges)
        candidate_bins = assign_bins(candidate_means, edges)
        selected_bins = assign_bins(mean_density[selected_ids], edges)
        active_bin_count = int(np.count_nonzero(np.bincount(candidate_bins, minlength=n_density_bins)))

    if selected_ids.size != n_select:
        raise RuntimeError(
            f"Sim {sim_id}: selection produced {selected_ids.size}, expected {n_select}"
        )
    if np.unique(selected_ids).size != selected_ids.size:
        raise RuntimeError(f"Sim {sim_id}: selection contains duplicate subvolume IDs")
    if selected_ids.min() < 0 or selected_ids.max() >= total_subvols:
        raise RuntimeError(f"Sim {sim_id}: selection contains out-of-range IDs")
    overlap = set(int(x) for x in selected_ids).intersection(excluded_ids)
    if overlap:
        raise RuntimeError(
            f"Sim {sim_id}: selection overlaps excluded IDs, e.g. {sorted(overlap)[:5]}"
        )

    all_hist = np.bincount(all_bins, minlength=n_density_bins).astype(np.int32)
    candidate_hist = np.bincount(candidate_bins, minlength=n_density_bins).astype(np.int32)
    selected_hist = np.bincount(selected_bins, minlength=n_density_bins).astype(np.int32)

    selected_means = mean_density[selected_ids].astype(np.float64)
    return SelectionInfo(
        sim_id=int(sim_id),
        selected_ids=selected_ids,
        selected_means=selected_means,
        selected_bins=selected_bins.astype(np.int16),
        bin_edges=edges.astype(np.float64),
        all_hist=all_hist,
        candidate_hist=candidate_hist,
        selected_hist=selected_hist,
        excluded_count=int(len(excluded_ids)),
        candidate_count=int(candidate_ids.size),
        active_bin_count=active_bin_count,
        constant_density=constant_density,
        exclude_source=str(exclude_source),
        density_min_all=float(mean_density.min()),
        density_max_all=float(mean_density.max()),
        density_min_candidate=lo,
        density_max_candidate=hi,
    )


def select_subvolumes(
    selection_mode: str,
    mean_density: np.ndarray,
    excluded_ids: Set[int],
    exclude_source: str,
    n_select: int,
    n_density_bins: int,
    seed: int,
    sim_id: int,
) -> SelectionInfo:
    if selection_mode == "density":
        return select_density_balanced_subvolumes(
            mean_density=mean_density,
            excluded_ids=excluded_ids,
            exclude_source=exclude_source,
            n_select=n_select,
            n_density_bins=n_density_bins,
            seed=seed,
            sim_id=sim_id,
        )
    if selection_mode == "random":
        return select_random_subvolumes(
            mean_density=mean_density,
            excluded_ids=excluded_ids,
            exclude_source=exclude_source,
            n_select=n_select,
            n_density_bins=n_density_bins,
            seed=seed,
            sim_id=sim_id,
        )
    raise ValueError(f"Unsupported selection_mode={selection_mode!r}")


def process_sim_to_batch_selected(
    isim: int,
    cfg: dict,
    nb: int,
    nax: int,
    n_pad: int,
    idx: np.ndarray,
    Nmax: int,
    cosmo_vals: np.ndarray,
    rho: np.ndarray,
) -> Tuple[dict, List[int], List[int]]:
    """
    Load one simulation and process an explicit set of selected subvolume IDs.
    """
    sc = cfg["sim_settings"]
    dc = cfg["data_settings"]
    grid = nb * nax
    nvox = nax ** 3
    z_snap = str(dc["z_snap"])
    redshift = redshift_from_cfg(cfg)
    nsubvol = int(idx.size)

    vel = load_velocity_full(dc["fastpm_dir"], isim, grid, z_snap)
    rho_pad = np.pad(rho, n_pad, "wrap")
    vel_pad = np.pad(vel, [(0, 0)] + [(n_pad, n_pad)] * 3, "wrap")

    rho_sub_pad = subvols_padded(rho_pad, nb, nax, n_pad)
    vel_sub_pad = subvols_padded(vel_pad, nb, nax, n_pad)
    rho_sub_unp = subvols_unpadded(rho, nb, nax)
    vel_sub_unp = subvols_unpadded(vel.transpose(1, 2, 3, 0), nb, nax)
    vel_sub_unp = np.moveaxis(vel_sub_unp, -1, 1)

    halos = load_halo_h5(dc["halo_hdf5_dir"], isim, redshift)
    halo_sub = split_halos_to_subvols(halos, nb, nax)

    rho_sel = rho_sub_pad[idx].copy()
    vel_sel = vel_sub_pad[idx].copy()
    rho_nsh = rho_sub_unp[idx].copy()
    vel_nsh = vel_sub_unp[idx].copy()

    dm_cube = np.concatenate([rho_sel[:, np.newaxis], vel_sel], axis=1)
    dm_nsh_4d = np.concatenate([rho_nsh[:, np.newaxis], vel_nsh], axis=1)
    dm_nsh = np.moveaxis(dm_nsh_4d, 1, -1).reshape(nsubvol, nvox, -1)

    N_sel = halo_sub["N_halos"][idx]
    M_sel = halo_sub["M_halos"][idx]
    pos_sel = halo_sub["pos_halos"][idx]
    c_sim_sel = halo_sub["c_halos_sim"][idx]
    v_diff_sel = halo_sub["v_halos_diff"][idx]

    cosmo_flat = np.broadcast_to(
        cosmo_vals[np.newaxis, np.newaxis, :],
        (nsubvol, nvox, 5),
    ).copy()

    halo_dict = prep_halo_catalog(
        df_Mh=M_sel.astype(np.float32),
        df_Nh=N_sel.astype(np.int16),
        cosmo=cosmo_flat.reshape(nsubvol, nax, nax, nax, 5),
        Mmin=float(sc["lgMmin"]),
        Mmax=float(sc["lgMmax"]),
        Nmax=Nmax,
        rescale_sub=float(sc.get("rescale_sub", 0.0)),
        df_v=v_diff_sel.astype(np.float32),
        df_c=c_sim_sel.astype(np.float32),
        df_pos=pos_sel.astype(np.float32),
        vmin=float(sc["vmin"]),
        vmax=float(sc["vmax"]),
        cmin=float(sc["cmin"]),
        cmax=float(sc["cmax"]),
    )
    dens_dict = {
        "dm_cube": dm_cube.astype(np.float16),
        "dm_nsh": dm_nsh.astype(np.float16),
    }
    batch = {**halo_dict, **dens_dict}
    return batch, [int(isim)] * nsubvol, idx.astype(np.int32).tolist()


def create_audit_datasets(
    f: h5py.File,
    n_total: int,
    n_sims: int,
    n_select: int,
    n_density_bins: int,
    args: argparse.Namespace,
) -> None:
    mg = f["metadata"]
    mg.create_dataset("mean_density", shape=(n_total,), dtype="float32", chunks=(1,))
    mg.create_dataset("density_bin", shape=(n_total,), dtype="int16", chunks=(1,))

    sg = f.create_group("selection")
    sg.attrs["selection_mode"] = args.selection_mode
    sg.attrs["candidate_mode"] = args.candidate_mode
    sg.attrs["exclude_source_requested"] = args.exclude_source
    sg.attrs["exclude_shard_dirs"] = json.dumps(args.exclude_shard_dirs)
    sg.attrs["selection_seed"] = -1 if args.selection_seed is None else int(args.selection_seed)
    sg.attrs["n_select"] = int(n_select)
    sg.attrs["n_density_bins"] = int(n_density_bins)
    sg.attrs["exclude_source_codes"] = json.dumps(CODE_EXCLUDE_SOURCE, sort_keys=True)
    sg.create_dataset("sim_ids", shape=(n_sims,), dtype="int32")
    sg.create_dataset("excluded_counts", shape=(n_sims,), dtype="int32")
    sg.create_dataset("candidate_counts", shape=(n_sims,), dtype="int32")
    sg.create_dataset("selected_counts", shape=(n_sims,), dtype="int32")
    sg.create_dataset("active_bin_counts", shape=(n_sims,), dtype="int16")
    sg.create_dataset("constant_density", shape=(n_sims,), dtype="bool")
    sg.create_dataset("exclude_source_code", shape=(n_sims,), dtype="int8")
    sg.create_dataset("bin_edges", shape=(n_sims, n_density_bins + 1), dtype="float32")
    sg.create_dataset("all_hist", shape=(n_sims, n_density_bins), dtype="int32")
    sg.create_dataset("candidate_hist", shape=(n_sims, n_density_bins), dtype="int32")
    sg.create_dataset("selected_hist", shape=(n_sims, n_density_bins), dtype="int32")
    sg.create_dataset("selected_subvol_ids", shape=(n_sims, n_select), dtype="int32")


def write_selection_audit(
    f: h5py.File,
    sim_row: int,
    row: int,
    info: SelectionInfo,
) -> None:
    ns = int(info.selected_ids.size)
    f["metadata/mean_density"][row:row + ns] = info.selected_means.astype(np.float32)
    f["metadata/density_bin"][row:row + ns] = info.selected_bins.astype(np.int16)

    sg = f["selection"]
    sg["sim_ids"][sim_row] = int(info.sim_id)
    sg["excluded_counts"][sim_row] = int(info.excluded_count)
    sg["candidate_counts"][sim_row] = int(info.candidate_count)
    sg["selected_counts"][sim_row] = int(ns)
    sg["active_bin_counts"][sim_row] = int(info.active_bin_count)
    sg["constant_density"][sim_row] = bool(info.constant_density)
    sg["exclude_source_code"][sim_row] = EXCLUDE_SOURCE_CODE[info.exclude_source]
    sg["bin_edges"][sim_row] = info.bin_edges.astype(np.float32)
    sg["all_hist"][sim_row] = info.all_hist.astype(np.int32)
    sg["candidate_hist"][sim_row] = info.candidate_hist.astype(np.int32)
    sg["selected_hist"][sim_row] = info.selected_hist.astype(np.int32)
    sg["selected_subvol_ids"][sim_row] = info.selected_ids.astype(np.int32)


def selection_summary_record(info: SelectionInfo) -> dict:
    return {
        "sim_id": int(info.sim_id),
        "selected_count": int(info.selected_ids.size),
        "excluded_count": int(info.excluded_count),
        "candidate_count": int(info.candidate_count),
        "active_bin_count": int(info.active_bin_count),
        "constant_density": bool(info.constant_density),
        "exclude_source": info.exclude_source,
        "density_min_all": info.density_min_all,
        "density_max_all": info.density_max_all,
        "density_min_candidate": info.density_min_candidate,
        "density_max_candidate": info.density_max_candidate,
        "density_min_selected": float(info.selected_means.min()),
        "density_max_selected": float(info.selected_means.max()),
        "all_hist": info.all_hist.astype(int).tolist(),
        "candidate_hist": info.candidate_hist.astype(int).tolist(),
        "selected_hist": info.selected_hist.astype(int).tolist(),
        "selected_subvol_ids": info.selected_ids.astype(int).tolist(),
    }


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def load_halo_subvol_stats(cfg: dict, isim: int, nb: int, nax: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return total halo count and occupied-voxel fraction per subvolume."""
    redshift = redshift_from_cfg(cfg)
    fpath = os.path.join(
        cfg["data_settings"]["halo_hdf5_dir"],
        str(isim),
        f"halos_rockstar_200c_z{redshift}.h5",
    )
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Halo HDF5 not found: {fpath}")
    with h5py.File(fpath, "r") as f:
        N_halos = f["N_halos"][:]
    sub = subvols_unpadded(N_halos, nb, nax).reshape(nb ** 3, -1)
    total_halos = sub.sum(axis=1, dtype=np.int64).astype(np.int64)
    occ_frac = (sub > 0).mean(axis=1).astype(np.float64)
    return total_halos, occ_frac


def finite_hist_bins(values: np.ndarray, nbins: int) -> np.ndarray:
    values = np.asarray(values)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(0.0, 1.0, nbins + 1)
    if np.isclose(lo, hi):
        width = max(abs(lo) * 0.05, 1.0)
        lo -= width
        hi += width
    return np.linspace(lo, hi, nbins + 1)


def plot_diagnostics_for_sim(
    cfg: dict,
    args: argparse.Namespace,
    isim: int,
    excluded_ids: Set[int],
    exclude_source: str,
    out_dir: str,
) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-charm")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = cfg["sim_settings"]
    dc = cfg["data_settings"]
    nb = int(sc["nb"])
    nax = int(sc["ns_h"]) // nb
    grid = nb * nax
    rho = load_density_full(dc["fastpm_dir"], isim, grid, str(dc["z_snap"]))
    mean_density = compute_subvol_mean_density(rho, nb, nax)
    seed = selection_seed_for_sim(cfg, args, isim)
    info = select_subvolumes(
        selection_mode=args.selection_mode,
        mean_density=mean_density,
        excluded_ids=excluded_ids,
        exclude_source=exclude_source,
        n_select=int(args.n_select),
        n_density_bins=int(args.n_density_bins),
        seed=seed,
        sim_id=isim,
    )
    total_halos, occ_frac = load_halo_subvol_stats(cfg, isim, nb, nax)

    selected = info.selected_ids
    excluded = np.array(sorted(excluded_ids), dtype=np.int32)
    all_ids = np.arange(mean_density.size, dtype=np.int32)
    candidate_mask = np.ones(mean_density.size, dtype=bool)
    if excluded.size:
        candidate_mask[excluded] = False
    candidate = all_ids[candidate_mask]

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    density_bins = finite_hist_bins(mean_density, 36)
    ax.hist(mean_density, bins=density_bins, histtype="step", lw=2.0, color="0.25",
            label="all 512")
    if excluded.size:
        ax.hist(mean_density[excluded], bins=density_bins, histtype="step", lw=1.4,
                color="tab:orange", label="excluded")
    ax.hist(mean_density[candidate], bins=density_bins, histtype="step", lw=1.5,
            color="tab:blue", label="candidates")
    ax.hist(mean_density[selected], bins=density_bins, histtype="stepfilled", alpha=0.35,
            color="tab:red", label="selected")
    ax.set_xlabel("mean density contrast")
    ax.set_ylabel("subvolumes")
    ax.set_title(f"Sim {isim}: density selection")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    halo_bins = finite_hist_bins(total_halos, 36)
    ax.hist(total_halos, bins=halo_bins, histtype="step", lw=2.0, color="0.25",
            label="all 512")
    ax.hist(total_halos[selected], bins=halo_bins, histtype="stepfilled", alpha=0.35,
            color="tab:red", label="selected")
    ax.set_xlabel("total halos per subvolume")
    ax.set_ylabel("subvolumes")
    ax.set_title("halo-count distribution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    occ_bins = finite_hist_bins(occ_frac, 36)
    ax.hist(occ_frac, bins=occ_bins, histtype="step", lw=2.0, color="0.25",
            label="all 512")
    ax.hist(occ_frac[selected], bins=occ_bins, histtype="stepfilled", alpha=0.35,
            color="tab:red", label="selected")
    ax.set_xlabel("occupied voxel fraction")
    ax.set_ylabel("subvolumes")
    ax.set_title("occupancy distribution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.scatter(mean_density, total_halos, s=10, color="0.65", alpha=0.55,
               edgecolors="none", label="all 512")
    ax.scatter(mean_density[selected], total_halos[selected], s=22, color="tab:red",
               alpha=0.85, edgecolors="none", label="selected")
    ax.set_xlabel("mean density contrast")
    ax.set_ylabel("total halos per subvolume")
    ax.set_title("density vs halo count")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"selection_mode={args.selection_mode}, candidate_mode={args.candidate_mode}, "
        f"exclude_source={exclude_source}, "
        f"excluded={len(excluded_ids)}, selected={selected.size}",
        fontsize=11,
    )
    prefix = "random" if args.selection_mode == "random" else "density_balanced"
    out_path = os.path.join(
        out_dir,
        f"{prefix}_diagnostics_sim_{isim}_{args.candidate_mode}.png",
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    summary = selection_summary_record(info)
    summary.update(
        {
            "plot_path": out_path,
            "halo_total_all_mean": float(np.mean(total_halos)),
            "halo_total_selected_mean": float(np.mean(total_halos[selected])),
            "occ_frac_all_mean": float(np.mean(occ_frac)),
            "occ_frac_selected_mean": float(np.mean(occ_frac[selected])),
        }
    )
    return summary


def default_diagnostic_sim_ids(
    cfg: dict,
    split: str,
    n_diagnostic_sims: int,
    diagnostic_seed: int,
) -> List[int]:
    sc = cfg["sim_settings"]
    nsims_train = int(sc.get("nsims_train", 1800))
    nsims_val = int(sc.get("nsims_val", 100))
    if split == "val":
        start = nsims_train
        count = nsims_val
    else:
        start = 0
        count = nsims_train
    if count <= 0:
        raise ValueError(f"No simulations available for split={split}")
    n_pick = min(max(int(n_diagnostic_sims), 1), count)
    rng = np.random.default_rng(seed=int(diagnostic_seed))
    ids = start + rng.choice(count, size=n_pick, replace=False)
    return sorted(set(int(x) for x in ids))


def make_diagnostics(cfg: dict, args: argparse.Namespace, split: str) -> dict:
    sim_ids = (
        [int(x) for x in args.diagnostic_sim_ids]
        if args.diagnostic_sim_ids
        else default_diagnostic_sim_ids(
            cfg,
            split,
            n_diagnostic_sims=int(args.n_diagnostic_sims),
            diagnostic_seed=int(args.diagnostic_seed),
        )
    )
    total_subvols = int(cfg["sim_settings"]["nb"]) ** 3
    exclusions, sources = prepare_exclusions(
        cfg=cfg,
        sim_ids=sim_ids,
        candidate_mode=args.candidate_mode,
        exclude_source=args.exclude_source,
        exclude_shard_dirs=args.exclude_shard_dirs,
        total_subvols=total_subvols,
    )
    out_dir = args.diagnostic_out_dir
    records = []
    for isim in tqdm(sim_ids, desc="diagnostics", leave=True):
        records.append(
            plot_diagnostics_for_sim(
                cfg=cfg,
                args=args,
                isim=isim,
                excluded_ids=exclusions[int(isim)],
                exclude_source=sources[int(isim)],
                out_dir=out_dir,
            )
        )

    payload = {
        "created_at_unix": time.time(),
        "config": args.config,
        "split": split,
        "selection_mode": args.selection_mode,
        "candidate_mode": args.candidate_mode,
        "exclude_source_requested": args.exclude_source,
        "exclude_shard_dirs": args.exclude_shard_dirs,
        "selection_seed": args.selection_seed,
        "diagnostic_sim_ids": sim_ids,
        "n_diagnostic_sims": int(args.n_diagnostic_sims),
        "diagnostic_seed": int(args.diagnostic_seed),
        "records": records,
    }
    prefix = "random" if args.selection_mode == "random" else "density_balanced"
    summary_path = os.path.join(out_dir, f"{prefix}_diagnostics_{split}.json")
    write_json_atomic(summary_path, payload)
    print(f"[diagnostics] wrote {len(records)} plot(s) and {summary_path}", flush=True)
    return payload


def determine_split_and_sims(
    cfg: dict,
    split: str,
    shard_rank: int,
) -> Tuple[str, str, List[int], str]:
    sc = cfg["sim_settings"]
    dc = cfg["data_settings"]
    n_shards = int(dc["n_shards"])
    nsims_train = int(sc.get("nsims_train", 1800))
    nsims_val = int(sc.get("nsims_val", 100))

    if shard_rank >= n_shards:
        split = "val"

    if split == "val":
        sim_ids = list(range(nsims_train, nsims_train + nsims_val))
        shard_id = "val"
        filename = "CHARM_val_shard.h5"
    else:
        if shard_rank < 0 or shard_rank >= n_shards:
            raise ValueError(f"Training shard_rank must be in [0, {n_shards}); got {shard_rank}")
        sim_ids = list(range(shard_rank, nsims_train, n_shards))
        shard_id = str(shard_rank)
        filename = f"CHARM_train_shard_{shard_rank}.h5"

    return split, shard_id, sim_ids, filename


def build_density_balanced_shard(
    cfg: dict,
    args: argparse.Namespace,
    shard_id: str,
    sim_ids_all: Sequence[int],
    out_path: str,
) -> dict:
    sc = cfg["sim_settings"]
    dc = cfg["data_settings"]
    nb = int(sc["nb"])
    nax = int(sc["ns_h"]) // nb
    nvox = nax ** 3
    n_pad = derive_padding(cfg)
    D_pad = nax + 2 * n_pad
    ninp = count_ninp(cfg)
    Nmax = int(sc["Nmax"])
    n_select = int(args.n_select)
    n_density_bins = int(args.n_density_bins)
    n_total = len(sim_ids_all) * n_select
    total_subvols = nb ** 3
    nsims_per_batch = int(cfg["train_settings"]["nsims_per_batch"])

    if n_total <= 0:
        raise ValueError("No output rows requested")
    if n_total % nsims_per_batch != 0:
        raise ValueError(
            f"n_total_subvols ({n_total}) is not divisible by "
            f"train_settings.nsims_per_batch ({nsims_per_batch})"
        )
    if n_select > total_subvols:
        raise ValueError(f"n_select={n_select} exceeds total subvolumes={total_subvols}")

    exclusions, sources = prepare_exclusions(
        cfg=cfg,
        sim_ids=sim_ids_all,
        candidate_mode=args.candidate_mode,
        exclude_source=args.exclude_source,
        exclude_shard_dirs=args.exclude_shard_dirs,
        total_subvols=total_subvols,
    )

    lh_cosmo = np.loadtxt(dc["lh_cosmo_file"])
    summary = {
        "created_at_unix": time.time(),
        "config": args.config,
        "shard_id": str(shard_id),
        "out_path": out_path,
        "selection_mode": args.selection_mode,
        "candidate_mode": args.candidate_mode,
        "exclude_source_requested": args.exclude_source,
        "exclude_shard_dirs": args.exclude_shard_dirs,
        "selection_seed": args.selection_seed,
        "n_select": n_select,
        "n_density_bins": n_density_bins,
        "n_total_subvols": n_total,
        "nsims": len(sim_ids_all),
        "sim_records": [],
    }

    print(
        f"[density-balanced shard {shard_id}] {len(sim_ids_all)} sims x "
        f"{n_select} subvols = {n_total} rows -> {out_path}",
        flush=True,
    )

    if args.dry_run:
        for isim in tqdm(sim_ids_all, desc=f"dry-run shard {shard_id}", leave=True):
            rho = load_density_full(dc["fastpm_dir"], int(isim), nb * nax, str(dc["z_snap"]))
            mean_density = compute_subvol_mean_density(rho, nb, nax)
            seed = selection_seed_for_sim(cfg, args, int(isim))
            info = select_subvolumes(
                selection_mode=args.selection_mode,
                mean_density=mean_density,
                excluded_ids=exclusions[int(isim)],
                exclude_source=sources[int(isim)],
                n_select=n_select,
                n_density_bins=n_density_bins,
                seed=seed,
                sim_id=int(isim),
            )
            summary["sim_records"].append(selection_summary_record(info))
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output shard exists; pass --overwrite to replace: {out_path}")
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    try:
        with create_shard(tmp_path, n_total, nvox, ninp, D_pad, Nmax, cfg) as f:
            create_audit_datasets(f, n_total, len(sim_ids_all), n_select, n_density_bins, args)
            row = 0
            mu_sig_written = False
            for sim_row, isim in enumerate(tqdm(sim_ids_all, desc=f"shard {shard_id}", leave=True)):
                isim = int(isim)
                rho = load_density_full(dc["fastpm_dir"], isim, nb * nax, str(dc["z_snap"]))
                mean_density = compute_subvol_mean_density(rho, nb, nax)
                seed = selection_seed_for_sim(cfg, args, isim)
                info = select_subvolumes(
                    selection_mode=args.selection_mode,
                    mean_density=mean_density,
                    excluded_ids=exclusions[isim],
                    exclude_source=sources[isim],
                    n_select=n_select,
                    n_density_bins=n_density_bins,
                    seed=seed,
                    sim_id=isim,
                )
                batch, sids, svids = process_sim_to_batch_selected(
                    isim=isim,
                    cfg=cfg,
                    nb=nb,
                    nax=nax,
                    n_pad=n_pad,
                    idx=info.selected_ids,
                    Nmax=Nmax,
                    cosmo_vals=lh_cosmo[isim],
                    rho=rho,
                )
                write_rows(f, row, batch, sids, svids)
                write_selection_audit(f, sim_row, row, info)

                if not mu_sig_written and "mu_all" in batch:
                    mg = f["metadata"]
                    mg.create_dataset("mu_all", data=batch["mu_all"])
                    mg.create_dataset("sig_all", data=batch["sig_all"])
                    mu_sig_written = True

                summary["sim_records"].append(selection_summary_record(info))
                row += n_select
                f.flush()

            if row != n_total:
                raise RuntimeError(f"Expected to write {n_total} rows, wrote {row}")

        os.replace(tmp_path, out_path)
        summary_path = f"{out_path}.selection_summary.json"
        write_json_atomic(summary_path, summary)
        print(f"[density-balanced shard {shard_id}] done: {out_path}", flush=True)
        print(f"[density-balanced shard {shard_id}] summary: {summary_path}", flush=True)
        return summary
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build density-balanced CHARM HDF5 shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Training YAML config path")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--shard_rank", type=int, default=None,
                   help="Training shard rank. Defaults to SLURM_ARRAY_TASK_ID or 0.")
    p.add_argument("--candidate_mode", default="remaining", choices=["remaining", "all"],
                   help="Select from remaining subvolumes or all 512 subvolumes.")
    p.add_argument("--selection_mode", default="density", choices=["density", "random"],
                   help="density = uniform over density bins; random = direct random draw.")
    p.add_argument("--exclude_shard_dir", default=None,
                   help="Shard dir containing previous metadata; defaults to config shard_dir.")
    p.add_argument("--exclude_shard_dirs", nargs="*", default=None,
                   help="One or more shard dirs whose metadata/subvol_ids are union-excluded.")
    p.add_argument("--exclude_source", default="auto", choices=["auto", "metadata", "rng"],
                   help="How to identify previously selected subvolumes in remaining mode.")
    p.add_argument("--out_dir", default=None,
                   help="Output shard directory. Defaults to a density-balanced sibling dir.")
    p.add_argument("--n_select", type=int, default=128,
                   help="Number of subvolumes to select per simulation.")
    p.add_argument("--n_density_bins", type=int, default=16,
                   help="Number of fixed-width density bins per simulation.")
    p.add_argument("--selection_seed", type=int, default=None,
                   help="Base random seed for the density-balanced subvolume realization.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing output shard if present.")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute selections and summaries without writing HDF5.")
    p.add_argument("--diagnostics_only", action="store_true",
                   help="Only save diagnostic plots/summaries; do not build a shard.")
    p.add_argument("--make_diagnostics", action="store_true",
                   help="Save diagnostic plots after dry-run/build.")
    p.add_argument("--diagnostic_sim_ids", type=int, nargs="*", default=None,
                   help="Specific simulation IDs for diagnostic plots.")
    p.add_argument("--n_diagnostic_sims", type=int, default=3,
                   help="Number of random diagnostic cosmologies when IDs are not provided.")
    p.add_argument("--diagnostic_seed", type=int, default=12345,
                   help="Seed for choosing random diagnostic cosmologies.")
    p.add_argument("--diagnostic_out_dir", default=DEFAULT_DIAGNOSTIC_OUT_DIR,
                   help="Directory for diagnostic plots and summaries.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.config = resolve_config_path(args.config)
    cfg = normalize_config_paths(load_config(args.config))

    if args.exclude_shard_dirs:
        args.exclude_shard_dirs = [repo_path(x) for x in args.exclude_shard_dirs]
    else:
        if args.exclude_shard_dir is None:
            args.exclude_shard_dir = cfg["data_settings"]["shard_dir"]
        args.exclude_shard_dirs = [repo_path(args.exclude_shard_dir)]
    args.exclude_shard_dir = args.exclude_shard_dirs[0]
    args.diagnostic_out_dir = repo_path(args.diagnostic_out_dir)

    if args.out_dir is None:
        base = cfg["data_settings"]["shard_dir"].rstrip(os.sep)
        tag = "random" if args.selection_mode == "random" else "density_balanced"
        args.out_dir = f"{base}_{tag}_{args.candidate_mode}"
    args.out_dir = repo_path(args.out_dir)

    if args.shard_rank is None:
        args.shard_rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    split, shard_id, sim_ids, filename = determine_split_and_sims(
        cfg=cfg,
        split=args.split,
        shard_rank=int(args.shard_rank),
    )

    if args.diagnostics_only:
        make_diagnostics(cfg, args, split)
        return

    out_path = os.path.join(args.out_dir, filename)
    build_density_balanced_shard(
        cfg=cfg,
        args=args,
        shard_id=shard_id,
        sim_ids_all=sim_ids,
        out_path=out_path,
    )

    if args.make_diagnostics:
        make_diagnostics(cfg, args, split)


if __name__ == "__main__":
    main()
