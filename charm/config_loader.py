"""
config_loader.py
----------------
Shared YAML loader supporting an `extends:` inheritance chain plus
mass-cut-namespaced data path derivation.

A child config may declare:
    extends: ../BASE_CONFIG.yaml

The parent is resolved relative to the child's directory and loaded first
(recursively, so chains are fine). Child values then deep-merge over the
parent:
  - dicts are merged recursively
  - scalars and lists are replaced wholesale by the child

List-replace semantics keep semantics predictable, especially for the
`phases:` list (each run tunes it independently — no append-vs-replace
ambiguity).

After merging, ``data_settings.halo_hdf5_dir`` and ``data_settings.shard_dir``
are auto-derived from ``data_settings.Mmin_cut_str`` (and the optional
``data_settings.data_root``) if not already set. This lets a single
``Mmin_cut_str`` change in the YAML cleanly produce a new, isolated
folder for both the per-sim halo HDF5 files and the assembled shards —
e.g. ``../data/halos_Mmin5e12/`` and ``../data/shards_Mmin5e12/``.

Used by:
  - charm/run_charm_joint_ddp.py  (training)
  - prep_data/process_halos_quijote_v2.py  (per-sim halo processing)
  - prep_data/build_training_shards.py  (shard assembly)
"""

from __future__ import annotations

import os
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _derive_data_paths(cfg: dict) -> dict:
    """Auto-fill data_settings.halo_hdf5_dir and shard_dir from Mmin_cut_str.

    Naming scheme:
        halo_hdf5_dir = {data_root}/halos_Mmin{Mmin_cut_str}
        shard_dir     = {data_root}/shards_Mmin{Mmin_cut_str}

    where ``data_root`` defaults to ``../data`` (relative to the repo
    root, where the prep scripts and training launcher run from). Either
    derived path can be overridden by setting it explicitly in the YAML —
    ``setdefault`` only fills in absent keys.
    """
    ds = cfg.get('data_settings')
    if not ds:
        return cfg
    mstr = ds.get('Mmin_cut_str')
    if mstr is None:
        return cfg
    root = ds.get('data_root', '../data')
    ds.setdefault('halo_hdf5_dir', f'{root}/halos_Mmin{mstr}')
    ds.setdefault('shard_dir',     f'{root}/shards_Mmin{mstr}')
    return cfg


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file, resolving ``extends:`` and deriving data paths.

    The `extends` key, if present, must be a string giving a path to a
    parent YAML, relative to this file's directory. The parent is loaded
    recursively, then this file's values deep-merge over it. After the
    merge, ``data_settings.halo_hdf5_dir`` and ``data_settings.shard_dir``
    are auto-filled from ``Mmin_cut_str`` if not explicitly set.
    """
    cfg = _load_raw(path)
    return _derive_data_paths(cfg)


def _load_raw(path: str) -> dict:
    """Recursive YAML+extends loader, no path derivation."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_rel = cfg.pop('extends', None)
    if parent_rel is None:
        return cfg

    parent_path = os.path.normpath(os.path.join(os.path.dirname(path), parent_rel))
    parent = _load_raw(parent_path)
    return _deep_merge(parent, cfg)
