#!/usr/bin/env python
"""
run_charm_joint_ddp.py
----------------------
Single DDP training runner for the unified CHARM model: jointly trains
halo count, mass (M1 + Mdiff), sub-voxel position, velocity, and
concentration heads sharing one CNN encoder.

Launch (single node, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 charm/run_charm_joint_ddp.py \
             --config run_configs/TRAIN_CHARM_JOINT.yaml

Launch (multi-node, e.g. 2 nodes × 4 GPUs via SLURM):
    srun torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=4 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        charm/run_charm_joint_ddp.py --config run_configs/TRAIN_CHARM_JOINT.yaml

Key improvements over the three separate runners:
  - All data loaded inside run_func() after DDP init (no module-level I/O)
  - LOCAL_RANK from environment — correct for multi-node
  - bfloat16 autocast actually wrapping the forward pass
  - Per-head loss logging to stdout and W&B
  - Staggered training phases with per-phase LR reset and parameter groups
  - Rank-0-gated checkpoint saves + dist.barrier() after save
  - Gradient clipping
  - Optional gradient checkpointing on encoder (saves GPU memory)
"""

import argparse
import collections
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from cnn_3d_stack_v2    import CNN3D_stackout_v2
from combined_models_v2 import CHARM_Model
from all_models_v2      import (
    SumGaussModel, NSF_1var_CNNcond, NSF_Autoreg_CNNcond, FCNN,
)
from utils_data_prep_v2 import load_from_hdf5, load_shard
from config_loader      import load_config

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True, help='Path to YAML config file.')
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint .pth to resume from.')
    p.add_argument('--wandb_run_name', default=None,
                   help='Override W&B run name.')
    p.add_argument('--no_wandb', action='store_true',
                   help='Disable W&B even if enabled in config.')
    return p.parse_args()


# ── LR scheduler ─────────────────────────────────────────────────────────────

def get_lr(step: int, nepochs: int, lr_max: float, lr_min: float,
           warmup_frac: float = 0.07) -> float:
    """
    Cosine decay with linear warmup.

    step        : iteration index within this phase (0-based)
    nepochs     : total iterations in this phase
    lr_max      : peak learning rate (reached at end of warmup)
    lr_min      : floor learning rate
    warmup_frac : fraction of nepochs used for linear warmup
    """
    warmup = max(1, int(nepochs * warmup_frac))
    if step < warmup:
        return lr_max * (step + 1) / warmup
    t = (step - warmup) / max(1, nepochs - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


# ── Model construction ────────────────────────────────────────────────────────

def _n_cnn_tot(layers_types):
    n = 0
    for lt in layers_types:
        n += 1 if lt == 'cnn' else 2
    return n


def build_model(cfg: dict) -> nn.Module:
    """Construct CHARM_Model with shared CNN3D_stackout_v2 encoder."""
    sc  = cfg['sim_settings']
    nc  = cfg['network_settings']
    tc  = cfg.get('train_settings', {})

    # ── infer derived quantities ───────────────────────────────────────────
    nfeature  = nc['nfeature_cnn']
    nout_cnn  = 4 * nfeature
    layers    = sc['layers_types']
    ksize     = sc['nf']
    z_all_FP  = sc.get('z_all_FP', [])
    ninp      = len(z_all_FP) + (2 if any('v' in str(z) for z in z_all_FP) else 0)
    Nmax      = sc['Nmax']
    use_film           = nc.get('use_film', True)
    concat_cosmo_film  = nc.get('concat_cosmo_after_film', False)
    ncosmo             = sc.get('num_cosmo_params', 5)

    # FiLM feeds cosmo into the encoder; concat_cosmo_after_film also appends
    # the raw cosmo vector to cond_out for a direct conditioning pathway.
    num_cond_base = nout_cnn + ninp
    if not use_film or concat_cosmo_film:
        num_cond_base += ncosmo

    # ── encoder ───────────────────────────────────────────────────────────
    encoder = CNN3D_stackout_v2(
        ksize       = ksize,
        nside_in    = sc['ns_d'],
        nside_out   = sc['ns_h'],
        nbatch      = sc['nb'],
        ninp        = ninp,
        nfeature    = nfeature,
        nout        = nout_cnn,
        layers_types= layers,
        act         = nc.get('encoder_act', 'lrelu'),
        padding     = 'valid',
        cosmo_dim   = ncosmo if use_film else 0,
        d_skip      = nout_cnn,
    )

    # Optional gradient checkpointing on encoder (trades compute for memory)
    if nc.get('grad_checkpoint_encoder', False):
        encoder.use_checkpoint = True

    # ── binary / multiclass heads ──────────────────────────────────────────
    # Both heads model an integer-valued target with a fixed-mu, fixed-sigma
    # mixture of Gaussians; the network only learns mixing weights. This is
    # equivalent to a soft classifier and avoids variance collapse (which
    # otherwise drives -log p(x) → -∞ and gradients to blow up).
    #   binary head (target ∈ {0,1})       : mu = [0, 1]
    #   multi  head (target ∈ {1,…,Nmax}) : mu = [1, 2, …, Nmax]
    sigma_binary = float(nc.get('fixed_sigma_binary', 0.05))
    sigma_multi  = float(nc.get('fixed_sigma_multi',  0.05))
    ngauss_binary = nc.get('ngauss_Ntot',   2)
    ngauss_multi  = nc.get('ngauss_Nhalos', Nmax)
    mu_binary  = np.arange(ngauss_binary, dtype=np.float32)              # [0, 1, …]
    sig_binary = np.full(ngauss_binary, sigma_binary, dtype=np.float32)
    mu_multi   = np.arange(1, ngauss_multi + 1, dtype=np.float32)        # [1, 2, …, Nmax]
    sig_multi  = np.full(ngauss_multi, sigma_multi, dtype=np.float32)

    model_binary = SumGaussModel(
        num_cond   = num_cond_base,
        hidden_dim = nc['hidden_dim_MAF'],
        ngauss     = ngauss_binary,
        mu_all     = mu_binary,
        sig_all    = sig_binary,
        base_dist  = nc.get('base_dist_Ntot', 'normal'),
    )
    model_multi = SumGaussModel(
        num_cond   = num_cond_base,
        hidden_dim = nc['hidden_dim_MAF'],
        ngauss     = ngauss_multi,
        mu_all     = mu_multi,
        sig_all    = sig_multi,
        base_dist  = nc.get('base_dist_Ntot', 'normal'),
    )

    # ── M1 head ───────────────────────────────────────────────────────────
    add_N_for_M1   = nc.get('add_Ntot_cond_for_M1', True)
    num_cond_M1    = num_cond_base + (1 if add_N_for_M1 else 0)
    model_M1 = NSF_1var_CNNcond(
        K          = nc['K_M1'],
        B          = nc['B_M1'],
        hidden_dim = nc['hidden_dim_MAF'],
        num_cond   = num_cond_M1,
        nflows     = nc['nflows_M1'],
        base_dist  = nc['base_dist_M1'],
        ngauss     = nc.get('ngauss_M1', 1),
    )

    # ── Mdiff head ────────────────────────────────────────────────────────
    add_NM1_for_Md  = nc.get('add_Ntot_M1_cond_for_Mdiff', True)
    num_cond_Mdiff  = num_cond_base + (2 if add_NM1_for_Md else 0)
    model_Mdiff = NSF_Autoreg_CNNcond(
        dim        = Nmax - 1,
        K          = nc['K_Mdiff'],
        B          = nc['B_Mdiff'],
        hidden_dim = nc['hidden_dim_MAF'],
        num_cond   = num_cond_Mdiff,
        nflows     = nc['nflows_Mdiff'],
        base_dist  = nc['base_dist_Mdiff'],
        mu_pos     = True,
    )

    # ── velocity head ─────────────────────────────────────────────────────
    num_cond_prop = num_cond_base + Nmax   # masses prepended
    model_vel = NSF_Autoreg_CNNcond(
        dim        = Nmax * 3,
        K          = nc['K_vel'],
        B          = nc['B_vel'],
        hidden_dim = nc['hidden_dim_MAF'],
        num_cond   = num_cond_prop,
        nflows     = nc['nflows_vel'],
        base_dist  = nc['base_dist_vel'],
        mu_pos     = False,
    )

    # ── concentration head ─────────────────────────────────────────────────
    model_conc = NSF_Autoreg_CNNcond(
        dim        = Nmax,
        K          = nc['K_conc'],
        B          = nc['B_conc'],
        hidden_dim = nc['hidden_dim_MAF'],
        num_cond   = num_cond_prop,
        nflows     = nc['nflows_conc'],
        base_dist  = nc['base_dist_conc'],
        mu_pos     = True,
    )

    # ── position head (NEW) ───────────────────────────────────────────────
    model_pos = NSF_Autoreg_CNNcond(
        dim        = Nmax * 3,
        K          = nc['K_pos'],
        B          = nc['B_pos'],
        hidden_dim = nc['hidden_dim_MAF'],
        num_cond   = num_cond_prop,
        nflows     = nc['nflows_pos'],
        base_dist  = nc['base_dist_pos'],
        mu_pos     = False,
    )

    # ── combined model ─────────────────────────────────────────────────────
    model = CHARM_Model(
        encoder           = encoder,
        ndim              = Nmax,
        binary_model      = model_binary,
        multiclass_model  = model_multi,
        m1_model          = model_M1,
        mdiff_model       = model_Mdiff,
        vel_model         = model_vel,
        conc_model        = model_conc,
        pos_model         = model_pos,
        cond_nhalos_on_m1        = add_N_for_M1,
        cond_m1_on_mdiff         = add_NM1_for_Md,
        use_film                 = use_film,
        concat_cosmo_after_film  = concat_cosmo_film,
        sep_binary_cond   = True,  num_cond_binary = num_cond_base,
        sep_multi_cond    = True,  num_cond_multi  = num_cond_base,
        sep_m1_cond       = True,  num_cond_m1     = num_cond_M1,
        sep_mdiff_cond    = True,  num_cond_mdiff  = num_cond_Mdiff,
        binary_loss_mode   = tc.get('binary_loss_mode', 'none'),
        binary_focal_gamma = tc.get('binary_focal_gamma', 2.0),
        # Subsample / alpha train under π=0.5 → trained pw_occ needs a
        # Bayesian renormalisation at inference (see CHARM_Model.sample).
        # Auto-set unless the user pinned a value explicitly.
        binary_train_prior = tc.get(
            'binary_train_prior',
            0.5 if tc.get('binary_loss_mode', 'none') in ('subsample', 'alpha')
            else None
        ),
    )
    return model


# ── Kendall multi-task uncertainty weighting ──────────────────────────────────

class KendallWeighting(nn.Module):
    """
    Multi-task uncertainty weighting from Kendall et al. 2018
    (https://arxiv.org/abs/1705.07115).

    For each task i the total loss becomes:

        L_total = Σ_i  [ exp(-s_i) · L_i  +  s_i ]

    where s_i = log(σ_i²) is a learnable log-variance, initialised to 0
    (σ_i = 1 → equal weighting at the start).  The +s_i regulariser
    prevents σ_i → ∞ from trivially zeroing all contributions.

    Intuition:
      - Large σ_i  → model is "uncertain" about task i → downweights it.
      - Small σ_i  → high confidence → upweights it.
      - This automatically balances, e.g., binary BCE (~0.3) vs vel NLL (~5).

    log_var_clamp : clamp |s_i| ≤ this value so no task is fully switched
                    off (default ±3 → σ in [0.22, 4.5]).

    Only heads present in the incoming losses dict are weighted; any others
    are passed through unchanged. This means inactive phases (staggered
    training) do not accidentally update log_vars for heads not yet trained.
    """

    HEAD_ORDER = ['binary', 'multi', 'm1', 'mdiff', 'pos', 'vel', 'conc']

    def __init__(self, head_names: list, log_var_clamp: float = 3.0):
        super().__init__()
        self.head_names    = head_names
        self.log_var_clamp = log_var_clamp
        # ParameterDict keys cannot contain dots
        self.log_vars = nn.ParameterDict({
            n: nn.Parameter(torch.zeros(1)) for n in head_names
        })

    def forward(self, losses: dict):
        """
        Parameters
        ----------
        losses : {head_name: scalar Tensor}

        Returns
        -------
        total  : scalar Tensor (differentiable w.r.t. log_vars)
        sigmas : {head_name: float}  — current σ_i values for logging
        """
        device = next(iter(losses.values())).device
        total  = torch.zeros(1, device=device)
        sigmas = {}
        for name, L in losses.items():
            if name in self.log_vars:
                s = self.log_vars[name].clamp(-self.log_var_clamp,
                                               self.log_var_clamp)
                total  = total + torch.exp(-s) * L + s
                sigmas[name] = math.exp(0.5 * s.item())
            else:
                total  = total + L
                sigmas[name] = 1.0
        return total.squeeze(), sigmas

    @torch.no_grad()
    def sigma_str(self) -> str:
        """Compact σ summary for stdout."""
        parts = []
        for n in self.HEAD_ORDER:
            if n in self.log_vars:
                s = math.exp(0.5 * self.log_vars[n].item())
                parts.append(f'{n}:{s:.2f}')
        return ' '.join(parts)


# ── Optimizer with parameter groups ──────────────────────────────────────────

def build_optimizer(
        model:            nn.Module,
        lr:               float,
        frozen_lr_scale:  float,
        weight_decay:     float = 1e-4,
        kendall_module:   nn.Module = None,
        kendall_lr_scale: float = 0.1,
        phase_idx:        int   = 0,
) -> torch.optim.Optimizer:
    """
    Three parameter groups:
      - encoder          : lr * frozen_lr_scale  (stable, already warm)
      - heads            : lr                    (new + previously active)
      - kendall (opt.)   : lr * kendall_lr_scale (slow-moving uncertainty params)

    Using a lower LR for the encoder at phase transitions prevents it from
    over-fitting to the new head's early high-loss conditioning signal.
    In phase 0 frozen_lr_scale is set to 1.0 (all parameters at full LR).
    """
    raw_model = model.module if isinstance(model, DDP) else model

    encoder_ids = {id(p) for p in raw_model.encoder.parameters()}
    group_enc, group_heads = [], []
    for _, param in raw_model.named_parameters():
        (group_enc if id(param) in encoder_ids else group_heads).append(param)

    enc_lr = lr if phase_idx == 0 else lr * frozen_lr_scale
    param_groups = [
        {'params': group_enc,   'lr': enc_lr, 'name': 'encoder'},
        {'params': group_heads, 'lr': lr,     'name': 'heads'},
    ]
    if kendall_module is not None:
        param_groups.append({
            'params': list(kendall_module.parameters()),
            'lr':     lr * kendall_lr_scale,
            'name':   'kendall',
        })
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ── Data loading ──────────────────────────────────────────────────────────────

def _cpu_to_gpu(cpu_data: dict, device: torch.device,
                pin: bool = True) -> dict:
    """Convert a cpu_data dict (from load_shard / load_from_hdf5) to GPU tensors.

    float16 and float32 arrays are kept as bfloat16 on GPU rather than being
    expanded to float32.  The shard data is already quantised to float16
    precision; there is zero information gain from widening to float32, but it
    doubles GPU memory consumption (~13 GB per rank for the standard shard
    layout).  bfloat16 matches the autocast dtype used during the forward pass
    so no extra cast occurs inside the training loop.
    """
    gpu = {}
    for k, v in cpu_data.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v.copy())
            if t.dtype in (torch.float16, torch.float32):
                t = t.bfloat16()
            elif t.dtype == torch.int16:
                t = t.long()
            gpu[k] = (t.pin_memory().to(device, non_blocking=True)
                      if pin else t.to(device))
        else:
            gpu[k] = v   # int / metadata scalars
    return gpu


def load_data_to_gpu(cfg: dict, rank: int, world_size: int,
                     device: torch.device) -> dict:
    """Load this rank's training data from a per-GPU shard or legacy monolithic HDF5."""
    tc       = cfg['train_settings']
    spb      = tc['nsims_per_batch']
    shard_dir = tc.get('shard_dir') or \
                cfg.get('data_settings', {}).get('shard_dir')

    if shard_dir is not None:
        shard_path = os.path.join(shard_dir, f'CHARM_train_shard_{rank}.h5')
        cpu_data = load_shard(shard_path, nsims_per_batch=spb)
    else:
        # Fallback: legacy monolithic HDF5
        cpu_data = load_from_hdf5(tc['h5_path'], rank, world_size,
                                  nsims_per_batch=spb)

    return _cpu_to_gpu(cpu_data, device, pin=True)


def load_val_data_to_gpu(cfg: dict, device: torch.device):
    """
    Load the validation shard.  Returns None if the shard file does not exist.
    Called on rank 0 only.
    """
    tc        = cfg['train_settings']
    spb       = tc['nsims_per_batch']
    shard_dir = tc.get('shard_dir') or \
                cfg.get('data_settings', {}).get('shard_dir')
    if shard_dir is None:
        return None

    val_path = os.path.join(shard_dir, 'CHARM_val_shard.h5')
    if not os.path.exists(val_path):
        print(f'  [rank0] Val shard not found at {val_path} — '
              'validation disabled.', flush=True)
        return None

    cpu_data = load_shard(val_path, nsims_per_batch=spb)
    print(f'  [rank0] Val shard loaded: {cpu_data["n_total_subvols"]} subvols  '
          f'n_outer={cpu_data["n_outer"]}', flush=True)
    return _cpu_to_gpu(cpu_data, device, pin=False)


# ── Verbose logging helpers ──────────────────────────────────────────────────

class EMATracker:
    """Per-key exponential moving averages — used to smooth noisy per-head losses."""

    def __init__(self, alpha: float = 0.05):
        self.alpha  = alpha
        self.values = {}

    def update(self, key: str, value: float) -> float:
        if key not in self.values:
            self.values[key] = value
        else:
            self.values[key] = (1.0 - self.alpha) * self.values[key] + self.alpha * value
        return self.values[key]

    def get(self, key: str, default: float = float('nan')) -> float:
        return self.values.get(key, default)


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return '--:--:--'
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


@torch.no_grad()
def _param_group_norms(optimizer: torch.optim.Optimizer) -> dict:
    """
    Per-named-group L2 norms of weights and gradients.

    Returns a flat dict of float scalars:
        {'weight_norm/encoder': ..., 'grad_norm/encoder': ..., ...}

    Useful for spotting:
      - encoder weights drifting at a different rate than head weights;
      - one parameter group whose grad collapses to zero (dead branch);
      - exploding norms in any group before they trigger clip_grad_norm.
    """
    out = {}
    for pg in optimizer.param_groups:
        name = pg.get('name', 'group')
        w_sq = 0.0
        g_sq = 0.0
        for p in pg['params']:
            if p is None:
                continue
            w_sq += float(p.detach().pow(2).sum().item())
            if p.grad is not None:
                g_sq += float(p.grad.detach().pow(2).sum().item())
        out[f'weight_norm/{name}'] = math.sqrt(w_sq)
        out[f'grad_norm/{name}']   = math.sqrt(g_sq)
    return out


# ── Profiler ─────────────────────────────────────────────────────────────────

class TrainProfiler:
    """
    Thin wrapper around ``torch.profiler.profile`` driven by the YAML
    ``profile:`` block.

    Behaviour
    ---------
    - Enabled only on rank 0 (profiling all ranks rarely adds insight and
      adds a lot of disk overhead).
    - Uses ``torch.profiler.schedule(wait, warmup, active, repeat)``.  The
      total number of profiled steps is ``(wait+warmup+active) * repeat``.
    - On every ``on_trace_ready`` callback (fires after each `active`
      window) we (a) print a top-K kernel summary by self-CUDA time and
      (b) export both a Chrome trace (open in ``chrome://tracing``) and
      a TensorBoard trace (``tensorboard --logdir=<trace_dir>`` with the
      PyTorch profiler plugin).
    - If ``exit_after`` is true, training exits cleanly once the schedule
      has captured ``(wait+warmup+active) * repeat`` steps — useful for
      one-shot profiling jobs that should not consume a long-running slot.

    The profiler is a no-op when ``enabled=false`` or on non-rank-0.
    """

    def __init__(self, cfg: dict, rank: int):
        pc = cfg.get('profile', {}) or {}
        self.enabled    = bool(pc.get('enabled', False)) and rank == 0
        self.exit_after = bool(pc.get('exit_after', False))
        self.step_count = 0
        self._prof      = None
        if not self.enabled:
            self.total_steps = 0
            return

        self.wait    = int(pc.get('wait',    1))
        self.warmup  = int(pc.get('warmup',  2))
        self.active  = int(pc.get('active',  5))
        self.repeat  = int(pc.get('repeat',  1))
        self.total_steps = (self.wait + self.warmup + self.active) * self.repeat

        record_shapes  = bool(pc.get('record_shapes',  True))
        with_stack     = bool(pc.get('with_stack',     False))
        with_modules   = bool(pc.get('with_modules',   True))
        profile_memory = bool(pc.get('profile_memory', False))
        self.topk      = int(pc.get('topk',            20))
        self.print_summary = bool(pc.get('print_summary', True))

        sort_default = ('self_cuda_time_total' if torch.cuda.is_available()
                        else 'self_cpu_time_total')
        self.sort_by = pc.get('sort_by', sort_default)

        self.trace_dir = pc.get('trace_dir') or os.path.join(
            cfg['train_settings'].get('checkpoint_dir', './'), 'profile')
        os.makedirs(self.trace_dir, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        sched = torch.profiler.schedule(
            wait    = self.wait,
            warmup  = self.warmup,
            active  = self.active,
            repeat  = self.repeat,
        )

        # Compose Chrome + TensorBoard handlers, plus a printed top-K
        # table so the bottleneck is visible in stdout/W&B logs without
        # opening any external tool.
        tb_handler = torch.profiler.tensorboard_trace_handler(self.trace_dir)

        def _on_trace_ready(p):
            tb_handler(p)   # writes <host>_<pid>.pt.trace.json
            try:
                chrome_path = os.path.join(
                    self.trace_dir, f'chrome_trace_step{self.step_count:06d}.json')
                p.export_chrome_trace(chrome_path)
                print(f'[profile] Chrome trace → {chrome_path}', flush=True)
            except Exception as e:
                print(f'[profile] export_chrome_trace failed: {e}', flush=True)

            if self.print_summary:
                try:
                    table = p.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.topk)
                    print(f'\n[profile] Top {self.topk} ops by {self.sort_by}:\n'
                          f'{table}', flush=True)
                except Exception as e:
                    print(f'[profile] table generation failed: {e}', flush=True)

        self._prof = torch.profiler.profile(
            activities      = activities,
            schedule        = sched,
            on_trace_ready  = _on_trace_ready,
            record_shapes   = record_shapes,
            profile_memory  = profile_memory,
            with_stack      = with_stack,
            with_modules    = with_modules,
        )

        print(
            f'[profile] enabled  schedule: wait={self.wait} warmup={self.warmup} '
            f'active={self.active} repeat={self.repeat}  '
            f'(total {self.total_steps} steps captured)\n'
            f'[profile] trace_dir={self.trace_dir}\n'
            f'[profile] view with:  tensorboard --logdir={self.trace_dir}  '
            f'(needs torch-tb-profiler) — or open *.json in chrome://tracing\n'
            f'[profile] exit_after_profile={self.exit_after}',
            flush=True,
        )

    def __enter__(self):
        if self.enabled and self._prof is not None:
            self._prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._prof is not None:
            return self._prof.__exit__(exc_type, exc_val, exc_tb)
        return False

    def step(self):
        if self.enabled and self._prof is not None:
            self._prof.step()
            self.step_count += 1

    def done(self) -> bool:
        """True once the profiler has consumed its full schedule."""
        return self.enabled and self.step_count >= self.total_steps


# ── W&B setup ─────────────────────────────────────────────────────────────────

def setup_wandb(cfg: dict, rank: int, world_size: int, run_name: str = None,
                disabled: bool = False):
    """Initialise W&B on rank 0 only.  Returns wandb run or None."""
    if rank != 0 or not _WANDB_AVAILABLE or disabled:
        return None
    wc = cfg.get('wandb', {})
    if not wc.get('enabled', False):
        return None

    api_key = os.getenv('WANDB_API_KEY', '')
    if api_key:
        wandb.login(key=api_key)

    name = run_name or wc.get('run_name') or \
        f"charm_joint_ws{world_size}_spb{cfg['train_settings']['nsims_per_batch']}"

    # Flatten the full config for W&B
    flat_cfg = {}
    for section, vals in cfg.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat_cfg[f'{section}/{k}'] = v
        else:
            flat_cfg[section] = vals
    flat_cfg['world_size'] = world_size

    run = wandb.init(
        project = wc.get('project', 'CHARM_joint'),
        entity  = wc.get('entity', None),
        name    = name,
        config  = flat_cfg,
        resume  = 'allow',
    )

    # ── custom step metric ─────────────────────────────────────────────
    # All metrics we log are placed on a `global_step` x-axis explicitly,
    # rather than W&B's auto-incrementing internal step counter.  Without
    # this, calls like `wandb.log({...}, step=N)` made out of order with
    # respect to wandb.watch's gradient-histogram hooks (which advance the
    # internal step) get silently rejected as non-monotonic — so the W&B
    # dashboard shows nothing but the auto-collected system metrics
    # (memory, GPU power), even though the training script is calling
    # wandb.log() repeatedly.  Defining a custom step metric makes our
    # logs robust to that.
    wandb.define_metric('global_step')
    for prefix in ('train', 'val', 'perf', 'kendall', 'system', 'profile'):
        wandb.define_metric(f'{prefix}/*', step_metric='global_step')
    return run


def log_metrics(
        wb_run,
        losses:       dict,
        global_step:  int,
        phase_idx:    int,
        lr_encoder:   float,
        lr_heads:     float,
        grad_norm:    float,
        device:       torch.device,
        sigmas:       dict = None,       # Kendall σ_i per head, or None
        total_loss:   float = None,      # pre-computed weighted total, or None → simple sum
        # ── verbose-only extras ───────────────────────────────────────────
        verbose:      bool  = False,
        ema:          'EMATracker' = None,
        group_norms:  dict  = None,    # {'weight_norm/encoder': ..., 'grad_norm/heads': ...}
        step_time:    float = None,    # seconds per step (rolling avg)
        fwd_time:     float = None,
        bwd_time:     float = None,
        opt_time:     float = None,
        steps_per_sec:float = None,
        samples_per_sec: float = None,
        eta_phase:    float = None,    # estimated seconds left in phase
        step_in_phase:int   = None,
        nepochs_phase:int   = None,
        stdout:       bool  = True,    # set False for high-frequency W&B-only pushes
):
    """
    Log to stdout and W&B.

    Per-head losses are always logged to W&B as ``train/loss_<head>``.
    If Kendall weighting is active, σ values are also logged as
    ``kendall/sigma_<head>`` — watching these tells you which heads the
    model is currently confident / uncertain about.

    When ``verbose=True`` the stdout summary becomes a multi-line block
    with timings, throughput, ETA, per-group grad/weight norms and EMA-
    smoothed per-head losses; the W&B payload is augmented with the same
    quantities so a high-frequency dashboard can plot them.
    """
    raw_total = sum(v.item() for v in losses.values())
    display_total = total_loss if total_loss is not None else raw_total

    # Update EMAs (also used in W&B payload below)
    ema_total = None
    if ema is not None:
        for k, v in losses.items():
            ema.update(f'loss_{k}', v.item())
        ema_total = ema.update('loss_total', display_total)

    # ── stdout ────────────────────────────────────────────────────────────
    head_str  = '  '.join(f'{k}={v.item():.4f}' for k, v in sorted(losses.items()))
    sigma_str = ''
    if sigmas:
        sigma_str = '  σ[' + ' '.join(f'{k}={v:.2f}' for k, v in sorted(sigmas.items())) + ']'

    if not verbose and stdout:
        print(
            f'  step={global_step:6d}  ph={phase_idx}  '
            f'loss={display_total:.4f}(raw={raw_total:.4f})  '
            f'{head_str}{sigma_str}  '
            f'|g|={grad_norm:.3f}  lr_enc={lr_encoder:.2e}  lr_h={lr_heads:.2e}',
            flush=True,
        )
    elif verbose and stdout:
        # Multi-line verbose block.
        progress = ''
        if step_in_phase is not None and nepochs_phase:
            pct = 100.0 * (step_in_phase + 1) / nepochs_phase
            progress = f' [{step_in_phase+1}/{nepochs_phase} {pct:5.1f}%]'

        timing_bits = []
        if step_time is not None:
            timing_bits.append(f'step={step_time*1000:.0f}ms')
        if fwd_time is not None:
            timing_bits.append(f'fwd={fwd_time*1000:.0f}ms')
        if bwd_time is not None:
            timing_bits.append(f'bwd={bwd_time*1000:.0f}ms')
        if opt_time is not None:
            timing_bits.append(f'opt={opt_time*1000:.0f}ms')
        if steps_per_sec is not None:
            timing_bits.append(f'{steps_per_sec:.2f} step/s')
        if samples_per_sec is not None:
            timing_bits.append(f'{samples_per_sec/1e3:.1f} kvox/s')
        eta_bit = f'  ETA={_format_eta(eta_phase)}' if eta_phase is not None else ''

        ema_str = ''
        if ema is not None:
            ema_parts = [f'{h}={ema.get("loss_" + h):.3f}' for h in sorted(losses)]
            ema_str = '  EMA[' + ' '.join(ema_parts) + ']'

        gn_str = ''
        if group_norms:
            gn_parts = []
            for key in sorted(group_norms):
                short = key.split('/')[1][:3] if '/' in key else key
                kind  = 'g' if key.startswith('grad_norm') else 'w'
                gn_parts.append(f'{kind}.{short}={group_norms[key]:.2f}')
            gn_str = '  norms[' + ' '.join(gn_parts) + ']'

        gpu_str = ''
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(device) / 1e9
            res   = torch.cuda.memory_reserved(device)  / 1e9
            peak  = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_str = f'  gpu={alloc:.1f}/{res:.1f}/{peak:.1f}GB'

        ema_total_str = f' (ema={ema_total:.4f})' if ema_total is not None else ''
        print(
            f'┌─ step={global_step:6d}  ph={phase_idx}{progress}  '
            f'loss={display_total:.4f}{ema_total_str} (raw={raw_total:.4f})\n'
            f'│  heads: {head_str}{sigma_str}\n'
            f'│  {"  ".join(timing_bits)}{eta_bit}{gpu_str}\n'
            f'│  |g|clip={grad_norm:.3f}  lr_enc={lr_encoder:.2e}  lr_h={lr_heads:.2e}'
            f'{gn_str}{ema_str}\n'
            f'└─',
            flush=True,
        )

    if wb_run is None:
        return

    # ── W&B ───────────────────────────────────────────────────────────────
    log = {
        'global_step':                 global_step,
        'train/total_loss':            display_total,
        'train/total_loss_unweighted': raw_total,
        'train/phase':                 phase_idx,
        'train/grad_norm':             grad_norm,
        'train/lr_encoder':            lr_encoder,
        'train/lr_heads':              lr_heads,
    }
    # Per-head losses — the most informative metrics for debugging convergence
    for k, v in losses.items():
        log[f'train/loss_{k}'] = v.item()

    # Kendall σ_i — shows which tasks the model is uncertain about
    if sigmas:
        for k, v in sigmas.items():
            log[f'kendall/sigma_{k}'] = v

    # GPU memory
    if torch.cuda.is_available():
        log['system/gpu_mem_alloc_gb']  = torch.cuda.memory_allocated(device)  / 1e9
        log['system/gpu_mem_reserv_gb'] = torch.cuda.memory_reserved(device)   / 1e9
        log['system/gpu_mem_peak_gb']   = torch.cuda.max_memory_allocated(device) / 1e9

    # ── verbose-only extras ─────────────────────────────────────────────
    if verbose:
        if ema is not None:
            log['train/loss_total_ema'] = ema_total
            for k in losses:
                log[f'train/loss_{k}_ema'] = ema.get(f'loss_{k}')
        if group_norms:
            for k, v in group_norms.items():
                log[f'train/{k}'] = v
        if step_time is not None:
            log['perf/step_time_s'] = step_time
        if fwd_time is not None:
            log['perf/fwd_time_s'] = fwd_time
        if bwd_time is not None:
            log['perf/bwd_time_s'] = bwd_time
        if opt_time is not None:
            log['perf/opt_time_s'] = opt_time
        if steps_per_sec is not None:
            log['perf/steps_per_sec'] = steps_per_sec
        if samples_per_sec is not None:
            log['perf/samples_per_sec'] = samples_per_sec
        if eta_phase is not None and math.isfinite(eta_phase):
            log['perf/phase_eta_s'] = eta_phase

    wb_run.log(log)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
        model:        nn.Module,
        val_gpu:      dict,
        active_heads: frozenset,
        device:       torch.device,
        use_film:     bool,
) -> dict:
    """
    Run a full pass over the validation shard on rank 0.

    Returns {head_name: mean_loss_over_batches} and also 'total' key.
    """
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()

    n_outer = val_gpu['n_outer']

    def _g(k):
        return val_gpu.get(k)

    # Pre-build the same derived views as in training
    N_halos     = _g('N_halos')
    x_binary    = (N_halos > 0).float().unsqueeze(-1)
    x_multi     = N_halos.unsqueeze(-1)
    x_m1        = _g('M1_norm').unsqueeze(-1) if _g('M1_norm') is not None else None
    x_mdiff     = _g('Mdiff_norm')
    x_vel       = _g('v_norm')
    x_conc      = _g('c_norm')
    x_pos       = _g('pos_norm')
    mask_m1     = _g('mask_M1').unsqueeze(-1)
    mask_mdiff  = _g('mask_Mdiff')
    mask_vel    = _g('mask_vel')
    mask_conc   = _g('mask_conc')
    mask_pos    = _g('mask_pos')
    cond_x      = _g('dm_cube')
    cond_nsh    = _g('dm_nsh')
    cosmo       = _g('cosmo')
    nhalos      = N_halos.float().unsqueeze(-1)
    m1_truth    = x_m1
    mhalos      = _g('M_norm')

    accum = {}
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        losses = raw(
            x_binary       = x_binary,
            x_multi        = x_multi,
            x_m1           = x_m1,
            x_mdiff        = x_mdiff,
            x_vel          = x_vel,
            x_conc         = x_conc,
            x_pos          = x_pos,
            mask_m1        = mask_m1,
            mask_mdiff     = mask_mdiff,
            mask_vel       = mask_vel,
            mask_conc      = mask_conc,
            mask_pos       = mask_pos,
            cond_x         = cond_x,
            cond_x_nsh     = cond_nsh,
            cond_cosmo     = cosmo,
            nhalos_truth   = nhalos,
            m1_truth       = m1_truth,
            mhalos_truth   = mhalos,
            heads_to_train = active_heads,
        )
    for k, v in losses.items():
        accum[k] = v.item()

    accum['total'] = sum(accum.values())
    raw.train()
    return accum


def log_val_metrics(
        wb_run,
        val_losses:  dict,
        global_step: int,
        phase_idx:   int,
        verbose:     bool  = False,
        best_total:  float = None,    # current best val total (for delta in stdout)
) -> None:
    raw_total = val_losses.get('total', sum(v for k, v in val_losses.items()
                                            if k != 'total'))
    head_str = '  '.join(
        f'{k}={v:.4f}' for k, v in sorted(val_losses.items()) if k != 'total'
    )
    if verbose and best_total is not None and math.isfinite(best_total):
        delta = raw_total - best_total
        arrow = '↓' if delta < 0 else '↑'
        delta_str = f'  Δbest={arrow}{abs(delta):.4f}'
    else:
        delta_str = ''
    print(
        f'  [VAL] step={global_step:6d}  ph={phase_idx}  '
        f'total={raw_total:.4f}{delta_str}  {head_str}',
        flush=True,
    )
    if wb_run is None:
        return
    log = {
        'global_step':    global_step,
        'val/total_loss': raw_total,
        'val/phase':      phase_idx,
    }
    for k, v in val_losses.items():
        if k != 'total':
            log[f'val/loss_{k}'] = v
    if best_total is not None and math.isfinite(best_total):
        log['val/total_loss_best'] = min(best_total, raw_total)
    wb_run.log(log)


def save_best_val_checkpoint(
        model:        nn.Module,
        global_step:  int,
        phase_idx:    int,
        val_loss:     float,
        cfg:          dict,
        kendall:      nn.Module = None,
) -> None:
    """Save the best-validation-loss checkpoint (rank 0 only, no barrier needed)."""
    raw = model.module if isinstance(model, DDP) else model
    ckpt_dir = cfg['train_settings']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'charm_joint_best_val.pth')
    torch.save({
        'model_state':   raw.state_dict(),
        'encoder_state': raw.encoder.state_dict(),
        'kendall_state': kendall.state_dict() if kendall is not None else None,
        'global_step':   global_step,
        'phase_idx':     phase_idx,
        'val_loss_min':  val_loss,
        'config':        cfg,
    }, path)
    print(f'  [rank0] Best-val checkpoint saved  val_loss={val_loss:.4f}  '
          f'→ {path}', flush=True)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
        model:         nn.Module,
        optimizer:     torch.optim.Optimizer,
        global_step:   int,
        phase_idx:     int,
        step_in_phase: int,
        loss_min:      float,
        val_loss_min:  float,
        cfg:           dict,
        rank:          int,
        kendall:       nn.Module = None,
        is_milestone:  bool = False,
):
    """Rank-0-only save; all ranks barrier afterwards.

    Always overwrites <ckpt_dir>/charm_joint_resume.pth so that --resume
    (or auto-detection) always has a file to point at, regardless of whether
    the training loss improved.

    When is_milestone=True also writes a step-specific archive file for
    reference (e.g. the best-train-loss snapshot at this phase).
    """
    if rank == 0:
        raw = model.module if isinstance(model, DDP) else model
        ckpt_dir = cfg['train_settings']['checkpoint_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        payload = {
            'model_state':    raw.state_dict(),
            'encoder_state':  raw.encoder.state_dict(),
            'optimizer_state':optimizer.state_dict(),
            'kendall_state':  kendall.state_dict() if kendall is not None else None,
            'global_step':    global_step,
            'phase_idx':      phase_idx,
            'step_in_phase':  step_in_phase,
            'loss_min':       loss_min,
            'val_loss_min':   val_loss_min,
            'config':         cfg,
        }
        # Always overwrite the canonical resume checkpoint.
        resume_path = os.path.join(ckpt_dir, 'charm_joint_resume.pth')
        torch.save(payload, resume_path)
        print(f'  [rank0] Resume checkpoint → {resume_path}  '
              f'(step={global_step} ph={phase_idx} step_in_ph={step_in_phase})',
              flush=True)
        # Milestone archive (best-train-loss snapshot).
        if is_milestone:
            milestone = os.path.join(
                ckpt_dir,
                f'charm_joint_step{global_step:06d}_ph{phase_idx}.pth',
            )
            torch.save(payload, milestone)
            print(f'  [rank0] Milestone checkpoint → {milestone}', flush=True)
    dist.barrier()


def load_checkpoint(
        model:    nn.Module,
        path:     str,
        device:   torch.device,
        kendall:  nn.Module = None,
) -> tuple:
    """Load checkpoint on all ranks.

    Returns
    -------
    global_step     : int   — absolute step counter at save time
    phase_idx       : int   — phase index at save time
    step_in_phase   : int   — steps completed within that phase (0 for old ckpts)
    loss_min        : float — best train loss seen so far
    val_loss_min    : float — best val loss seen so far (inf for old ckpts)
    optimizer_state : dict  — raw optimizer state_dict to restore after building
                              the optimizer for the resumed phase
    """
    ckpt = torch.load(path, map_location=device)
    raw  = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt['model_state'])
    if kendall is not None and ckpt.get('kendall_state') is not None:
        try:
            kendall.load_state_dict(ckpt['kendall_state'])
        except Exception as e:
            print(f'Warning: could not restore Kendall state: {e}')
    step_in_phase  = int(ckpt.get('step_in_phase', 0))
    val_loss_min   = float(ckpt.get('val_loss_min', float('inf')))
    optimizer_state = ckpt.get('optimizer_state', None)
    return (ckpt['global_step'], ckpt['phase_idx'], step_in_phase,
            ckpt['loss_min'], val_loss_min, optimizer_state)


# ── Main training function ────────────────────────────────────────────────────

def run_func(args):
    # ── DDP initialisation ────────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    dist.init_process_group(
        backend  = 'nccl',
        timeout  = timedelta(minutes=60),
    )
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = load_config(args.config)
    tc  = cfg['train_settings']
    staggered = tc.get('staggered', True)

    if rank == 0:
        print(f'World size: {world_size}   Local rank: {local_rank}', flush=True)
        print(f'Staggered training: {staggered}', flush=True)

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    # torch.compile before DDP: Triton-based kernel fusion for ~10–20% gain.
    # DDP wraps the compiled graph without interfering with the compilation.
    # suppress_errors: Dynamo falls back to eager for the autoregressive RQS
    # loops (whose per-step bounds confuse the speculation log) while still
    # compiling the CNN encoder and other static-shape ops.
    if tc.get('torch_compile', False):
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
    # find_unused_parameters=True is REQUIRED for staggered training:
    # in early phases (e.g. phase 0 = binary + multi only) the parameters of
    # the inactive heads (m1, mdiff, vel, conc, pos) never receive gradients,
    # which trips DDP's default all-params-must-have-grads reduction check.
    # It is also needed for the empty-batch guards in CHARM_Model.forward,
    # where a head can be skipped on iterations with no occupied voxels.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Model parameters: {n_params/1e6:.1f}M', flush=True)

    # ── Load training data ────────────────────────────────────────────────
    if rank == 0:
        print('Loading training data ...', flush=True)
    t0 = time.time()
    gpu_data = load_data_to_gpu(cfg, rank, world_size, device)
    n_outer  = gpu_data['n_outer']
    if rank == 0:
        print(f'Train data loaded in {time.time()-t0:.1f}s  '
              f'n_outer_batches_per_rank={n_outer}', flush=True)
        print(f'GPU mem after train data: '
              f'{torch.cuda.memory_allocated(device)/1e9:.2f} GB alloc / '
              f'{torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved', flush=True)

    # ── Load validation data (rank 0 only) ────────────────────────────────
    val_gpu = None
    if rank == 0:
        print('Loading validation data ...', flush=True)
        val_gpu = load_val_data_to_gpu(cfg, device)

    # NOTE: SumGauss mu/sig for the binary and multiclass heads are now fixed
    # at construction time inside build_model() — see the integer-mu / fixed-σ
    # block there. The historical post-init register_buffer code was a no-op
    # (SumGaussModel._free is set in __init__ and never re-read), and it was
    # also wrong for the binary head (mu_all[:2] from HDF5 = [1,2] rather than
    # [0,1] which is what x_binary ∈ {0,1} actually needs).

    # ── Define training phases ────────────────────────────────────────────
    if staggered:
        phases = tc['phases']          # list of {heads, nepochs, learning_rate, warmup_frac}
    else:
        phases = [{
            'heads':        tc.get('heads', ['binary','multi','m1','mdiff','vel','conc','pos']),
            'nepochs':      tc['nepochs'],
            'learning_rate':tc['learning_rate'],
            'warmup_frac':  tc.get('warmup_frac', 0.07),
        }]

    # ── Kendall uncertainty weighting ─────────────────────────────────────
    use_kendall      = tc.get('use_kendall', False)
    kendall_lr_scale = tc.get('kendall_lr_scale', 0.1)
    all_head_names   = ['binary', 'multi', 'm1', 'mdiff', 'pos', 'vel', 'conc']
    kendall = None
    if use_kendall:
        kendall = KendallWeighting(
            head_names    = all_head_names,
            log_var_clamp = tc.get('kendall_log_var_clamp', 3.0),
        ).to(device)
        if rank == 0:
            print(f'Kendall uncertainty weighting ENABLED  '
                  f'(lr_scale={kendall_lr_scale}, '
                  f'clamp=±{tc.get("kendall_log_var_clamp", 3.0)})', flush=True)
    else:
        if rank == 0:
            print('Kendall weighting DISABLED — using simple sum of losses.', flush=True)

    # ── Resume checkpoint ─────────────────────────────────────────────────
    global_step          = 0
    start_phase          = 0
    start_step_in_phase  = 0
    loss_min             = float('inf')
    resume_opt_state     = None   # optimizer state to restore at phase entry
    resume_val_loss_min  = float('inf')

    resume_path = args.resume or tc.get('resume_checkpoint')
    # Auto-detect: if no explicit path given, pick up charm_joint_resume.pth
    # if it already exists in the checkpoint directory (makes re-launching a
    # preempted job as simple as re-submitting without config changes).
    if resume_path is None:
        _auto = os.path.join(tc['checkpoint_dir'], 'charm_joint_resume.pth')
        if os.path.exists(_auto):
            resume_path = _auto
            if rank == 0:
                print(f'Auto-detected resume checkpoint: {_auto}', flush=True)

    if resume_path:
        (global_step, start_phase, start_step_in_phase,
         loss_min, resume_val_loss_min, resume_opt_state) = load_checkpoint(
            model, resume_path, device, kendall=kendall)
        if rank == 0:
            print(f'Resumed from {resume_path}\n'
                  f'  global_step={global_step}  phase={start_phase}  '
                  f'step_in_phase={start_step_in_phase}\n'
                  f'  loss_min={loss_min:.4f}  val_loss_min={resume_val_loss_min:.4f}',
                  flush=True)

    # ── W&B ───────────────────────────────────────────────────────────────
    dist.barrier()
    wb_run = setup_wandb(cfg, rank, world_size,
                         run_name=args.wandb_run_name,
                         disabled=args.no_wandb)
    if wb_run is not None:
        wlf = int(tc.get('watch_log_freq', 200))
        # 0 → disable wandb.watch entirely.  watch() registers backward
        # hooks on every parameter and serialises gradient histograms
        # over the wire — useful occasionally, surprisingly expensive
        # when log_freq is small or when ranks are out of sync.
        if wlf > 0:
            wandb.watch(model, log='gradients', log_freq=wlf)
            print(f'  [rank0] wandb.watch enabled, log_freq={wlf}', flush=True)
        else:
            print('  [rank0] wandb.watch DISABLED (watch_log_freq=0).',
                  flush=True)

    # ── Training loop ─────────────────────────────────────────────────────
    log_every       = tc.get('log_every',  10)
    save_every      = tc.get('save_every', 100)
    val_every       = tc.get('val_every',  save_every)  # validate every N steps
    grad_clip       = tc.get('grad_clip',  1.0)
    min_lr          = tc.get('min_lr',     1e-6)
    frozen_lr_scale = tc.get('frozen_lr_scale', 0.3)
    val_loss_min    = resume_val_loss_min   # restored from checkpoint (inf on fresh start)

    # ── Verbose-logging knobs ─────────────────────────────────────────────
    verbose         = bool(tc.get('verbose', False))
    wandb_log_every = int(tc.get('wandb_log_every', log_every))
    ema_alpha       = float(tc.get('ema_alpha', 0.05))
    ema_tracker     = EMATracker(alpha=ema_alpha) if (verbose and rank == 0) else None
    # Rolling step-time window — used for steady throughput / phase-ETA estimates.
    # Reset at every phase boundary so the ETA isn't biased by the previous phase.
    step_time_window = collections.deque(maxlen=50)
    # Rolling val + checkpoint share so the user can see if those are dominating
    # step time (the dominant cause of "log file isn't updating" in trial configs
    # with val_every=1 / save_every=1).
    last_val_save_time = 0.0
    # ── Heartbeat ─────────────────────────────────────────────────────────
    # When set, rank 0 emits a single short ``[hb] step=N: tag`` line at the
    # entry of every major section of every step.  Lets you SEE where
    # execution actually is when the rich log_every block hasn't fired yet
    # (e.g. when a step is unexpectedly long because of profiler overhead,
    # a slow validation pass, or a network stall in wandb).
    heartbeat = bool(tc.get('heartbeat', False))
    def _hb(tag: str, step: int):
        if heartbeat and rank == 0:
            print(f'[hb] step={step:6d}: {tag}', flush=True)
    if rank == 0:
        print(f'Verbose logging: {verbose}  (wandb_log_every={wandb_log_every}, '
              f'ema_alpha={ema_alpha})  heartbeat={heartbeat}', flush=True)

    # ── Profiler ──────────────────────────────────────────────────────────
    profiler = TrainProfiler(cfg, rank)

    # Precompute static GPU tensor views for target data
    # These are already on GPU from load_data_to_gpu()
    def _g(key, default=None):
        return gpu_data.get(key, default)

    # Binary targets: derived from N_halos
    N_halos_gpu    = _g('N_halos')          # (n_outer, nsims*nvox) long
    x_binary_gpu   = (N_halos_gpu > 0).float().unsqueeze(-1)  # (n_outer, N_vox, 1)
    x_multi_gpu    = N_halos_gpu.unsqueeze(-1)                 # (n_outer, N_vox, 1) long
    x_m1_gpu       = _g('M1_norm').unsqueeze(-1) \
                     if _g('M1_norm') is not None else None    # (n_outer, N_vox, 1)
    x_mdiff_gpu    = _g('Mdiff_norm')       # (n_outer, N_vox, Nmax-1)
    x_vel_gpu      = _g('v_norm')           # (n_outer, N_vox, Nmax*3)
    x_conc_gpu     = _g('c_norm')           # (n_outer, N_vox, Nmax)
    x_pos_gpu      = _g('pos_norm')         # (n_outer, N_vox, Nmax*3)
    mask_m1_gpu    = _g('mask_M1').unsqueeze(-1)   # (n_outer, N_vox, 1)
    mask_mdiff_gpu = _g('mask_Mdiff')
    mask_vel_gpu   = _g('mask_vel')
    mask_conc_gpu  = _g('mask_conc')
    mask_pos_gpu   = _g('mask_pos')
    cond_x_gpu     = _g('dm_cube')   # (n_outer, nsims_per_batch, ninp, D, D, D)
    cond_nsh_gpu   = _g('dm_nsh')    # (n_outer, N_vox, ninp)
    cosmo_gpu      = _g('cosmo')     # (n_outer, N_vox, ncosmo) or None
    nhalos_gpu     = N_halos_gpu.float().unsqueeze(-1)
    m1_truth_gpu   = x_m1_gpu
    mhalos_gpu     = _g('M_norm')    # (n_outer, N_vox, Nmax)

    use_film = cfg['network_settings'].get('use_film', True)

    should_exit_for_profile = False
    with profiler:
     for phase_idx, phase_cfg in enumerate(phases):
        if phase_idx < start_phase:
            # Phases already completed — global_step was restored from the
            # checkpoint and already reflects all prior steps; do NOT add to it.
            continue

        active_heads = frozenset(phase_cfg['heads'])
        lr_max       = phase_cfg['learning_rate']
        nepochs_ph   = phase_cfg['nepochs']
        warmup_frac  = phase_cfg.get('warmup_frac', 0.07)

        # Step offset within this phase: 0 for fresh phases, >0 when resuming
        # mid-phase.  Governs both the inner-loop range and the LR schedule
        # position so the cosine curve continues from exactly where it was.
        phase_start_step = start_step_in_phase if phase_idx == start_phase else 0

        optimizer = build_optimizer(
            model,
            lr               = lr_max,
            frozen_lr_scale  = frozen_lr_scale,
            kendall_module   = kendall,
            kendall_lr_scale = kendall_lr_scale,
            phase_idx        = phase_idx,
        )

        # Restore optimizer state when resuming mid-phase so AdamW first/second
        # moment estimates are not reset to zero.
        if phase_idx == start_phase and resume_opt_state is not None:
            try:
                optimizer.load_state_dict(resume_opt_state)
                if rank == 0:
                    print(f'  Optimizer state restored from checkpoint '
                          f'(phase {phase_idx}, step_in_phase={phase_start_step}).',
                          flush=True)
            except Exception as e:
                if rank == 0:
                    print(f'  Warning: could not restore optimizer state '
                          f'({e}); starting with fresh AdamW moments.',
                          flush=True)
            resume_opt_state = None   # free the dict; only needed once

        if rank == 0:
            print(f'\n{"="*70}')
            print(f' Phase {phase_idx}: heads={sorted(active_heads)}  '
                  f'nepochs={nepochs_ph}  lr_max={lr_max:.2e}')
            print(f'{"="*70}', flush=True)
            if wb_run is not None:
                wb_run.log({'global_step':        global_step,
                            'train/phase':        phase_idx,
                            'train/active_heads': str(sorted(active_heads))})

        # Reset rolling step-time window at each phase boundary so phase ETA
        # is built only from this phase's step times.
        step_time_window.clear()

        # Per-step throughput unit: total voxels processed in one model() call
        # on this rank.  ×world_size gives the global figure.
        # N_halos_gpu has shape (n_outer, N_vox); each step iterates the full
        # outer dimension inside CHARM_Model.forward, so all rows are touched.
        nvox_per_step_rank = int(N_halos_gpu.numel())

        step_in_phase = phase_start_step
        for _ in range(phase_start_step, nepochs_ph):
            _hb('entering step (lr update)', global_step)
            step_t0 = time.time() if verbose else None
            lr = get_lr(step_in_phase, nepochs_ph, lr_max, min_lr, warmup_frac)
            for pg in optimizer.param_groups:
                if pg['name'] == 'encoder':
                    pg['lr'] = lr if phase_idx == 0 else lr * frozen_lr_scale
                elif pg['name'] == 'kendall':
                    pg['lr'] = lr * kendall_lr_scale
                else:
                    pg['lr'] = lr

            # ── forward + backward (gradient accumulation over outer batches) ──
            # Root cause of OOM: the previous code ran all n_outer sub-volume
            # batches inside one model.forward() call, accumulating every batch's
            # activation graph before a single .backward(). With n_outer ~ 500+
            # and ~150 MB of activations per batch this saturated an 80 GB H100
            # regardless of nsims_per_batch.
            # Fix: one forward+backward per outer batch; activations are freed
            # immediately after each .backward(), so peak memory is O(1 batch).
            _hb('starting forward+backward', global_step)
            if verbose and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            fwd_t0 = time.time() if verbose else None

            kendall_active = (
                use_kendall
                and kendall is not None
                and phase_idx >= tc.get('kendall_start_phase', 0)
            )

            def _s(t, jb):
                return t[jb:jb + 1] if t is not None else None

            loss_sum_log = {h: 0.0 for h in active_heads}
            loss_scalar  = torch.zeros((), device=device, dtype=torch.float32)
            sigmas = None

            for jb in range(n_outer):
                # Suppress DDP all-reduce on every step except the last so
                # gradients accumulate locally; one all-reduce fires at the end.
                ddp_ctx = model.no_sync() if jb < n_outer - 1 else nullcontext()
                with ddp_ctx, torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    losses_jb = model(
                        x_binary    = _s(x_binary_gpu,   jb),
                        x_multi     = _s(x_multi_gpu,    jb),
                        x_m1        = _s(x_m1_gpu,       jb),
                        x_mdiff     = _s(x_mdiff_gpu,    jb),
                        x_vel       = _s(x_vel_gpu,      jb),
                        x_conc      = _s(x_conc_gpu,     jb),
                        x_pos       = _s(x_pos_gpu,      jb),
                        mask_m1     = _s(mask_m1_gpu,    jb),
                        mask_mdiff  = _s(mask_mdiff_gpu, jb),
                        mask_vel    = _s(mask_vel_gpu,   jb),
                        mask_conc   = _s(mask_conc_gpu,  jb),
                        mask_pos    = _s(mask_pos_gpu,   jb),
                        cond_x      = cond_x_gpu[jb:jb + 1],
                        cond_x_nsh  = cond_nsh_gpu[jb:jb + 1],
                        cond_cosmo  = _s(cosmo_gpu,      jb),
                        nhalos_truth= nhalos_gpu[jb:jb + 1],
                        m1_truth    = _s(m1_truth_gpu,   jb),
                        mhalos_truth= _s(mhalos_gpu,     jb),
                        heads_to_train = active_heads,
                    )
                if kendall_active:
                    loss_jb, sigmas = kendall(losses_jb)
                else:
                    loss_jb = sum(losses_jb.values())
                (loss_jb / n_outer).backward()
                loss_scalar = loss_scalar + loss_jb.detach() / n_outer
                for h, v in losses_jb.items():
                    loss_sum_log[h] += v.detach().item()

            # Reconstruct per-head mean losses (detached scalars) for logging
            losses = {h: torch.tensor(loss_sum_log[h] / n_outer,
                                      device=device, dtype=torch.float32)
                      for h in active_heads}
            loss = loss_scalar

            if verbose and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            fwd_time = (time.time() - fwd_t0) if verbose else None
            bwd_time = None  # fwd and bwd are interleaved in the accumulation loop
            _hb('forward+backward done', global_step)

            # ── grad clip ─────────────────────────────────────────────────
            all_params = list(model.parameters())
            if kendall_active:
                all_params += list(kendall.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params, max_norm=grad_clip
            ).item()
            if verbose and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            _hb('grad-clip done', global_step)

            # Decide which logging cadences fire this step.
            do_stdout    = (rank == 0 and global_step % log_every == 0)
            do_wandb_hf  = (rank == 0 and verbose and wb_run is not None
                            and global_step % wandb_log_every == 0
                            and not do_stdout)

            # Per-group weight + grad norms — captured BEFORE zero_grad so
            # gradient norms reflect the post-clip values actually applied.
            group_norms = None
            if rank == 0 and (do_stdout or do_wandb_hf):
                group_norms = _param_group_norms(optimizer)

            _hb('starting optimizer.step', global_step)
            opt_t0 = time.time() if verbose else None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if verbose and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            opt_time = (time.time() - opt_t0) if verbose else None
            _hb('optimizer.step done', global_step)

            # ── cross-rank loss for logging / checkpointing ───────────────
            _hb('starting all_reduce(loss)', global_step)
            loss_t = loss.detach().clone()
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            global_loss = loss_t.item()
            _hb('all_reduce done', global_step)

            # ── step-time bookkeeping (rank 0, verbose only) ──────────────
            avg_step_time = None
            steps_per_sec = None
            samples_per_sec = None
            eta_phase = None
            if verbose and rank == 0 and step_t0 is not None:
                step_time = time.time() - step_t0
                step_time_window.append(step_time)
                avg_step_time   = sum(step_time_window) / len(step_time_window)
                steps_per_sec   = (1.0 / avg_step_time) if avg_step_time > 0 else 0.0
                samples_per_sec = nvox_per_step_rank * world_size * steps_per_sec
                remaining_steps = max(0, nepochs_ph - step_in_phase - 1)
                eta_phase       = remaining_steps * avg_step_time

            # ── logging ───────────────────────────────────────────────────
            if do_stdout:
                lr_enc = optimizer.param_groups[0]['lr']
                lr_h   = optimizer.param_groups[1]['lr']
                log_metrics(
                    wb_run, losses, global_step,
                    phase_idx, lr_enc, lr_h, grad_norm, device,
                    sigmas=sigmas, total_loss=global_loss,
                    verbose=verbose, ema=ema_tracker,
                    group_norms=group_norms,
                    step_time=avg_step_time,
                    fwd_time=fwd_time, bwd_time=bwd_time, opt_time=opt_time,
                    steps_per_sec=steps_per_sec,
                    samples_per_sec=samples_per_sec,
                    eta_phase=eta_phase,
                    step_in_phase=step_in_phase,
                    nepochs_phase=nepochs_ph,
                    stdout=True,
                )
                # Also print Kendall σ table when active (verbose-only — the
                # multi-line block already includes σ inline, but the explicit
                # table is easier to scan when staring at a long log).
                if kendall_active and verbose:
                    print(f'    σ-table: {kendall.sigma_str()}', flush=True)
            elif do_wandb_hf:
                # High-frequency W&B push — skip stdout entirely.
                lr_enc = optimizer.param_groups[0]['lr']
                lr_h   = optimizer.param_groups[1]['lr']
                log_metrics(
                    wb_run, losses, global_step,
                    phase_idx, lr_enc, lr_h, grad_norm, device,
                    sigmas=sigmas, total_loss=global_loss,
                    verbose=verbose, ema=ema_tracker,
                    group_norms=group_norms,
                    step_time=avg_step_time,
                    fwd_time=fwd_time, bwd_time=bwd_time, opt_time=opt_time,
                    steps_per_sec=steps_per_sec,
                    samples_per_sec=samples_per_sec,
                    eta_phase=eta_phase,
                    step_in_phase=step_in_phase,
                    nepochs_phase=nepochs_ph,
                    stdout=False,
                )

            # ── checkpoint + validation block (timed for bottleneck hints) ─
            _hb('entering checkpoint/val block', global_step)
            vs_t0 = time.time()

            # ── periodic checkpointing ────────────────────────────────────
            # Always save to charm_joint_resume.pth so there is always a
            # checkpoint to resume from, even if the loss never improves.
            # Additionally write a step-specific milestone file when the
            # train loss reaches a new minimum.
            if global_step % save_every == 0 or step_in_phase == nepochs_ph - 1:
                is_new_best = global_loss < loss_min
                if is_new_best:
                    loss_min = global_loss
                _hb('saving checkpoint', global_step)
                save_checkpoint(model, optimizer, global_step,
                                phase_idx, step_in_phase,
                                loss_min, val_loss_min,
                                cfg, rank, kendall=kendall,
                                is_milestone=is_new_best)

            # ── validation (rank 0 only, every val_every steps) ───────────
            if rank == 0 and val_gpu is not None and global_step % val_every == 0:
                _hb('starting validation pass', global_step)
                val_losses = run_validation(
                    model, val_gpu, active_heads, device, use_film,
                )
                log_val_metrics(
                    wb_run, val_losses, global_step, phase_idx,
                    verbose=verbose, best_total=val_loss_min,
                )
                if val_losses['total'] < val_loss_min:
                    val_loss_min = val_losses['total']
                    save_best_val_checkpoint(
                        model, global_step, phase_idx,
                        val_loss_min, cfg, kendall=kendall,
                    )
                _hb('validation pass done', global_step)
            # All ranks must stay in sync — barrier after rank-0 validation.
            # NOTE: val_gpu is only populated on rank 0, so gating on it would
            # leave non-zero ranks skipping the barrier and rank 0 hanging
            # forever. global_step/val_every are identical across ranks, so
            # this predicate alone is a valid collective gate.
            if dist.is_initialized() and global_step % val_every == 0:
                _hb('entering post-val dist.barrier', global_step)
                dist.barrier()
                _hb('post-val barrier released', global_step)

            val_save_time = time.time() - vs_t0
            # Smoothed val+save time so a single slow checkpoint doesn't
            # trigger spurious hints.
            if last_val_save_time == 0.0:
                last_val_save_time = val_save_time
            else:
                last_val_save_time = 0.9 * last_val_save_time + 0.1 * val_save_time

            # If val+save is dominating step time, point the user at the
            # most likely culprit — often val_every=1 / save_every=1.
            if (verbose and rank == 0 and avg_step_time is not None
                    and val_save_time > 0.5 * (avg_step_time + val_save_time)
                    and val_save_time > 0.5):
                share = 100.0 * val_save_time / (avg_step_time + val_save_time)
                print(
                    f'  [hint] val+save took {val_save_time:.2f}s '
                    f'({share:.0f}% of total step). '
                    f'Raise val_every (now {val_every}) / save_every '
                    f'(now {save_every}) if logs feel stuck.',
                    flush=True,
                )

            # Push the val/save share to W&B so it shows up as a separate
            # perf chart — useful for confirming the bottleneck.
            # Wrapped in try/except: if the wandb daemon is stuck (e.g.
            # network issue or full local queue), the training loop must
            # not freeze along with it.
            if rank == 0 and verbose and wb_run is not None:
                try:
                    wb_run.log({
                        'global_step':              global_step,
                        'perf/val_save_time_s':     val_save_time,
                        'perf/val_save_time_s_ema': last_val_save_time,
                    })
                except Exception as e:
                    print(f'  [warn] wandb.log(perf) failed: {e}', flush=True)

            # Advance the profiler schedule (no-op when profile.enabled=false).
            _hb('profiler.step()', global_step)
            profiler.step()
            _hb('end of step', global_step)

            global_step    += 1
            step_in_phase  += 1

            # Clean exit once the configured profile schedule is done, so
            # one-shot profiling jobs don't squat on a long-running slot.
            if profiler.done() and profiler.exit_after:
                if rank == 0:
                    print(
                        f'\n[profile] All scheduled profiling steps captured '
                        f'({profiler.step_count}/{profiler.total_steps}). '
                        f'Exiting because profile.exit_after=true.',
                        flush=True,
                    )
                should_exit_for_profile = True
                break

        if should_exit_for_profile:
            break

    # ── Finalise ──────────────────────────────────────────────────────────
    if rank == 0:
        print('\nTraining complete.', flush=True)
        print(f'Best train loss: {loss_min:.4f}', flush=True)
        if val_gpu is not None:
            print(f'Best val   loss: {val_loss_min:.4f}', flush=True)
        if wb_run is not None:
            wb_run.summary['best_train_loss'] = loss_min
            wb_run.summary['best_val_loss']   = val_loss_min
            wandb.finish()

    dist.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_func(args)


if __name__ == '__main__':
    main()
