# CHARM v2 — Current Live Code Summary

## What the Code Does

CHARM trains a deep generative model to paint dark matter halo catalogues onto fast approximate N-body (FastPM) simulations. Given only a cheap FastPM density/velocity field and a 5-parameter cosmology vector, the trained model samples a full halo catalogue complete with:

| Property | Model head | Output shape per voxel |
|---|---|---|
| Voxel occupancy (binary) | SumGauss (Bernoulli surrogate) | scalar |
| Halo count Nhalos (1–Nmax) | SumGauss (categorical surrogate) | scalar |
| Heaviest halo mass M1 | NSF_1var_CNNcond | scalar |
| Mass differences M2–M1, …, MNmax–M(Nmax-1) | NSF_Autoreg_CNNcond | (Nmax−1,) |
| 3D peculiar velocities | NSF_Autoreg_CNNcond | (Nmax×3,) |
| NFW concentrations | NSF_Autoreg_CNNcond | (Nmax,) |
| Sub-voxel 3D position offsets | NSF_Autoreg_CNNcond | (Nmax×3,) |

The model is fully probabilistic: all property heads are normalising flows (neural spline flows) or Gaussian mixture heads, so inference is a forward sample, not a point estimate.

---

## File-by-File Reference

### `charm/` — Python Package

#### `__init__.py`

Flat re-export of the five active modules (all `*` imports from `all_models_v2`, `cnn_3d_stack_v2`, `combined_models_v2`, `utils_data_prep_v2`, `utils`).

---

#### `charm/utils.py`

Pure-PyTorch implementation of the **rational-quadratic spline (RQS)** transform, taken from the [bayesiains/nsf](https://github.com/bayesiains/nsf) repo.

Key functions:

- `unconstrained_RQS(inputs, W, H, D, inverse, tail_bound)` — applies one RQS layer; identity outside the tail bounds.
- `RQS(...)` — inner implementation that solves the spline quadratic and returns `(outputs, log|det J|)`.

These are the bijections used inside every NSF head.

---

#### `charm/config_loader.py`

Shared YAML loader supporting an `extends:` config inheritance chain and mass-cut-namespaced data path derivation. **New since the arxiv version.**

Key behaviour:

- A child config declares `extends: ../BASE_CONFIG.yaml`; the parent is loaded recursively and deep-merged (dicts merged recursively; scalars and lists replaced wholesale by the child).
- After merging, `data_settings.halo_hdf5_dir` and `data_settings.shard_dir` are auto-derived from `data_settings.Mmin_cut_str` (e.g. `'1e14'` → `../data/halos_Mmin1e14/`, `../data/shards_Mmin1e14/`) if not explicitly set.
- Used by all three pipeline scripts: training, halo processing, and shard building.

---

#### `charm/all_models_v2.py`

All distribution model classes. These act as the **property heads** of the CHARM model.

**Helper utilities:**

- `interpolate(x, xp, fp)` — batched 1-D linear interpolation (used for physical HMF base distribution).
- `FCNN` — two-hidden-layer MLP with Tanh activations; used as the conditioner network inside all flows.
- `_normalize_bounds`, `_apply_rqs`, `_get_gauss_params`, `_sample_gaussian_mixture` — shared logic for constructing and sampling flow layers and Gaussian mixtures.

**Distribution heads:**

| Class | Role | `forward` | `inverse` |
|---|---|---|---|
| `BinaryMaskModel` | Bernoulli occupancy | BCE-with-logits loss | sigmoid probabilities |
| `MultiClassMaskModel` | Categorical Nhalos | cross-entropy loss | softmax class probs |
| `SumGaussModel` | Gaussian mixture scalar | negative log-likelihood | sample via multinomial + Gaussian |
| `NSF_1var_CNNcond` | Conditional NSF over one scalar (M1) | log p(x\|cond) | sample x ~ p(·\|cond) |
| `NSF_Autoreg_CNNcond` | Autoregressive NSF over a vector (Mdiff, vel, conc, pos) | sum of per-component log p | autoregressive sample |
| `NSF_M_all_uncond` | Unconditional NSF (scalar) | (logp, log_det, z) | sample ntot draws |
| `M1_reg_model` | Deterministic M1 regression (rarely used) | — | point prediction |

**NSF_1var_CNNcond** supports multiple base distributions: `gauss`, `halfgauss`, `weibull`, `gumbel`, `physical_hmf` (tabulated Halo Mass Function CDF lookup).

**NSF_Autoreg_CNNcond** implements a fully autoregressive NSF: component `jd` is conditioned on `cond_inp` concatenated with all already-sampled components `z[:, :jd]`. Each component has its own base-distribution MLP and `nflows` RQS coupling layers.

---

#### `charm/cnn_3d_stack_v2.py`

The **3D CNN encoder** that maps padded density sub-cubes to per-voxel feature vectors.

**`ResidualBlock`** — pre-activation 3D residual block:

- `Conv3d → Act → Conv3d → (+ skip) → Act`
- Valid padding; total shrinkage = `2*(ksize-1)` voxels per block.
- Channel-mismatch skip handled via bias-free `Linear` (NDHWC trick).

**`FiLMGenerator`** — Feature-wise Linear Modulation for cosmology conditioning:

- Maps `(N, cosmo_dim)` cosmology vector to per-channel `(gamma, beta)`.
- Applies `F = (1 + gamma) * F + beta` after each block.
- Final linear layer is zero-initialised (identity at init).

**`CNN3D_stackout_v2`** — full encoder:

- Configurable block sequence (`layers_types` = list of `'cnn'` or `'res'`).
- After each block, feature maps are pooled to `dim_out³` via `AdaptiveAvgPool3d`, projected to `d_skip` channels via `1×1 Conv3d`, and stored as skip.
- All block skips are concatenated channel-wise and projected to `nout` features via a final `1×1 Conv3d`.
- When `cosmo_dim > 0`, `FiLMGenerator` is applied after each block.
- Output shape: `(nsim * dim_out³, nout)` — one feature vector per voxel.
- `n_cnn_tot` attribute tracks total valid-conv shrinkage; callers pad input by `n_cnn_tot * (ksize-1)` voxels.

---

#### `charm/combined_models_v2.py`

The **unified `CHARM_Model`** that wires encoder → all property heads.

**Architecture:**

```text
Input: padded DM sub-cube (nsim, ninp, D_pad³)
       cosmology vector  (nsim, ncosmo)
                ↓
       CNN3D_stackout_v2  [FiLM cosmo modulation, multi-res skip readout]
                ↓
   cond_out: (N_vox, nout_cnn + ninp [+ ncosmo if concat_cosmo_after_film])
                ↓
   ┌── optional FCNN projector (sep_*_cond) per head
   ↓
   BinaryMaskModel     → occupancy loss / sample
   MultiClassMaskModel → Nhalos loss / sample
   NSF_1var_CNNcond    → M1 loss / sample   [cond: Nhalos, cond_out]
   NSF_Autoreg_CNNcond → Mdiff loss / sample [cond: Nhalos, M1_TRUE, cond_out]
   NSF_Autoreg_CNNcond → vel loss / sample   [cond: M_all halos (TRUE), cond_out]
   NSF_Autoreg_CNNcond → conc loss / sample  [cond: M_all halos (TRUE), cond_out]
   NSF_Autoreg_CNNcond → pos loss / sample   [cond: M_all halos (TRUE), cond_out]
```

**Conditioning chain during training (uses ground-truth values):**

- Mdiff head sees `[Nhalos_true, M1_true, cond_out]`.
- Vel/conc/pos heads see `[M_all_true, cond_out]` (all Nmax true masses).

**Conditioning chain during sampling (uses model output):**

1. Sample `ntot` (Nhalos) from binary + multiclass heads.
2. Sample M1 conditioned on `(ntot_sampled, cond_out)`.
3. Sample Mdiff conditioned on `(ntot_sampled, M1_sampled, cond_out)` → reconstruct all halo masses.
4. Sample vel, conc, pos all conditioned on `(M_all_sampled, cond_out)`.

**Binary head class-imbalance handling (`binary_loss_mode`):** New feature supporting four modes:

- `none` — standard mean NLL over all voxels (original behaviour, fine for small Mmin).
- `subsample` — keep all occupied voxels + an equal-size random draw of empty voxels (1:1 balance). Trained under `π_train = 0.5`; needs Bayesian correction at inference.
- `alpha` — per-voxel inverse-frequency weights, mean-1 normalised. Also trains under `π_train = 0.5`.
- `focal` — focal loss `(1 − p_correct)^γ` down-weighting easy negatives (no prior correction needed).

Modes `subsample` and `alpha` store `binary_train_prior = 0.5` on the model; at inference the trained `pw_occ` is renormalised to the true `π_target` via an odds-ratio correction:

```text
pw_occ_corrected = pw_occ * r / (1 - pw_occ + pw_occ * r)
    where r = (π_target / π_train) * ((1 - π_train) / (1 - π_target))
```

**`_build_halo_mask`** — constructs per-voxel float occupancy masks from `ntot` arrays; `vel_style=True` repeats each halo slot 3× for 3-component properties.

**`forward`** iterates over outer batches (`nbatches`), encodes each sub-cube, then computes per-head NLL losses for the subset specified by `heads_to_train`.

**`sample`** runs the full pipeline end-to-end; each head can optionally use ground-truth values instead of sampled ones (useful for ablations and debugging).

---

#### `charm/utils_data_prep_v2.py`

Data preparation and HDF5 I/O for the training pipeline.

**Key functions:**

| Function | Role |
|---|---|
| `build_halo_masks(N_halos, Nmax)` | Vectorised construction of `mask_M1`, `mask_Mdiff`, `mask_halo` occupancy tensors |
| `sort_halos_by_mass(arr, argsort, Nmax)` | Sort halo arrays by descending mass; handles scalar and vector-per-halo layouts |
| `normalize_masses / velocities / concentrations` | Linear normalisation to `[0,1]` or `[cmin, cmax]` |
| `compute_subvoxel_positions(pos_halos_sorted, nax_h)` | Convert absolute positions to sub-voxel offsets in `[-0.5, 0.5]` |
| `prep_halo_catalog(...)` | Process one batch: reshape, sort, normalise masses/velocities/concentrations/positions, build masks |
| `prep_density_fields(df_d, df_d_nsh)` | Package padded CNN input and non-shifted voxel features |
| `prep_training_data_v2(...)` | Full batched pipeline over all simulations; returns flat `(nsims_total, ...)` arrays |
| `save_to_hdf5(...)` | Write flat data dict to HDF5 with LZF compression; rank-aware chunking |
| `load_from_hdf5(h5_path, rank, world_size, nsims_per_batch)` | Rank-sliced load; reshapes to `(n_outer, nsims_per_batch, ...)` |
| `load_shard(h5_path, nsims_per_batch)` | Load a dedicated per-GPU shard HDF5 (no rank slicing needed) |

HDF5 layout:

```text
/density/dm_cube    (N, ninp, D_pad, D_pad, D_pad)   float16
/density/dm_nsh     (N, nvox, ninp)                  float16
/halos/N_halos      (N, nvox)                        int16
/halos/M_norm       (N, nvox, Nmax)                  float16
/halos/M1_norm      (N, nvox)                        float16
/halos/Mdiff_norm   (N, nvox, Nmax-1)               float16
/halos/mask_*       ...
/halos/v_norm       (N, nvox, Nmax*3)                float16
/halos/c_norm       (N, nvox, Nmax)                  float16
/halos/pos_norm     (N, nvox, Nmax*3)                float16
/halos/cosmo        (N, nvox, 5)                     float32
/metadata/mu_all, sig_all, sim_ids, subvol_ids
```

---

#### `charm/run_charm_joint_ddp.py`

Single **DDP training script** for all heads jointly.

**Major components:**

- **`parse_args`**  — `--config`, `--resume`, `--wandb_run_name`, `--no_wandb`.
- **`load_config`** — delegates to `config_loader.load_config` (supports `extends:` inheritance).
- **`get_lr(step, nepochs, lr_max, lr_min, warmup_frac)`** — cosine decay with linear warmup.
- **`build_model(cfg)`** — constructs `CNN3D_stackout_v2` + all seven heads + `CHARM_Model` from YAML config. Reads `binary_loss_mode`, `binary_focal_gamma`, and all new config keys.
- **`KendallWeighting`** — learnable per-head log-variances `s_i`; total loss = `Σ[exp(-s_i)·L_i + s_i]`. Prevents any head from being zeroed out via `log_var_clamp=3.0`.
- **`build_optimizer`** — three parameter groups: encoder (lower LR at phase transitions), heads (full LR), Kendall sigmas (slow LR). Uses AdamW.
- **`load_data_to_gpu`** / **`load_val_data_to_gpu`** — load per-GPU shard or legacy monolithic HDF5; pin memory for async GPU transfer.
- **`run_validation`** — full pass over val shard on rank 0; returns per-head mean NLL.
- **`save_checkpoint`** / **`save_best_val_checkpoint`** — rank-0 gated save with `dist.barrier()`.
- **`torch.compile` support** — wraps model via `torch.compile` before DDP if `train_settings.torch_compile: true`. Provides ~10–20% throughput gain on A100/H100.
- **`run_func(args)`** — main training loop:

  1. DDP init (NCCL backend).
  2. Build model, optionally `torch.compile`, wrap in `DDP`.
  3. Load per-rank training data to GPU.
  4. Initialise `SumGauss` mu/sig from HDF5 metadata.
  5. Define phases (staggered or monolithic from config).
  6. For each phase: reset optimizer, cosine LR schedule, iterate over epochs.
  7. Each step: `autocast(bfloat16)` forward → Kendall or sum loss → `backward` → grad clip → `optimizer.step`.
  8. Cross-rank loss reduction via `dist.all_reduce`.
  9. Periodic logging (stdout + W&B), checkpointing on best train loss, validation + best-val checkpoint.

---

#### `charm/calibrate_binary_prior.py`

**New.** Fits a tiny analytical regressor mapping cosmological parameters to expected total halo count, for use at inference time to auto-estimate `π_target`.

**Motivation:** When the binary head is trained with `subsample` or `alpha` modes, it learns under a balanced prior (`π_train = 0.5`). At inference, the trained `pw_occ` must be Bayesian-corrected to the true per-voxel occupancy fraction `π_target = N_halos / ns_h³`. This fraction varies by up to ~50× across the Latin hypercube (e.g. `π ≈ 3.3e-4` for one sim vs `π ≈ 1.5e-2` for another), so a single global value is insufficient.

**Implementation:**

- Loads per-sim halo HDF5 files for all training simulations and counts `N_halos`.
- Fits a degree-2 polynomial (21 coefficients, 5 cosmology params + cross-terms) on `(cosmo, log10(N_halos))` via standardised least squares.
- Typically achieves `R² > 0.99` with ~200 training sims.
- Saves coefficients + feature standardisation stats to `binary_prior_calibrator.npz` in the checkpoint directory.

**API:** `predict(cosmo_vec, coeffs, feature_means, feature_stds)` returns `log10(N_halos)` for a single cosmology vector. Used by `run_inference_v2.py` to auto-estimate `π_target` when `--binary_target_prior` is not given.

---

#### `charm/run_inference_v2.py`

**New.** End-to-end inference script: loads a checkpoint, processes a single simulation's density+velocity field, runs `model.sample()`, and saves the reconstructed mock halo catalog to `.npz`.

**Key functions:**

- `load_fastpm(fastpm_dir, isim, grid, z_snap)` — loads CIC density and velocity pickles for one simulation.
- `load_cosmology(lh_cosmo_file, isim)` — reads the 5-parameter cosmology from the LH parameter file.
- `build_cond_tensors(rho, vel, cosmo_vals, nb, nax, n_pad)` — assembles the three model input tensors `(cond_x, cond_x_nsh, cond_cosmo)` using the same padding/striding as the training shard builder.
- `estimate_target_prior_from_cosmology(cosmo_vec, calibrator_path)` — auto-estimates `π_target` using the polynomial calibrator, falling back to a warning if no calibrator file is found.
- `reconstruct_catalog(sample_out, ...)` — converts raw model output tensors to physical halo arrays: positions in Mpc/h (periodic BC), masses in log10 M☉/h, velocities in km/s (applying `v_true = v_FastPM_interp − v_diff_pred`), and concentrations.
- `flat_idx_to_global_voxel(nb, nax)` — precomputes the global 3D voxel index for every flat entry in the `(nsubs × nvox,)` output tensors.
- `build_dm_velocity_interpolators(vel, BoxSize)` — builds `RegularGridInterpolator` objects for each velocity component, used to interpolate `v_FastPM` at the sub-voxel halo positions.

**Output `.npz` keys:** `pos_mock (N,3)`, `lgM_mock (N,)`, `vel_mock (N,3)`, `conc_mock (N,)`, `ntot_vol (ns_h³)`, plus metadata scalars (sim_id, lgMmin, lgMmax, vmin, vmax, cmin, cmax, BoxSize, ns_h, z, cosmo).

**Binary prior correction** is applied automatically if `model.binary_train_prior` is set: the calibrator is loaded from `<checkpoint_dir>/binary_prior_calibrator.npz` and `π_target` is estimated from the simulation's cosmology. The odds-ratio correction `r` is logged to stdout.

---

#### `charm/plot_inference_v2.py`

**New.** Diagnostic plotting script that compares mock and true halo catalogs across six statistical panels.

**Panels produced:**

1. Projected 2D density maps (mock, true, mock/true ratio) — log-scale `128²` histogram over a 20% slab depth.
2. Per-voxel halo count `P(N_tot)` histogram — mock vs true, log-y scale.
3. Halo mass function `dn/dlgM` with mock/true ratio panel.
4. Real-space P(k) — unweighted and mass-weighted, with ratio panels.
5. RSD P(k) — unweighted and mass-weighted (RSD along z-axis), with ratio panels.
6. Velocity PDF (all three components, log-y) and velocity dispersion vs. mass.
7. Concentration–mass (c–M) median relation with error bars, and concentration PDF.

**Power spectrum** uses Pylians (`MAS_library`, `Pk_library`) with TSC mass assignment on a `384³` grid. **RSD** applies the standard plane-parallel approximation `s_∥ = r_∥ + v_∥ (1+z) / H(z)`.

Figures saved as both PDF and PNG to the output directory.

---

### `prep_data/` — Data Pipeline Scripts

#### `process_density_NGP_fastpm.py`

Legacy script that reads FastPM particle data from BigFile format and paints CIC density contrast fields onto a `128³` grid. Saves padded sub-volume pickle files (`density_HR_subvol_*.pk`) and full-box pickle files (`density_HR_full_*.pk`). Uses multiprocessing (one core per simulation). Velocity processing is analogous (`process_velocity_NGP_fastpm.py`).

---

#### `process_halos_quijote_v2.py`

**Per-simulation halo catalogue processor.** Designed to run as a SLURM array (one task per simulation, 0–1999).

Steps per simulation:

1. Load cosmological parameters for that simulation from `latin_hypercube_params.txt`.
2. Set colossus cosmology.
3. Read Rockstar `*_pid.list` catalog; filter to parent halos (`PID == -1`); apply mass cut.
4. Compute concentration: `c_sim = R200c / Rs`; compute Diemer+2019 model concentration `c_func`; store residual `c_diff = c_sim - c_func`.
5. Load FastPM velocity field pickle; interpolate at halo positions to get `vel_pred`; store residual `vel_diff = vel_pred - vel_true`.
6. Paint all properties (`lgM`, `c_sim`, `c_diff`, `v_true`, `v_diff`) onto a `128³` NGP grid in one call to the Cython `NGP_xyz_prop` function. Also stores sub-voxel position offsets in `[-0.5, 0.5]` voxel units.
7. Sort halos by descending mass within each voxel.
8. Write compact per-sim HDF5 (`halos_rockstar_200c_z0.5.h5`) with LZF compression.

---

#### `process_velocity_NGP_fastpm.py`

Similar to the density script but also extracts particle velocities from BigFile and saves velocity CIC fields as pickle files. Part of the older preprocessing workflow.

---

#### `build_training_shards.py`

**Assembles per-GPU HDF5 training shards.** Designed to run as a SLURM array (tasks 0–7 = training shards, task 8 = validation shard).

Steps:

1. Derive padding `n_pad` from config (`(ksize-1)//2 * n_cnn_layers`).
2. For each simulation in this shard's interleaved set:
   - Load full-box density + velocity pickle.
   - Split into `nb³=512` padded and unpadded sub-volumes using zero-copy `as_strided` views.
   - Load per-sim halo HDF5; split halos to sub-volumes via reshape/transpose.
   - Select `nsubvol_per_ji=128` random sub-volumes (deterministic seed per sim).
   - Stack density + velocity channels: `dm_cube (nsubvol, 4, D_pad, D_pad, D_pad)`.
   - Call `prep_halo_catalog` to normalise all halo properties.
   - Write rows into the pre-allocated shard HDF5.

3. Output: `CHARM_train_shard_{r}.h5` (225 sims × 128 subvols = 28,800 rows each) and `CHARM_val_shard.h5` (100 sims × 128 subvols = 12,800 rows).

---

#### `run_process_halos.sh` / `run_build_shards.sh`

SLURM batch scripts:

- `run_process_halos.sh`: array `0–1999`, 1 CPU, 32 GB, 1 hr — runs `process_halos_quijote_v2.py` for each simulation.
- `run_build_shards.sh`: array `0–8`, 8 CPUs, 128 GB, 4 hrs — runs `build_training_shards.py` for each shard.

---

### `run_configs/`

#### `BASE_CONFIG.yaml`

**New.** Master shared defaults for all CHARM joint training runs. Variant configs declare `extends: BASE_CONFIG.yaml` and override only what genuinely differs.

Key sections:

- `sim_settings`: Grid sizes (`ns_d=128`, `ns_h=128`, `nb=8`), kernel size `nf=3`, block types `[res, res]`, input channels `z_all_FP=[0.5,'v_0.5']` (4 channels), `Nmax=4`, 5 cosmological parameters.
- `network_settings`: `nfeature_cnn=32`, FiLM enabled, `hidden_dim_MAF=128`. Per-head spline knots `K`, bounds `B`, number of flows `nflows`, and base distribution types (gumbel for M1 and Mdiff, gauss for vel/pos, gumbel for conc).
- `train_settings`: Standard 6-phase staggered schedule; `binary_loss_mode=none` default; `torch_compile=true`; Kendall weighting active from phase 0; grad clip 1.0; AdamW weight decay 1e-4; encoder frozen to `0.3×` at each new phase.
- `data_settings`: Paths to Rockstar/FastPM data, LH cosmo file. `Mmin_cut`, `Mmin_cut_str`, `n_shards`, `halo_hdf5_dir`, and `shard_dir` are mandatory per-run overrides (or auto-derived from `Mmin_cut_str`).

**Staggered 6-phase training schedule (BASE defaults):**

```text
Phase 0: binary, multi          (800 epochs,  lr=5e-4)
Phase 1: + m1                   (800 epochs,  lr=5e-4)
Phase 2: + mdiff                (1200 epochs, lr=2e-4)
Phase 3: + pos                  (800 epochs,  lr=2e-4)
Phase 4: + vel                  (1200 epochs, lr=2e-4)
Phase 5: + conc                 (1600 epochs, lr=2e-4)
```

---

#### `TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml`

Trial run config: `extends: BASE_CONFIG.yaml`. Mmin = 1×10¹⁴ M☉/h, 200 training sims, smaller network (`nfeature_cnn=16`, `hidden_dim_MAF=64`), 4 GPUs / 4 shards. Phases are 100–600 epochs instead of 800–1600. Used to validate the full pipeline before the production run.

Key overrides:

- `lgMmin=14.0`, `lgMmax=15.7`, `vmin/vmax=±1350`, `cmin/cmax=[1,15]` (raw `c_sim`, not residual).
- `data_settings.Mmin_cut_str='1e14'` → auto-derives `halo_hdf5_dir` and `shard_dir`.
- `data_settings.nMax_h_raw=6` (halos stored per voxel in per-sim HDF5; ≥ Nmax).

---

#### `TRAIN_CHARM_JOINT_trial_Mmin1e14_balanced.yaml`

Same as above but with `binary_loss_mode: subsample` (or `alpha`) to handle the severe class imbalance at Mmin=1e14 (only ~0.26% of voxels are occupied). The binary prior calibrator (`calibrate_binary_prior.py`) must be run after training to enable auto-estimated prior correction at inference.

---

#### `charm/trained_models/` (in run_configs/)

Three pre-trained `.pth` checkpoints from the old separate-model regime: `charm_model_massNtot_bestfit_v2.pth`, `charm_model_vel_bestfit_v2.pth`, `charm_model_conc_bestfit_v2.pth`. These are the frozen weights used for inference with the older separate-model API (now in `charm/arxiv/`).

---

### `notebooks/testing/`

**`ngp_funcs.pyx`** — Cython extension `NGP_xyz_prop` that simultaneously assigns particle positions and a vector of properties to grid voxels in a single pass (NGP mass assignment). Used in `process_halos_quijote_v2.py`. Compiled `.so` files for Python 3.10 and 3.11 are included.

**`dm_field_sanity.ipynb`** / **`halo_sentence_sanity.ipynb`** — sanity-check notebooks for density field loading and halo catalogue visualisation.

**`plot_shard_histograms.py`** — quick diagnostic script to plot the marginal distributions of all properties in a training shard HDF5.

---

### `run_scripts/`

#### `create_bash_script_savehalocats.py`

Legacy script that generates SLURM submission scripts for the old halo catalogue saving workflow (batches of 200 sims using `predict_save_halo_cats.py`). No longer the primary workflow.

---

## Data Flow

```text
FastPM simulations (BigFile format)
         │
         ▼
process_density_NGP_fastpm.py  ──►  density_HR_full_*.pk, velocity_HR_full_*.pk
         │
         │
Rockstar halo catalogs (*.list)
         │
         ▼
process_halos_quijote_v2.py  ──►  halos_rockstar_200c_z0.5.h5 (per sim)
         │
         ▼
build_training_shards.py  ──►  CHARM_train_shard_{r}.h5  (one per GPU)
                               CHARM_val_shard.h5
         │
         ▼
run_charm_joint_ddp.py (torchrun)
    ├── CNN3D_stackout_v2  (shared encoder, FiLM cosmo)
    ├── SumGaussModel      (binary + multiclass heads)
    ├── NSF_1var_CNNcond   (M1 head)
    ├── NSF_Autoreg_CNNcond (Mdiff, vel, conc, pos heads)
    └── KendallWeighting   (multi-task loss balancing)
         │
         ▼
    checkpoint: charm_joint_best_val.pth
         │
         ├── calibrate_binary_prior.py  ──►  binary_prior_calibrator.npz
         │
         ▼
    run_inference_v2.py  ──►  mock_catalog_sim{NNNN}.npz
         │
         ▼
    plot_inference_v2.py  ──►  inference_stats_sim{NNNN}.{pdf,png}
```

---

## Training Architecture Summary

The joint model has one shared CNN encoder and seven property heads. Loss is the sum of per-head negative log-likelihoods (NLLs):

```text
L_total = w_binary · L_bce  +  w_multi · L_ce
        + w_m1 · L_nll_m1  +  w_mdiff · L_nll_mdiff
        + w_vel · L_nll_vel +  w_conc · L_nll_conc
        + w_pos · L_nll_pos
```

With Kendall weighting active, `w_i = exp(-s_i)` where `s_i = log σ_i²` is a learnable log-variance per head.

During each forward pass, the encoder runs once per outer batch; all heads share the resulting `cond_out` feature vector. Conditioning is hierarchical:

- Binary/multi heads see `cond_out` only.
- M1 head sees `[Nhalos, cond_out]`.
- Mdiff head sees `[Nhalos, M1_TRUE, cond_out]` (uses true M1 during training).
- Vel/conc/pos heads see `[M_all_TRUE halos, cond_out]` (all Nmax true masses during training).

This hierarchy encodes the physical prior that the mass function drives the velocity and concentration distributions.

---

## Key Design Decisions

1. **FiLM over concatenation** for cosmology: avoids the encoder treating cosmology as a spatial channel; instead modulates feature maps directly, enabling richer cosmology-dependent representations.

2. **Multi-resolution skip readout**: all encoder blocks contribute to the output via adaptive pooling + projection, giving the heads access to features at every receptive field scale simultaneously.

3. **Staggered training**: adding one head at a time with reduced encoder LR prevents catastrophic forgetting. The encoder progressively adapts to richer targets.

4. **Kendall uncertainty weighting**: handles the mismatch in NLL scales across heads (e.g., binary BCE ≈ 0.3 vs velocity NLL ≈ 5) without manual tuning of loss weights.

5. **Per-GPU HDF5 shards**: each GPU loads its own shard, avoiding inter-rank data-loading contention. SWMR mode allows multiple processes to read simultaneously.

6. **float16 storage**: all training arrays stored as float16 in HDF5; converted to float32 on GPU. Halves I/O bandwidth and shard file sizes.

7. **Config inheritance**: `extends:` chains + auto-derived data paths from `Mmin_cut_str` let each experimental variant override only what genuinely differs, avoiding YAML duplication and config drift.

8. **Binary prior calibration**: for imbalanced datasets (Mmin=1e14, ~0.26% occupancy), subsample/alpha modes train under a balanced prior. A lightweight polynomial regressor maps cosmology → expected halo count, enabling per-simulation Bayesian correction of the occupancy probability at inference with no extra forward pass.

---

## Future Plans / Upgrades

### Occupancy and Total Halo Count Accuracy

#### Problem Statement

The current CHARM_JOINT_v1 model predicts the total number of halos per simulation with 20–50% errors, well above the <5% target. Empirical breakdown for 4 held-out simulations (v1 checkpoint at phase 3, global step 1100):

| Sim | N_mock | N_true | Ratio | occ_mock | occ_true | occ ratio | mean_N\|occ mock | mean_N\|occ true |
|---|---|---|---|---|---|---|---|---|
| 0 | 76 407 | 117 133 | 0.65 | 2.6% | 4.4% | 0.60 | 1.394 | 1.271 |
| 1 | 466 029 | 395 219 | 1.18 | 16.1% | 13.3% | 1.21 | 1.379 | 1.415 |
| 2 | 101 566 | 155 861 | 0.65 | 3.4% | 5.7% | 0.61 | 1.410 | 1.310 |
| 3 | 701 310 | 532 657 | 1.32 | 23.4% | 16.4% | 1.43 | 1.431 | 1.553 |

**Key observation**: the dominant source of error is the **binary occupancy head** (occ ratio 0.60–1.43), not the multiclass N_halos head (mean N|occ ratio 0.92–1.10 after controlling for voxel selection). This means fixing the binary head is the priority.

---

#### Root Cause 1 — Critical Bug: Wrong π_target in Binary Prior Correction

**This is a code bug, fixable without retraining.**

The binary head is trained with `binary_loss_mode: subsample` (1:1 occupied:empty balance, π_train = 0.5). At inference time, the trained `pw_occ` is Bayesian-corrected to the true per-simulation occupancy fraction π_target:

```text
r = (π_target / π_train) × ((1 - π_train) / (1 - π_target))
pw_occ_corrected = r × pw_occ / (1 + (r - 1) × pw_occ)
```

`π_target` should be the **voxel occupancy fraction** = N_occ_voxels / ns_h³ (fraction of voxels that have ≥1 halo). But the calibrator (`calibrate_binary_prior.py`) predicts **total halo count** N_halos, and the inference script computes:

```python
pi = n_pred / ns_cube   # ← WRONG: N_halos / 128³, not N_occ_voxels / 128³
```

These two quantities differ by the **multi-halo factor** = N_halos / N_occ_voxels, which ranges from 1.26 to 1.62 across the LH and is strongly correlated with occupancy (r = 0.98):

| Sim | occ_true | multi-halo factor | π_wrong | r_wrong | r_correct | r ratio |
|---|---|---|---|---|---|---|
| 0 | 0.044 | 1.27 | 0.054 | 0.057 | 0.046 | 1.23× |
| 1 | 0.133 | 1.44 | 0.192 | 0.238 | 0.154 | 1.55× |
| 2 | 0.057 | 1.32 | 0.075 | 0.081 | 0.060 | 1.35× |
| 3 | 0.164 | 1.55 | 0.254 | 0.341 | 0.196 | 1.74× |

For a sim at the high-occupancy extreme (occ_frac ≈ 26%), r_wrong is **2× larger** than r_correct. This makes the mock over-predict occupancy by a factor of ~1.4–1.5 for those cosmologies.

**Fix (one-line change in `calibrate_binary_prior.py`):**

Change line 171 from:
```python
n = int(f['N_halos'][:].sum())            # counts total halos — WRONG
```
to:
```python
n = int((f['N_halos'][:] > 0).sum())      # counts occupied voxels — CORRECT
```
Then rerun `python charm/calibrate_binary_prior.py --config run_configs/TRAIN_CHARM_JOINT_v1.yaml` to produce a new `binary_prior_calibrator.npz`.

**Expected improvement after bug fix (no retraining):**
- High-occupancy sims (sim1, sim3): error drops from +18–32% → ~0–5%
- Low-occupancy sims (sim0, sim2): error remains ~−35% (see Root Cause 2)

---

#### Root Cause 2 — Binary Head Calibration Fails at Low-Occupancy Extremes

Even with the inflated wrong-r correction (which should over-correct), sims 0 and 2 are still under-predicted by 35–40%. This means the binary head's `pw_occ` values for truly-occupied voxels in sparse cosmologies are **systematically too low** — the model is not confidently labelling even the densest voxels as occupied when the overall field is very sparse.

**Contributing factors:**

1. **Bayesian correction calibration assumption**: `binary` is trained in every phase, so by step 1100 (phase 3) the binary head has received 1100 gradient epochs and is likely converged — more training would not help here. The deeper problem is that the Bayesian correction formula `pw_occ_corr = r × pw_occ / (1 + (r-1) × pw_occ)` requires `pw_occ` to be a perfectly calibrated likelihood ratio under π_train=0.5. In practice the FCNN produces scores that are not perfectly calibrated as probabilities, especially for cosmologies at the extreme low-occupancy end of the LH where the model has the least gradient signal relative to high-occupancy sims (fewer occupied voxels per sim → fewer subsample pairs per epoch). Even with the correct π_target (bug fixed), residual miscalibration causes systematic under-prediction for sparse cosmologies.

2. **Raw density input has cosmology-dependent scale**: FiLM modulation at every ResBlock IS architecturally sufficient to learn cosmology-dependent occupancy thresholds. The subtler issue is that the raw input δ = (ρ − ρ̄)/ρ̄ has variance ∝ σ₈², so the first CNN layer sees cosmology-correlated amplitudes. FiLM must learn to re-interpret the same raw feature values differently per cosmology, which requires more gradient steps than the simpler case of cosmology-scale-invariant inputs. A practical improvement with no architectural change: **normalise the density field by its per-simulation standard deviation** (`rho /= rho.std()`) in `build_training_shards.py` and `build_cond_tensors()` before the CNN. This makes the encoder input approximately scale-invariant; the cosmology scale (σ₈) is still fully accessible via FiLM and the raw cosmo vector in `cond_out`, but FiLM no longer needs to correct for raw input amplitude — it can focus on modulating feature representations. This requires rebuilding the shards and retraining.

3. **Competing gradient signals in shared FCNN**: `cond_out` = `[CNN_features, cosmo_vec]`, so the raw cosmology vector IS directly available to the binary head FCNN. In principle it can learn the global occupancy scale directly from cosmology. In practice, the voxel-level density signal (varying across 128³ voxels per sim) generates far more gradient updates per epoch than the simulation-level cosmology signal (1800 unique values across the whole dataset). This imbalance may slow convergence of the cosmology-dependent occupancy threshold, making more training epochs the primary fix rather than any architectural change.

4. **Coupling to voxel-selection bias**: when the binary head under-selects occupied voxels (sims 0, 2), the multiclass head sees a biased set of high-density voxels → `mean_N|occ` is inflated (1.394 and 1.410 vs truth 1.271 and 1.310). This is a secondary artifact of the binary error, not an intrinsic multiclass problem.

---

#### Root Cause 3 — Multiclass Head Contributing 5–10% Secondary Error

Even controlling for binary errors, the multiclass head (N_halos | occupied) has 5–10% error in `mean_N_halos|occ`. This is partly the binary-selection bias above, and partly an intrinsic calibration issue: the multiclass SumGauss head with 8 Gaussian components (mu=[1,…,8], sigma=0.05) requires precise mixing-weight predictions across a wide range of voxel densities and cosmologies. At 300 training epochs, convergence may be incomplete.

---

#### Proposed Solutions (Priority Order)

**Priority 1 — Immediate, no retraining (fix the π_target bug)**

Change `calibrate_binary_prior.py:171` as described above. Rerun the calibrator. Expected: sims 1 and 3 drop from +18–32% to <5% error. Total-count RMSE across many test sims should drop by roughly half.

**Priority 2 — Next training run: switch to focal loss**

`binary_loss_mode: focal` (already implemented) eliminates the π_target correction requirement entirely: the focal model learns a properly calibrated `P(occ | local_features, cosmology)` directly from the training data, down-weighting easy (clearly empty/occupied) voxels and focusing gradient signal on the uncertain boundary cases. This makes the binary head robust to both the π_target estimation error and the low-occupancy under-calibration.

Config change: in `TRAIN_CHARM_JOINT_v1.yaml` (or a new v2 config), replace:
```yaml
binary_loss_mode: subsample
```
with:
```yaml
binary_loss_mode:    focal
binary_focal_gamma:  2.0
```
No calibrator file needed; remove the `binary_prior_calibrator` logic from `run_inference_v2.py` (set `binary_target_prior=None`).

**Priority 3 — Architecture: direct cosmology → binary logit bias**

Add a small direct branch in `combined_models_v2.py` (or in `build_model`) that maps the raw cosmology vector to a scalar bias term added to the binary head logit before softmax. This gives the binary head a first-order "expected occupancy" adjustment before the CNN features are evaluated:

```python
# inside CHARM_Model.__init__:
self.binary_cosmo_bias = nn.Linear(ncosmo, 1, bias=True)
nn.init.zeros_(self.binary_cosmo_bias.weight)  # neutral at init

# inside forward() / sample(), before binary head evaluation:
logit_bias = self.binary_cosmo_bias(cosmo_vec)   # (N_vox, 1)
cond_b_aug  = cond_b  # unchanged; apply bias inside head forward
# OR: modify cond_out before passing to binary head
```

This is a lightweight change (5 parameters for 5 cosmo dims + 1 bias) that provides a direct shortcut from cosmology to occupancy scale, complementing the FiLM path.

**Priority 4 — Post-hoc probability calibration**

If subsample mode is retained, apply isotonic regression or Platt scaling on the validation set to re-calibrate `pw_occ` → calibrated probability per cosmology bin. This is a lightweight post-processing step that can partially correct miscalibration without retraining. However, switching to focal loss (Priority 2) is a cleaner and more principled solution that eliminates this entire problem class.

---

#### Expected Performance After Fixes

| Fix | Sims 0, 2 (low-occ) | Sims 1, 3 (high-occ) |
|---|---|---|
| Current state | −35–40% | +18–32% |
| Fix 1 (π_target bug fix) | −35–40% (unchanged) | ~0–5% |
| Fix 1 + Fix 2 (focal loss) | ~−10–20% | ~0–5% |
| Fix 1 + Fix 2 + Fix 4 (more training) | ~0–5% | ~0–5% |

The π_target bug fix is the single highest-leverage, zero-cost change. Switching to focal loss removes an entire failure mode class. More training epochs address the fundamental underfitting.

---

### Autoregressive Conditioning Bias: Train–Test Mismatch (Exposure Bias)

#### Problem Statement

CHARM's generative model is a hierarchical autoregressive chain. Each head is conditioned on the outputs of all preceding heads in the chain. During **training** (`CHARM_Model.forward()`), every conditioning input is taken from the ground truth ("teacher forcing"). During **sampling/inference** (`CHARM_Model.sample()`), the same conditioning inputs must come from the outputs of the upstream samplers — which are imperfect. This train–test mismatch is the classical **exposure bias** problem and it affects every head downstream of the binary/multi heads.

The full conditioning chain with all affected links is:

```text
cond_out (CNN encoder, always fixed)
    │
    ▼
[binary] → occupancy mask                          ← no upstream samples; NOT affected
    │
    ▼
[multi]  → Nhalos_sampled  (cond: cond_out)        ← conditioned only on cond_out; NOT affected
    │                                                  (but Nhalos_truth used in training, Nhalos_samp at test)
    ▼
[M1]     → M1_sampled      (cond: Nhalos, cond_out)
    │       ⚡ BIAS-1: training uses Nhalos_truth, inference uses Nhalos_samp
    │         (mild — Nhalos ∈ {1,2,3,4}, discrete)
    ▼
[Mdiff]  → Mdiff_sampled   (cond: Nhalos, M1, cond_out)
    │       ⚡ BIAS-2: training uses M1_truth, inference uses M1_sampled
    │         (HIGH — continuous; entire mass sequence depends on this)
    ▼
[vel  ]  → vel_sampled     (cond: M_all, cond_out)
[conc ]  → conc_sampled    (cond: M_all, cond_out)   ⚡ BIAS-3/4/5: training uses M_all_truth
[pos  ]  → pos_sampled     (cond: M_all, cond_out)      inference uses M_all_samp = [M1_samp, Mdiff_samp]
                                                         (HIGH — compound error from both M1 and Mdiff)
```

Every head marked ⚡ learns a conditional distribution that is calibrated against the true upstream value. At inference, it receives a sample from an imperfect upstream model instead. The head has never seen out-of-distribution conditioning inputs during training, so its response can be systematically wrong in ways that do not show up in individual head NLL scores.

The downstream consequences are:

1. **Mass function distortion** — Mdiff is trained on `p(Mdiff | M1_true)` but evaluated on `p(Mdiff | M1_samp)`. If M1 is biased in a mass-dependent way, the resulting halo mass function is distorted in a way that is invisible from per-head NLL metrics.
2. **c–M relation bias** — the conc head learns `p(c | M_all_true)`; at inference it sees `M_all_samp`. Even if the conc head is perfectly trained under teacher forcing, the c–M relation of the mock catalog may be offset relative to the truth because the conditioning mass is a draw, not the true mass.
3. **Velocity–mass scaling bias** — the vel head has the same issue. The mock velocity dispersion vs. mass relation can be biased compared to truth even for a well-trained vel head.
4. **Positional scatter bias** — the pos head receives `M_all_samp`; any systematic mass over/under-prediction shifts the sub-voxel position distribution.
5. **Error compounding** — errors accumulate multiplicatively down the chain. A modest M1 bias leads to a larger Mdiff bias, which leads to even larger vel/conc/pos biases through the `M_all_samp` vector.

#### Affected Code Locations

All exposure bias lives in `CHARM_Model.forward()` in [charm/combined_models_v2.py](charm/combined_models_v2.py). The five specific lines/blocks where truth conditioning is hardcoded are:

**BIAS-1 — Nhalos → M1 (lines 388–389):**

```python
if self.cond_nhalos_on_m1:
    cond_m1 = torch.cat([nhalos_jb, cond_m1], dim=1)   # nhalos_jb = nhalos_truth[jb]
```

**BIAS-2 — M1 → Mdiff (lines 397–400):**

```python
m1_jb = m1_truth[jb].to(device)          # ← always truth during training
if self.cond_m1_on_mdiff:
    cond_md = torch.cat([nhalos_jb, m1_jb, cond_md], dim=1)
```

**BIAS-3/4/5 — M_all → vel / conc / pos (lines 411–412):**

```python
mhalos_jb = mhalos_truth[jb].to(device)  # ← always truth during training
cond_prop = torch.cat([mhalos_jb, cond_out], dim=1)
```

The corresponding sampling code in `CHARM_Model.sample()` already uses sampled values (lines 580–609), so the sample path itself is correct. The problem is exclusively in `forward()`.

#### Proposed Solution: Scheduled Sampling Across All Heads

The cleanest fix is **scheduled sampling** (Bengio et al. 2015), generalised to cover every biased conditioning link in the chain. A scalar probability `p_sched ∈ [0, 1]` is annealed from 0 at the start of Phase 2 toward a target `p_max` (e.g. 0.5–0.8) by the final phase. In each training iteration, a per-voxel Bernoulli draw with probability `p_sched` decides whether a given voxel uses its teacher-forced (truth) conditioning or its student (sampled) conditioning.

**The key constraint for consistency:** because vel/conc/pos are conditioned on `M_all = [M1, Mdiff]`, the same per-voxel Bernoulli mask used to decide "use sampled M1 for Mdiff?" must also govern "use sampled M_all for vel/conc/pos?" — otherwise a voxel could see teacher-forced M1 at the Mdiff stage but sampled M_all at the vel stage, creating an inconsistent training signal.

**Modified forward pass (pseudocode):**

```python
# --- inside CHARM_Model.forward(), per outer batch jb ---

with torch.no_grad():   # upstream sampling never gets gradients

    # BIAS-1 fix: optionally replace Nhalos_truth with a sample
    if p_sched > 0 and 'multi' in trained_so_far:
        nhalos_samp = multiclass_model.inverse(cond_out[mask_occ]).round()
        mix1 = torch.bernoulli(full(n_vox, p_sched)).bool()
        nhalos_cond = where(mix1, nhalos_samp_expanded, nhalos_truth)
    else:
        nhalos_cond = nhalos_truth                 # teacher forcing (phases 0–1)

    # BIAS-2 fix: optionally replace M1_truth with a sample
    if p_sched > 0 and 'm1' in trained_so_far:
        cond_m1_for_samp = cat([nhalos_cond, cond_out], dim=1)
        m1_samp, _ = m1_model.inverse(cond_m1_for_samp[occ_gt0], mask_m1)
        mix2 = torch.bernoulli(full(n_vox, p_sched)).bool()   # independent draw
        m1_cond = where(mix2, m1_samp_full, m1_truth)         # per-voxel mix
    else:
        m1_cond = m1_truth                         # teacher forcing

    # BIAS-3/4/5 fix: build M_all from the same sampled chain
    if p_sched > 0 and 'mdiff' in trained_so_far:
        cond_md_for_samp = cat([nhalos_cond, m1_cond, cond_out], dim=1)
        mdiff_samp, _ = mdiff_model.inverse(cond_md_for_samp[occ_gt1], mask_mdiff)
        mhalos_samp = cat([m1_cond, mdiff_samp_full], dim=1)
        # reuse mix2 mask: same voxels that used sampled M1 use full sampled chain
        mhalos_cond = where(mix2.unsqueeze(-1), mhalos_samp, mhalos_truth)
    else:
        mhalos_cond = mhalos_truth                 # teacher forcing

# Now use the (possibly mixed) conditioning in every downstream head:
# M1 loss:    conditioned on nhalos_cond   (instead of nhalos_truth)
# Mdiff loss: conditioned on nhalos_cond, m1_cond   (instead of nhalos_truth, m1_truth)
# vel/conc/pos losses: conditioned on mhalos_cond   (instead of mhalos_truth)
```

#### Concrete Implementation Steps

The following changes are needed across three files:

**1. `charm/combined_models_v2.py` — `CHARM_Model.forward()` (primary change)**

- Add `p_sched: float = 0.0` and `trained_heads: frozenset = frozenset()` to the `forward()` signature.
- Replace the three hardcoded truth-conditioning blocks (lines 388–389, 397–400, 411–412) with the conditional mixing logic from the pseudocode above.
- Wrap all upstream sampling calls in `torch.no_grad()`.
- Guard each mixing block with `if p_sched > 0 and <upstream_head> in trained_heads` so that phases that have not yet introduced a head never attempt to sample from an untrained model.
- Use a single per-voxel Bernoulli mask (`mix2`) shared between Mdiff and vel/conc/pos conditioning to maintain chain consistency.

**2. `charm/run_charm_joint_ddp.py` — training loop (schedule computation)**

- Compute `p_sched` as a linear ramp (or sigmoid) from 0 at the beginning of the first phase that includes `mdiff` (Phase 2 by default in `BASE_CONFIG.yaml`) to `p_max` by the end of the final phase. A concrete formula:

```python
global_step_in_ss_window = global_step - ss_ramp_start_step
ss_window_length         = ss_ramp_end_step - ss_ramp_start_step
p_sched = min(p_max, p_sched_base * global_step_in_ss_window / ss_window_length)
```

- Pass `p_sched` and the current `trained_heads` set to every `model.forward()` call.
- Log `p_sched` to wandb alongside the per-head losses so the annealing curve is visible.

**3. `run_configs/BASE_CONFIG.yaml` and variant configs — new `train_settings` keys**

Add the following keys to `train_settings` in `BASE_CONFIG.yaml`:

```yaml
train_settings:
  # Scheduled sampling (exposure bias mitigation)
  scheduled_sampling_p_max:         0.0    # 0.0 = disabled; set 0.5–0.8 to enable
  scheduled_sampling_ramp_start_phase: 2   # phase index at which p_sched begins rising
  scheduled_sampling_ramp_end_phase:   5   # phase index at which p_sched reaches p_max
```

Variant configs that want to enable scheduled sampling override only `scheduled_sampling_p_max`. Setting it to `0.0` (the default) leaves the existing teacher-forcing behaviour entirely unchanged, preserving backward compatibility.

#### Severity Ranking and Suggested Rollout

Not all five bias points are equally severe. A pragmatic rollout order:

1. **BIAS-2 first (M1 → Mdiff)** — the primary concern. Implement and validate this alone before touching the rest. It requires only one upstream sample call (`m1_model.inverse`) per batch.
2. **BIAS-3/4/5 second (M_all → vel/conc/pos)** — follows naturally once BIAS-2 is working; reuses the same Bernoulli mask and adds one more upstream sample call (`mdiff_model.inverse`).
3. **BIAS-1 last (Nhalos → M1)** — lowest severity due to the discrete, bounded nature of Nhalos. Can be deferred or omitted if diagnostic metrics show no improvement.

#### Expected Benefits

- All five downstream heads (Mdiff, vel, conc, pos, and to a lesser extent M1) become robust to the imperfections of their upstream samplers, rather than being overfit to the always-correct teacher signal.
- The joint HMF, c–M relation, velocity–mass scaling, and positional statistics should converge toward truth as `p_sched` increases toward `p_max`.
- No architectural changes to the flows; overhead is at most two extra `model.inverse()` calls per batch under `torch.no_grad()`, i.e., less than one additional gradient step of compute per iteration.
- Backward compatible: setting `scheduled_sampling_p_max = 0.0` (default) recovers the existing training loop exactly.

#### Alternative: Two-Stage Offline Re-Training

A simpler but less adaptive approach: (1) train all phases with teacher forcing as currently done; (2) for each training voxel, draw a large ensemble of M1 samples from the trained M1 head and save them to disk alongside the existing HDF5 shards; (3) retrain the Mdiff head (and subsequently vel/conc/pos heads) from scratch using those saved M1 samples as conditioning inputs. This fully eliminates the mismatch at the cost of one extra training pass and storage for the pre-sampled conditioning tensors. It is simpler to implement but less data-efficient and requires two separate model training runs rather than a single end-to-end curriculum.

#### Diagnostic Protocol

Run inference on a held-out set of test simulations (not used in training) both before and after the fix. Compare:

- **(a) Halo mass function** — systematic offset in the HMF at each mass bin indicates compounding Mdiff bias (BIAS-2).
- **(b) c–M relation** — systematic offset in mock concentration vs. mass at fixed redshift indicates BIAS-4.
- **(c) σ_v–M scaling** — velocity dispersion vs. mass offset indicates BIAS-3.
- **(d) Cross-power spectrum** `P_cross(k) / sqrt(P_mock P_true)` at fixed mass threshold — a ratio below 1 at specific k-modes can reveal mass-scale-dependent biases from BIAS-2.
- **(e) Per-head NLL on test set** — if scheduled sampling is working, per-head test NLL should be comparable to or better than the teacher-forced baseline, because the heads have been trained on a distribution closer to their inference-time input distribution.

A reduction in the *systematic* component of the offset (i.e. a consistent shift of the mock statistics toward truth across many test cosmologies, not just reduced scatter) would confirm that scheduled sampling is addressing the exposure bias rather than merely adding noise.
