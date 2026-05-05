# CHARM v2

**Creating Halos with Auto-Regressive Multi-stage networks** — a deep generative model that paints full dark matter halo catalogues (count, mass, velocity, concentration, position) onto fast approximate N-body simulations conditioned on cosmological parameters.

Given a FastPM density and velocity field plus a 5-parameter cosmology `(Ωm, Ωb, h, ns, σ8)`, CHARM samples a probabilistic halo catalogue over a `128³` grid at redshift z = 0.5.

---

## What CHARM Predicts

For each voxel in a `16³` sub-cube of a 1 Gpc/h box, the model jointly samples:

| Property | Description |
| --- | --- |
| **Nhalos** | Number of halos in the voxel (0 – Nmax) |
| **M1** | Log mass of the most massive halo [log₁₀ M☉/h] |
| **Mdiff** | Log mass differences M₁−M₂, …, M_(Nmax-1)−M_Nmax |
| **v** | 3D peculiar velocity residual for each halo [km/s] |
| **c** | NFW concentration (c_sim) per halo |
| **pos** | Sub-voxel 3D position offset in [−0.5, 0.5] voxel units |

All properties are learned as **normalising flows** (Neural Spline Flows), making the model fully probabilistic — each forward pass draws an independent sample from the learned posterior.

---

## Architecture

```text
FastPM sub-cube  (4, D_pad³)   +  Cosmology (5,)
         │                              │
         ▼                              │ FiLM modulation
   CNN3D_stackout_v2 ←─────────────────┘
   (3D CNN with multi-resolution skip
    readout; blocks = [res, res])
         │
   cond_out  (N_vox, nout_cnn + ninp)
         │
   ┌─────┼──────────────────────────────┐
   ▼     ▼        ▼         ▼           ▼
Binary  Multi    NSF M1   NSF Mdiff   NSF vel / conc / pos
(SumG) (SumG)  (1var)   (autoreg.)   (autoreg., cond on masses)
```

**Conditioning chain during sampling:**

1. Sample `Nhalos` from binary (occupied?) + multiclass (how many?) heads.
2. Sample M1 conditioned on `(Nhalos, cond_out)`.
3. Sample Mdiff conditioned on `(Nhalos, M1, cond_out)` → full mass function.
4. Sample vel, conc, pos each conditioned on `(all halo masses, cond_out)`.

The 3D CNN encoder runs **once** per sub-cube; all seven heads share its output.

---

## Repository Layout

```text
CHARM/
├── charm/                              # Python package
│   ├── __init__.py
│   ├── all_models_v2.py               # All distribution heads (NSF, SumGauss, etc.)
│   ├── cnn_3d_stack_v2.py             # 3D CNN encoder with FiLM + skip readout
│   ├── combined_models_v2.py          # CHARM_Model: encoder + all heads
│   ├── config_loader.py               # YAML config loader with inheritance (extends:)
│   ├── utils_data_prep_v2.py          # Data normalisation and HDF5 I/O
│   ├── utils.py                       # RQS spline implementation
│   ├── run_charm_joint_ddp.py         # DDP training entry point
│   ├── run_inference_v2.py            # Full catalogue inference script
│   ├── calibrate_binary_prior.py      # Post-hoc binary prior calibration (v1/subsample only)
│   ├── diagnose_nmax.py               # Diagnostic: check Nmax coverage in shards
│   ├── plot_inference_v2.py           # Inference validation plots
│   ├── trained_configs/               # Legacy YAML configs for pre-trained models
│   └── trained_models/               # Pre-trained model weights (.pth)
│
├── prep_data/                         # Data pipeline scripts
│   ├── process_halos_quijote_v2.py   # Halo catalogue → per-sim HDF5
│   ├── build_training_shards.py      # Per-sim HDF5 → per-GPU training shards
│   ├── ngp_funcs.pyx                 # Cython NGP halo property assignment
│   ├── setup_ngp.py                  # Build script for ngp_funcs
│   ├── run_process_halos.sh          # SLURM array script for halo processing
│   ├── run_build_shards.sh           # SLURM array script for shard building (Mmin=5e12)
│   ├── run_build_shards_trial_Mmin1e14.sh  # Shard building for Mmin=1e14 trial
│   ├── run_process_halos_trial_Mmin1e14.sh # Halo processing for Mmin=1e14 trial
│   └── sbatch_workers/               # Helper SLURM worker scripts
│
├── run_configs/
│   ├── BASE_CONFIG.yaml              # Shared defaults; all variants inherit from this
│   ├── TRAIN_CHARM_JOINT_v1.yaml     # Full run v1: Mmin=5e12, subsample binary loss
│   ├── TRAIN_CHARM_JOINT_v2.yaml     # Full run v2: Mmin=5e12, focal binary loss
│   ├── TRAIN_CHARM_JOINT_v0.yaml     # Full run v0 (reference)
│   ├── TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml          # Trial run at Mmin=1e14
│   └── TRAIN_CHARM_JOINT_trial_Mmin1e14_balanced.yaml # Trial run with balanced loss
│
├── run_scripts/
│   ├── run_train_full_v1.sh          # SLURM: 3-node H100 training (v1, subsample)
│   ├── run_train_full_v2.sh          # SLURM: 3-node H100 training (v2, focal)
│   ├── run_train_full_v0.sh          # SLURM: 3-node H100 training (v0)
│   ├── run_train_trial_Mmin1e14.sh   # SLURM: single-node trial at Mmin=1e14
│   └── run_train_trial_Mmin1e14_balanced.sh  # SLURM: trial with balanced loss
│
├── notebooks/
│   ├── testing/                      # Sanity check notebooks + NGP Cython build
│   │   ├── ngp_funcs.pyx             # Cython NGP mass assignment
│   │   ├── setup_ngp.py              # Build script for ngp_funcs
│   │   └── *.ipynb                   # Various sanity check notebooks
│   └── plot_shard_histograms.py      # Shard data distribution diagnostics
│
├── requirements.txt
├── setup.py
├── setup.cfg
└── CURRENT_LIVE_CODE_SUMMARY.md      # Detailed technical summary of all live code
```

---

## Installation

### Requirements

```text
Python >= 3.10
PyTorch >= 2.0 (with CUDA for GPU training)
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

Build the Cython NGP extension (required for halo data preprocessing):

```bash
cd prep_data
python setup_ngp.py build_ext --inplace
cd ..
```

### External Data Dependencies

CHARM is trained on:

- **FastPM Latin Hypercube (LH) simulations**: 2000 simulations at 1 Gpc/h, 128³ grid. Density and velocity fields at z = 0.5 in BigFile format.
- **Rockstar halo catalogues**: `out_3_pid.list` files from the Quijote Latin Hypercube suite (HR), Rockstar M200c masses at z = 0.5.
- **Cosmology parameter file**: `latin_hypercube_params.txt` (2000 × 5 array of `[Ωm, Ωb, h, ns, σ8]`).

Update the paths in `run_configs/BASE_CONFIG.yaml` under `data_settings` to point to your data location.

---

## Data Preparation

### Step 1 — Process Halo Catalogues

Run `process_halos_quijote_v2.py` for all 2000 simulations. Each task reads one Rockstar catalogue, computes concentrations and velocity residuals, and writes a per-sim HDF5 file.

**Submit as SLURM array (recommended):**

```bash
sbatch prep_data/run_process_halos.sh
```

This runs 2000 array tasks (array `0–1999`). Each task takes ~10–20 minutes with 4 CPUs and 32 GB RAM.

**Single simulation (for testing):**

```bash
python prep_data/process_halos_quijote_v2.py \
    --config run_configs/TRAIN_CHARM_JOINT_v1.yaml \
    --isim 0
```

Output: `../data/halos_Mmin5e12/{isim}/halos_rockstar_200c_z0.5.h5`

---

### Step 2 — Build Training Shards

After all per-sim HDF5 files are ready, assemble per-GPU training shards. Each shard combines multiple simulations × randomly selected sub-volumes into a flat HDF5 file for fast GPU I/O.

**Submit as SLURM array:**

```bash
sbatch prep_data/run_build_shards.sh
```

This runs 13 array tasks (0–11 = 12 training shards, 12 = validation shard).

**Single shard (for testing):**

```bash
python prep_data/build_training_shards.py \
    --config run_configs/TRAIN_CHARM_JOINT_v1.yaml \
    --shard_rank 0
```

**Validation shard:**

```bash
python prep_data/build_training_shards.py \
    --config run_configs/TRAIN_CHARM_JOINT_v1.yaml \
    --split val
```

Output:

- `../data/shards_Mmin5e12/CHARM_train_shard_{0..11}.h5` — training shards
- `../data/shards_Mmin5e12/CHARM_val_shard.h5` — validation shard

---

## Training

### Config Inheritance

All run configs inherit from `run_configs/BASE_CONFIG.yaml` via the `extends:` key. Child configs override only the keys that differ from the base (Mmin-dependent ranges, dataset size, paths, binary loss mode, W&B run name). Dict keys deep-merge; lists replace wholesale.

```yaml
# Example: TRAIN_CHARM_JOINT_v2.yaml
extends: BASE_CONFIG.yaml
train_settings:
  binary_loss_mode:  focal
  binary_focal_gamma: 2.0
  checkpoint_dir: ../model_checkpoints/CHARM_JOINT_v2/
```

### Single Node (4 GPUs)

```bash
torchrun --standalone --nproc_per_node=4 \
    charm/run_charm_joint_ddp.py \
    --config run_configs/TRAIN_CHARM_JOINT_v2.yaml
```

### Multi-Node via SLURM

```bash
# v2 (focal loss, recommended):
sbatch run_scripts/run_train_full_v2.sh

# v1 (subsample + Bayesian correction):
sbatch run_scripts/run_train_full_v1.sh
```

Both scripts target 3 H100 nodes × 4 GPUs = 12 GPUs total, 24h time limit.

### Resume from Checkpoint

```bash
sbatch run_scripts/run_train_full_v2.sh \
    --resume ../model_checkpoints/CHARM_JOINT_v2/charm_joint_step001000_ph3.pth
```

The `--resume` flag is forwarded to `run_charm_joint_ddp.py`.

### Binary Loss Modes

The binary head (empty vs. occupied voxel) supports four loss modes, set via `binary_loss_mode` in the config:

| Mode | Description | When to use |
| --- | --- | --- |
| `none` | Standard mean NLL over all voxels (legacy) | Only for high-occupancy runs where class imbalance is negligible |
| `subsample` | Keep all occupied + equal random draw of empty (1:1 ratio); apply Bayesian prior correction at inference | v1 production run |
| `alpha` | Per-voxel inverse-frequency weighting, mean-1 normalised | Intermediate occupancy |
| `focal` | Focal loss `(1−p_correct)^γ · NLL`; directly learns P(occ\|features, π_true) | **v2 production run (recommended)** |

**Why focal over subsample:** Subsample trains under an artificial balanced prior (π=0.5) and requires a Bayesian correction factor at inference. This correction fails at extreme cosmologies (very low/high occupancy), causing 20–50% total-count errors. Focal loss uses all voxels under the true class distribution — no correction needed.

**Gamma choice:** `gamma=2.0` is the standard value (Lin et al. 2017). At the degenerate empty-plateau (pw_occ≈π), gamma=2 gives a 66,886:1 occupied:empty gradient ratio, decisively breaking the plateau. The hard/easy suppression curve has a knee at gamma≈1.5–2, making values above 2 give diminishing returns. The gradient balance in the well-trained regime is insensitive to gamma across [1, 3].

**Implementation note — base loss must be `−log(pw_correct)`, not the raw GMM NLL.** The `SumGaussModel` with `sigma=0.05` gives `raw_nll = −log(pw_correct) − 2.077` (the `2.077 = log(peak_density)` offset from the narrow Gaussian). This offset makes `raw_nll` **negative** as soon as `pw_correct > 12.5%`, inverting the focal gradient direction and driving the model to a pathological equilibrium at `pw_correct ≈ 0.33` for *both* occupied and empty voxels, causing 3–6× too many halos at inference. The fix in `combined_models_v2.py` uses `cls_nll = −log(pw_correct)` (standard cross-entropy, always ≥ 0) as the base loss.

### Post-training: Binary Prior Calibration (v1/subsample only)

After a v1 (subsample) training run, calibrate the binary prior for accurate occupancy counts:

```bash
python charm/calibrate_binary_prior.py \
    --config run_configs/TRAIN_CHARM_JOINT_v1.yaml \
    --checkpoint ../model_checkpoints/CHARM_JOINT_v1/charm_joint_best_val.pth
```

This fits a polynomial ridge regression on the log occupancy fraction and saves a `binary_prior_calibrator.npz` alongside the checkpoint. **This step is not needed for v2 (focal loss) runs.**

### Training Configuration

Key settings in `run_configs/BASE_CONFIG.yaml`:

```yaml
sim_settings:
  Nmax: 4              # max halos per voxel (overridden to 8 in full runs)
  ns_h: 128            # halo grid side
  z_all_FP: [0.5, 'v_0.5']   # density + velocity at z=0.5 (4 channels)

network_settings:
  nfeature_cnn: 32     # CNN base channel width
  use_film: true       # FiLM cosmology modulation in encoder
  hidden_dim_MAF: 128  # MLP width for all flow heads
  ngauss_Nhalos: 4     # one component per Nmax class (overridden to 8 in full runs)

train_settings:
  nsims_per_batch: 200
  use_kendall: true    # automatic multi-task loss balancing
  staggered: true      # add one head per phase (recommended)
  torch_compile: true  # ~10–20% throughput gain on H100/A100
  binary_loss_mode: none  # override to 'focal' (v2) or 'subsample' (v1)
```

### Staggered Training Phases (v1 / v2)

The v1 and v2 phase schedules are identical; only the binary loss mode differs:

| Phase | Active heads | Epochs | Peak LR |
| --- | --- | --- | --- |
| 0 | binary, multi | 300 | 5e-4 |
| 1 | + m1 | 300 | 5e-4 |
| 2 | + mdiff | 300 | 2e-4 |
| 3 | + pos | 300 | 2e-4 |
| 4 | + vel | 300 | 2e-4 |
| 5 | + conc | 4500 | 2e-4 |

At each phase transition, old heads' LR is scaled by `frozen_lr_scale=0.3`.

### Weights & Biases

W&B logging is enabled by default. Set your API key:

```bash
export WANDB_API_KEY=your_key
# or: wandb login
```

Disable per run: add `--no_wandb` to the training command.

Logged metrics: per-head NLL losses, Kendall σ values, gradient norm, learning rates, GPU memory usage.

---

## Inference / Sampling

Run `charm/run_inference_v2.py` with a trained checkpoint:

```bash
python charm/run_inference_v2.py \
    --config run_configs/TRAIN_CHARM_JOINT_v2.yaml \
    --checkpoint ../model_checkpoints/CHARM_JOINT_v2/charm_joint_best_val.pth \
    --isim 1800
```

For v1 (subsample) checkpoints, also pass the calibrator:

```bash
python charm/run_inference_v2.py \
    --config run_configs/TRAIN_CHARM_JOINT_v1.yaml \
    --checkpoint ../model_checkpoints/CHARM_JOINT_v1/charm_joint_best_val.pth \
    --binary_prior_calibrator ../model_checkpoints/CHARM_JOINT_v1/binary_prior_calibrator.npz \
    --isim 1800
```

For focal-loss (v2) checkpoints, no calibrator is needed — the model predicts P(occ|features, π_true) directly.

### Programmatic Sampling

```python
import torch
from charm import CHARM_Model, CNN3D_stackout_v2
# ... build model with same config as training ...

ckpt = torch.load('charm_joint_best_val.pth', map_location='cuda')
model.load_state_dict(ckpt['model_state'])
model.eval()

with torch.no_grad():
    samples = model.sample(
        cond_x     = dm_cube_gpu,    # (1, nbatches, ninp, D_pad, D_pad, D_pad)
        cond_x_nsh = dm_nsh_gpu,     # (1, N_vox, ninp)
        cond_cosmo = cosmo_gpu,      # (1, N_vox, ncosmo)
    )

# samples is a dict:
# samples['ntot']  — (N_vox,) array: number of halos per voxel
# samples['m1']    — (N_vox,) tensor: log mass of heaviest halo
# samples['mdiff'] — (N_vox, Nmax-1) tensor: mass differences
# samples['vel']   — (N_vox, Nmax*3) tensor: velocity residuals
# samples['conc']  — (N_vox, Nmax) tensor: concentrations
# samples['pos']   — (N_vox, Nmax*3) tensor: position offsets
```

---

## Pre-trained Models

Three pre-trained checkpoints from an earlier separate-model training regime are included in `charm/trained_models/`:

| File | Contents |
| --- | --- |
| `charm_model_massNtot_bestfit_v2.pth` | Occupancy (binary + multi) + mass (M1 + Mdiff) heads |
| `charm_model_vel_bestfit_v2.pth` | Velocity head |
| `charm_model_conc_bestfit_v2.pth` | Concentration head |

The joint model (`run_charm_joint_ddp.py`) supersedes these.

---

## Technical Notes

### Memory

- Full joint training (all 7 heads, `nfeature_cnn=32`, `nsims_per_batch=200`) requires ~20–24 GB GPU memory per rank.
- Reduce `nsims_per_batch` to 100 if OOM at Phase 5 (all heads active).
- Enable gradient checkpointing: `grad_checkpoint_encoder: true` in the config.
- `torch_compile: true` gives ~10–20% throughput on H100/A100; set to `false` if Triton compilation fails.

### Numerical Precision

- Training uses `bfloat16` autocast for the forward pass.
- Training data is stored as `float16` in HDF5 and cast to `float32` on GPU.
- TF32 is enabled for both matmul and cuDNN operations.

### Shard Layout

12 training shards use **interleaved** assignment: shard `r` contains sims `r, r+12, r+24, …`. This ensures each GPU sees a diverse cosmological distribution throughout training. The number of shards must divide `nsims_train` (1800).

### Nmax

`Nmax=8` in v1 and v2 full runs (halos per voxel). The multiclass head (`ngauss_Nhalos`) must equal `Nmax`. Training shards store up to `nMax_h_raw=12` halos per voxel in raw HDF5 (to allow future Nmax increases without reprocessing halos).

---

## Dependencies

| Package | Purpose |
| --- | --- |
| `torch` | Model training and inference |
| `numpy`, `scipy` | Numerical operations |
| `h5py` | HDF5 training data I/O |
| `colossus` | Halo concentration model (Diemer+2019) |
| `Pylians` (MAS_library) | NGP/CIC mass assignment for density fields |
| `nbodykit` | BigFile format reading for FastPM outputs |
| `tqdm` | Progress bars |
| `Cython` | NGP property assignment extension (`ngp_funcs`) |
| `wandb` | (optional) Experiment tracking |
| `yaml` | Config loading |
