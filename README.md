# CHARM v2

CHARM paints full dark-matter halo catalogs onto FastPM simulations. Given a
FastPM density/velocity field and a 5-parameter cosmology vector, it samples
halo counts, masses, velocities, concentrations, and sub-voxel positions on a
128^3 halo grid.

The current codebase is the joint CHARM v2 pipeline: one shared 3D CNN encoder
with separate probabilistic heads for occupancy/counts and halo properties.

## Model

```text
FastPM density + velocity + cosmology
        |
        v
CNN3D_stackout_v2 encoder with FiLM cosmology conditioning
        |
        v
per-voxel conditioning features
        |
        +--> binary occupancy head       SumGauss classifier
        +--> multiclass Nhalos head      SumGauss classifier
        +--> M1 head                     neural spline flow
        +--> Mdiff head                  autoregressive neural spline flow
        +--> velocity head               autoregressive neural spline flow
        +--> concentration head          autoregressive neural spline flow
        +--> position head               autoregressive neural spline flow
```

Sampling order:

1. Sample occupied voxels and per-voxel halo counts.
2. Sample M1 conditioned on sampled count.
3. Sample Mdiff conditioned on sampled count and sampled M1.
4. Reconstruct all halo masses.
5. Sample velocity, concentration, and position conditioned on sampled masses.

Training can be teacher-forced or exposure-robust. The exposure trainer mixes
teacher-forced property losses with losses conditioned on sampled upstream
quantities, matching inference-time inputs more closely.

## Important Files

| Path | Purpose |
| --- | --- |
| `charm/combined_models_v2.py` | `CHARM_Model`; shared encoder plus all heads; sampling and loss logic. |
| `charm/all_models_v2.py` | Distribution heads: SumGauss classifiers and neural spline flows. |
| `charm/cnn_3d_stack_v2.py` | 3D CNN encoder with FiLM cosmology conditioning. |
| `charm/run_charm_joint_ddp.py` | Standard DDP joint trainer. |
| `charm/run_charm_joint_exposure_ddp.py` | Exposure-robust DDP trainer with rollout validation. |
| `charm/run_inference_v2.py` | Full catalog inference for one simulation. |
| `charm/run_test_inference_v2.py` | Batch inference over held-out test simulations. |
| `charm/calibrate_binary_prior.py` | Binary prior calibrator for subsampled/weighted binary heads. |
| `charm/plot_inference_v2.py` | Single-simulation mock/true diagnostics. |
| `charm/plot_inference_ratios_v2.py` | Aggregate mock/true ratio diagnostics. |
| `prep_data/process_halos_quijote_v2.py` | Rockstar halos to per-simulation HDF5. |
| `prep_data/build_training_shards.py` | Per-simulation HDF5 to per-GPU training shards. |
| `CURRENT_LIVE_CODE_SUMMARY.md` | Detailed developer-facing map of the live code. |

## Configs and Launchers

All configs inherit from `run_configs/BASE_CONFIG.yaml`.

| Config | Launcher | Notes |
| --- | --- | --- |
| `TRAIN_CHARM_JOINT_v2.yaml` | `run_train_full_v2.sh` | Current running baseline DDP training. |
| `TRAIN_CHARM_JOINT_v3.yaml` | none currently in `run_scripts/` | Exposure config using `binary_loss_mode: subsample`; needs binary prior calibration for inference. |
| `TRAIN_CHARM_JOINT_v4.yaml` | `run_train_full_v4.sh` | Exposure config using `binary_loss_mode: none`; longer final joint phase. |
| `TRAIN_CHARM_JOINT_v0/v1.yaml` | `run_train_full_v0/v1.sh` | Earlier production runs. |
| trial configs | trial SLURM scripts | Mmin=1e14 experiments. |

Current recommended new exposure-bias run:

```bash
sbatch run_scripts/run_train_full_v4.sh
```

Manual DDP launch:

```bash
torchrun --standalone --nproc_per_node=4 \
  charm/run_charm_joint_exposure_ddp.py \
  --config run_configs/TRAIN_CHARM_JOINT_v4.yaml
```

Resume:

```bash
sbatch run_scripts/run_train_full_v4.sh \
  --resume ../model_checkpoints/CHARM_JOINT_v4/charm_joint_resume.pth
```

## Training Modes

### Standard Trainer

`charm/run_charm_joint_ddp.py` computes teacher-forced likelihood losses:

- M1 sees true `Nhalos`.
- Mdiff sees true `Nhalos` and true M1.
- velocity/concentration/position see true masses.

This is stable and fast, but validation is also teacher-forced, so it can miss
inference-time error accumulation.

### Exposure-Robust Trainer

`charm/run_charm_joint_exposure_ddp.py` keeps supervised count losses and adds
scheduled sampled-upstream conditioning for downstream heads:

```text
L_property = (1 - p_student) * L_teacher_forced
           + p_student * L_sampled_conditioned
```

It also logs rollout validation metrics:

- occupied voxel count ratio
- total halo count ratio
- per-voxel Ntot PDF ratio
- normalized HMF ratio
- mass monotonicity violations
- NaN and bound violation rates

It saves `checkpoint_best_teacher.pth`, `checkpoint_best_rollout.pth`, and
`charm_joint_resume.pth`.

## Binary Occupancy Calibration

The binary head is the main control point for total halo counts.

| `binary_loss_mode` | Inference rule |
| --- | --- |
| `none` | No automatic prior correction. Check rollout count ratios. |
| `subsample` | Requires binary prior calibration/correction. |
| `alpha` | Requires binary prior calibration/correction. |
| `focal` | Intended to learn the true-prior posterior directly, but validate count calibration carefully. |

For `subsample` or `alpha`, fit the calibrator after training:

```bash
python charm/calibrate_binary_prior.py \
  --config run_configs/TRAIN_CHARM_JOINT_v3.yaml
```

Then pass it to inference, or let `run_inference_v2.py` auto-load
`binary_prior_calibrator.npz` from the checkpoint directory when applicable.

## Data Preparation

Process halo catalogs:

```bash
sbatch prep_data/run_process_halos.sh
```

Build training and validation shards:

```bash
sbatch prep_data/run_build_shards.sh
```

Expected outputs for the 5e12 mass cut:

```text
../data/halos_Mmin5e12/
../data/shards_Mmin5e12/CHARM_train_shard_{0..11}.h5
../data/shards_Mmin5e12/CHARM_val_shard.h5
```

## Inference

Single simulation:

```bash
python charm/run_inference_v2.py \
  --config run_configs/TRAIN_CHARM_JOINT_v4.yaml \
  --checkpoint ../model_checkpoints/CHARM_JOINT_v4/checkpoint_best_rollout.pth \
  --sim_id 1900
```

For a subsampled/alpha binary head:

```bash
python charm/run_inference_v2.py \
  --config run_configs/TRAIN_CHARM_JOINT_v3.yaml \
  --checkpoint ../model_checkpoints/CHARM_JOINT_v3/checkpoint_best_rollout.pth \
  --binary_prior_calibrator ../model_checkpoints/CHARM_JOINT_v3/binary_prior_calibrator.npz \
  --sim_id 1900
```

Batch inference over held-out test simulations:

```bash
python charm/run_test_inference_v2.py \
  --config run_configs/TRAIN_CHARM_JOINT_v4.yaml \
  --checkpoint ../model_checkpoints/CHARM_JOINT_v4/checkpoint_best_rollout.pth
```

Aggregate diagnostics:

```bash
python charm/plot_inference_ratios_v2.py \
  --config run_configs/TRAIN_CHARM_JOINT_v4.yaml
```

## Environment

Core requirements:

- Python 3.10+
- PyTorch 2.x with CUDA for training
- `numpy`, `scipy`, `h5py`, `pyyaml`, `tqdm`
- `colossus` for halo concentration processing
- `nbodykit` and Pylians/MAS tools for simulation IO and power spectra
- W&B is optional but supported

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

Build the Cython NGP helper for preprocessing:

```bash
cd prep_data
python setup_ngp.py build_ext --inplace
cd ..
```

## Notes

- Full production runs use 12 H100 GPUs through 3 SLURM nodes.
- Full-run configs use `Nmax: 8`; keep `network_settings.ngauss_Nhalos` equal to `Nmax`.
- Training uses bfloat16 autocast and TF32 matmul/cuDNN.
- Exposure training disables `torch_compile` by default because the sampled training path is dynamic.
- Do not evaluate exposure-bias fixes only with teacher-forced validation; use rollout and catalog-level diagnostics.
