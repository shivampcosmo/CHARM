#!/bin/bash
#SBATCH --job-name=charm_db_shards
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/density_balanced_shards_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/density_balanced_shards_%A_%a.err

# Build density-balanced CHARM HDF5 shards.
#
# Default setup:
#   - CONFIG: TRAIN_CHARM_JOINT_v2vel.yaml
#   - candidate mode: remaining
#   - exclude source: auto, preferring metadata in the existing random shards
#   - n_select: 128 subvolumes per simulation
#   - n_density_bins: 16 density bins per simulation
#
# Tasks 0-11 -> training shards
# Task  12   -> validation shard
#
# Submit:
#   sbatch prep_data/run_build_density_balanced_shards.sh
#
# Useful overrides:
#   CANDIDATE_MODE=all sbatch prep_data/run_build_density_balanced_shards.sh
#   CONFIG=/path/to/config.yaml sbatch prep_data/run_build_density_balanced_shards.sh
#   OUT_DIR=/path/to/output_shards sbatch prep_data/run_build_density_balanced_shards.sh
#   OVERWRITE=1 sbatch prep_data/run_build_density_balanced_shards.sh
#   MAKE_DIAGNOSTICS=1 sbatch prep_data/run_build_density_balanced_shards.sh

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM

CONFIG=${CONFIG:-$REPO/run_configs/TRAIN_CHARM_JOINT_v2vel.yaml}
CANDIDATE_MODE=${CANDIDATE_MODE:-remaining}       # remaining | all
EXCLUDE_SOURCE=${EXCLUDE_SOURCE:-auto}            # auto | metadata | rng
EXCLUDE_SHARD_DIR=${EXCLUDE_SHARD_DIR:-}          # default: config data_settings.shard_dir
OUT_DIR=${OUT_DIR:-}                              # default: *_density_balanced_${CANDIDATE_MODE}
N_SELECT=${N_SELECT:-128}
N_DENSITY_BINS=${N_DENSITY_BINS:-16}
OVERWRITE=${OVERWRITE:-0}
MAKE_DIAGNOSTICS=${MAKE_DIAGNOSTICS:-0}
DIAGNOSTIC_OUT_DIR=${DIAGNOSTIC_OUT_DIR:-$REPO/prep_data/test_output}

mkdir -p "$REPO/logs"
mkdir -p "$DIAGNOSTIC_OUT_DIR"

source ~/miniconda3/bin/activate nbodykit

cd "$REPO"

cmd=(
    python prep_data/build_density_balanced_training_shards.py
    --config "$CONFIG"
    --shard_rank "$SLURM_ARRAY_TASK_ID"
    --candidate_mode "$CANDIDATE_MODE"
    --exclude_source "$EXCLUDE_SOURCE"
    --n_select "$N_SELECT"
    --n_density_bins "$N_DENSITY_BINS"
    --diagnostic_out_dir "$DIAGNOSTIC_OUT_DIR"
)

if [[ -n "$EXCLUDE_SHARD_DIR" ]]; then
    cmd+=(--exclude_shard_dir "$EXCLUDE_SHARD_DIR")
fi

if [[ -n "$OUT_DIR" ]]; then
    cmd+=(--out_dir "$OUT_DIR")
fi

if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
fi

# Avoid every array task racing to write the same diagnostic plots.
if [[ "$MAKE_DIAGNOSTICS" == "1" && "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    cmd+=(--make_diagnostics)
fi

echo "Starting density-balanced shard task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "Command: ${cmd[*]}"
"${cmd[@]}"
echo "Done density-balanced shard task $SLURM_ARRAY_TASK_ID at $(date)"
