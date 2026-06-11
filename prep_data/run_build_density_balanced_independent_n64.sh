#!/bin/bash
#SBATCH --job-name=charm_db_ind64
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/density_balanced_ind64_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/density_balanced_ind64_%A_%a.err

# Build a third independent density-balanced n64 shard set.
#
# Candidate pool per simulation:
#   all 512 subvolumes
#   minus metadata/subvol_ids already used in:
#     1. /mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12
#     2. /mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12_density_balanced_remaining_n64
#
# Selection:
#   - n_select = 64 subvolumes per simulation
#   - n_density_bins = 10
#   - exclude_source = metadata, so the job fails rather than silently falling
#     back if previous-shard metadata is unavailable.
#
# Tasks 0-11 -> training shards
# Task  12   -> validation shard
#
# Submit:
#   sbatch prep_data/run_build_density_balanced_independent_n64.sh

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM

CONFIG=${CONFIG:-$REPO/run_configs/TRAIN_CHARM_JOINT_v2vel.yaml}
OUT_DIR=${OUT_DIR:-/mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12_density_balanced_independent_n64}
N_SELECT=${N_SELECT:-64}
N_DENSITY_BINS=${N_DENSITY_BINS:-10}
OVERWRITE=${OVERWRITE:-0}
MAKE_DIAGNOSTICS=${MAKE_DIAGNOSTICS:-1}
N_DIAGNOSTIC_SIMS=${N_DIAGNOSTIC_SIMS:-10}
DIAGNOSTIC_SEED=${DIAGNOSTIC_SEED:-20260529}
DIAGNOSTIC_OUT_DIR=${DIAGNOSTIC_OUT_DIR:-$REPO/prep_data/test_output/independent_n64_nbins10}

EXCLUDE_SHARD_DIRS=(
    /mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12
    /mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12_density_balanced_remaining_n64
)

mkdir -p "$REPO/logs"
mkdir -p "$DIAGNOSTIC_OUT_DIR"

source ~/miniconda3/bin/activate nbodykit

cd "$REPO"

cmd=(
    python prep_data/build_density_balanced_training_shards.py
    --config "$CONFIG"
    --shard_rank "$SLURM_ARRAY_TASK_ID"
    --candidate_mode remaining
    --exclude_source metadata
    --exclude_shard_dirs "${EXCLUDE_SHARD_DIRS[@]}"
    --out_dir "$OUT_DIR"
    --n_select "$N_SELECT"
    --n_density_bins "$N_DENSITY_BINS"
    --n_diagnostic_sims "$N_DIAGNOSTIC_SIMS"
    --diagnostic_seed "$DIAGNOSTIC_SEED"
    --diagnostic_out_dir "$DIAGNOSTIC_OUT_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
fi

# Avoid every array task racing to write the same diagnostic plots.
if [[ "$MAKE_DIAGNOSTICS" == "1" && "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    cmd+=(--make_diagnostics)
fi

echo "Starting independent density-balanced n64 shard task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "Output dir: $OUT_DIR"
echo "Excluding previous shard dirs:"
printf '  %s\n' "${EXCLUDE_SHARD_DIRS[@]}"
echo "Command: ${cmd[*]}"
"${cmd[@]}"
echo "Done independent density-balanced n64 shard task $SLURM_ARRAY_TASK_ID at $(date)"
