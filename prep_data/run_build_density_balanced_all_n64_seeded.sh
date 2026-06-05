#!/bin/bash
#SBATCH --job-name=charm_db_all64
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/density_balanced_all64_seed%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/density_balanced_all64_seed%A_%a.err

# Build density-balanced shards from all 512 subvolumes per simulation.
#
# This does not exclude any previous shards. It selects n64 subvolumes per
# simulation uniformly over per-simulation background-density bins.
#
# Multiple realizations:
#   SELECTION_SEED=0 sbatch prep_data/run_build_density_balanced_all_n64_seeded.sh
#   SELECTION_SEED=1 sbatch prep_data/run_build_density_balanced_all_n64_seeded.sh
#
# Output defaults to:
#   /mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12_density_balanced_all_n64_seed${SELECTION_SEED}
#
# Tasks 0-11 -> training shards
# Task  12   -> validation shard

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM

CONFIG=${CONFIG:-$REPO/run_configs/TRAIN_CHARM_JOINT_v2vel.yaml}
SELECTION_SEED=${SELECTION_SEED:-0}
OUT_DIR=${OUT_DIR:-/mnt/ceph/users/spandey/CHARM_v2/data/shards_Mmin5e12_density_balanced_all_n64_seed${SELECTION_SEED}}
N_SELECT=${N_SELECT:-64}
N_DENSITY_BINS=${N_DENSITY_BINS:-10}
OVERWRITE=${OVERWRITE:-0}
MAKE_DIAGNOSTICS=${MAKE_DIAGNOSTICS:-1}
N_DIAGNOSTIC_SIMS=${N_DIAGNOSTIC_SIMS:-10}
DIAGNOSTIC_SEED=${DIAGNOSTIC_SEED:-20260530}
DIAGNOSTIC_OUT_DIR=${DIAGNOSTIC_OUT_DIR:-$REPO/prep_data/test_output/all_n64_seed${SELECTION_SEED}_nbins${N_DENSITY_BINS}}

mkdir -p "$REPO/logs"
mkdir -p "$DIAGNOSTIC_OUT_DIR"

source ~/miniconda3/bin/activate nbodykit

cd "$REPO"

cmd=(
    python prep_data/build_density_balanced_training_shards.py
    --config "$CONFIG"
    --shard_rank "$SLURM_ARRAY_TASK_ID"
    --candidate_mode all
    --out_dir "$OUT_DIR"
    --n_select "$N_SELECT"
    --n_density_bins "$N_DENSITY_BINS"
    --selection_seed "$SELECTION_SEED"
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

echo "Starting all-subvolume density-balanced n64 seed=$SELECTION_SEED task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "Output dir: $OUT_DIR"
echo "Command: ${cmd[*]}"
"${cmd[@]}"
echo "Done all-subvolume density-balanced n64 seed=$SELECTION_SEED task $SLURM_ARRAY_TASK_ID at $(date)"
