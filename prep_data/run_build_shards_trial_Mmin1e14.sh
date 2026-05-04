#!/bin/bash
#SBATCH --job-name=charm_shards_trial
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/shards_trial_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/shards_trial_%A_%a.err

# Build per-GPU HDF5 training shards for the trial run (Mmin = 1e14 Msun/h).
#   Tasks 0-3 → training shards (50 sims × 64 subvols = 3200 rows each)
#   Task  4   → validation shard (20 sims × 64 subvols = 1280 rows)
#
# Requires run_process_halos_trial_Mmin1e14.sh to have completed for sims 0-219.
#
# Submit:
#   sbatch prep_data/run_build_shards_trial_Mmin1e14.sh

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml

mkdir -p $REPO/logs

source ~/miniconda3/bin/activate nbodykit

cd $REPO

echo "Starting shard $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
python prep_data/build_training_shards.py \
    --config     $CONFIG \
    --shard_rank $SLURM_ARRAY_TASK_ID
echo "Done shard $SLURM_ARRAY_TASK_ID at $(date)"
