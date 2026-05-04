#!/bin/bash
#SBATCH --job-name=charm_shards
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/shards_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/shards_%A_%a.err

# Build per-GPU HDF5 training shards for the production run
# (Mmin = 5e12 Msun/h, lgMmin ≈ 12.70).
#
#   Tasks 0-11 → training shards (1800/12 = 150 sims × 64 subvols = 9,600 rows each)
#   Task  12   → validation shard (100 sims × 64 subvols = 6,400 rows)
#
# build_training_shards.py auto-routes any task id >= n_shards to the val
# shard (n_shards=12 here, so task 12 builds val/* automatically).
#
# Requires run_process_halos.sh to have completed for sims 0-1899 (the
# test split, 1900-1999, is not sharded — it's loaded sim-by-sim at
# inference time).
#
# Submit (after halo processing is done):
#   sbatch prep_data/run_build_shards.sh
#
# Expected wall time: ~30-60 min per shard (dominated by HDF5 I/O).

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT_v0.yaml

mkdir -p $REPO/logs

source ~/miniconda3/bin/activate nbodykit

cd $REPO

echo "Starting shard $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
python prep_data/build_training_shards.py \
    --config     $CONFIG \
    --shard_rank $SLURM_ARRAY_TASK_ID
echo "Done shard $SLURM_ARRAY_TASK_ID at $(date)"
