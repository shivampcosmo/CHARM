#!/bin/bash
#SBATCH --job-name=charm_halos_trial
#SBATCH --array=0-219
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/halos_trial_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/halos_trial_%A_%a.err

# Process halo catalogs for the trial run (Mmin = 1e14 Msun/h).
#   sims 0-199   → training
#   sims 200-219 → validation
#
# Submit:
#   sbatch prep_data/run_process_halos_trial_Mmin1e14.sh
#
# Safe to rerun — process_halos_quijote_v2.py skips existing files.

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14.yaml

mkdir -p $REPO/logs

source ~/miniconda3/bin/activate nbodykit

cd $REPO

echo "Starting sim $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
python prep_data/process_halos_quijote_v2.py \
    --config  $CONFIG \
    --isim    $SLURM_ARRAY_TASK_ID
echo "Done sim $SLURM_ARRAY_TASK_ID at $(date)"
