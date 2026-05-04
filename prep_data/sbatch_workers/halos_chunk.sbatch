#!/bin/bash
#SBATCH --job-name=charm_halos_chunk
#SBATCH --array=0-199
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -p cmbas
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/halos_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/halos_%A_%a.err

# Worker for a single 200-sim chunk of the halo-processing pipeline.
# Not intended for direct submission — the master `run_process_halos.sh`
# submits all 10 chunks at once and chains them via
# `--dependency=afterany:<prev_jobid>`, so chunks 1-9 sit in
# Pending(Dependency) state and SLURM releases each one only when its
# predecessor terminates.
#
# Actual sim id = OFFSET + SLURM_ARRAY_TASK_ID:
#   OFFSET=0    → sims 0-199
#   OFFSET=200  → sims 200-399
#   …
#   OFFSET=1800 → sims 1800-1999
#
# Direct re-runs of failed sims (using whatever ids you need):
#   sbatch --array=42,107 prep_data/sbatch_workers/halos_chunk.sbatch
# (defaults OFFSET to 0; pass --export=ALL,OFFSET=N to shift.)

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT.yaml

mkdir -p $REPO/logs

source ~/miniconda3/bin/activate nbodykit

cd $REPO

ISIM=$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))

echo "Starting sim $ISIM (array task $SLURM_ARRAY_TASK_ID, OFFSET=${OFFSET:-0}) on $(hostname) at $(date)"
python prep_data/process_halos_quijote_v2.py \
    --config  $CONFIG \
    --isim    $ISIM
echo "Done sim $ISIM at $(date)"
