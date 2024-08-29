#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=TEST0
#SBATCH -p ccm
#SBATCH --output=/mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/slurm_logs/%x.%j.err

source ~/miniconda3/bin/activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/prep_data/
time srun python process_halos_wvel_quijote.py
echo "done"


