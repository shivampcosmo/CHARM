#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --job-name=try2
#SBATCH -p ccm
#SBATCH --mem=256G
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/charm/sweep/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/charm/sweep/slurm_logs/%x.%j.err


module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

cd /mnt/home/spandey/ceph/CHARM/charm/sweep/
srun --cpu-bind=cores python run_sbi_mp.py quad s0_s1_s2
srun --cpu-bind=cores python run_sbi_mp.py quad s1_s2
srun --cpu-bind=cores python run_sbi_mp.py quad None