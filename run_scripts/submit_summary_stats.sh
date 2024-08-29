#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=save_stats
#SBATCH -p ccm
#SBATCH --mem=256G
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err


module purge
module load python
source ~/miniconda3/bin/activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/charm/
srun python test_mp_wavelets.py 0 128
