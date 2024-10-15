#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=SAVESUBSAMP_200c
#SBATCH --cpus-per-task=16
#SBATCH -p ccm
#SBATCH --mem=256G
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/prep_data/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/prep_data/slurm_logs/%x.%j.err


source ~/miniconda3/bin/activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/prep_data/
time srun python random_subsamp_halo_rho_vel_conc_peak.py 200c
echo "done"


