#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=TEST0_vir
#SBATCH -p ccm
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/prep_data/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/prep_data/slurm_logs/%x.%j.err

source ~/miniconda3/bin/activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/prep_data/
time srun python process_halos_peakheight_conc_wvel_quijote.py rockstar_vir 3
echo "done"


