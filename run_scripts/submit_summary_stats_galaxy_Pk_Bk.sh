#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=save_stats_mp_gals
#SBATCH -p ccm
#SBATCH --mem=256G
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err

module purge
module load python
source ~/miniconda3/bin/activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/charm/
time srun python calc_summary_stats_galaxy_Pk_Bk_mp.py 0 2000

