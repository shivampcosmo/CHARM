#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=50
#SBATCH --time=2:00:00
#SBATCH --job-name=gpu_mp_save_wavelets
#SBATCH -p gpu
#SBATCH -C a100-40gb
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err

module purge
module load python
source ~/miniconda3/bin/activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/charm/
time srun python calc_summary_stats_galaxy_wavelets.py 0 2000 25

