#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=TEST0
#SBATCH -p ccm
#SBATCH --output=/mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/slurm_logs/%x.%j.err

# __conda_setup="$('/mnt/home/spandey/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/mnt/home/spandey/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

# conda activate nbodykit

cd /mnt/home/spandey/ceph/CHARM/prep_data/
time srun python /mnt/home/spandey/ceph/CHARM/prep_data/random_subsamp_halo_rho_vel.py
echo "done"


