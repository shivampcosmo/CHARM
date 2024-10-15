#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=4:00:00
#SBATCH --job-name=test0_conc
#SBATCH -p gpu
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err

# module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

# master_node=$SLURMD_NODENAME

cd /mnt/home/spandey/ceph/CHARM/
srun python /mnt/home/spandey/ceph/CHARM/charm/run_conc_test.py
echo "done"
