#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=16:00:00
#SBATCH --job-name=run_infer_SNLE_rsd_mock
#SBATCH -p gpu
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err

module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi


cd /mnt/home/spandey/ceph/CHARM/charm/
srun python run_inference_hod.py rsd all SNLE mock s1
srun python run_inference_hod.py rsd all SNLE mock s1_s2
srun python run_inference_hod.py rsd all SNLE mock s0_s1_s2

srun python run_inference_hod.py rsd quad SNLE mock s1
srun python run_inference_hod.py rsd quad SNLE mock s1_s2
srun python run_inference_hod.py rsd quad SNLE mock s0_s1_s2

srun python run_inference_hod.py rsd mono SNLE mock s1
srun python run_inference_hod.py rsd mono SNLE mock s1_s2
srun python run_inference_hod.py rsd mono SNLE mock s0_s1_s2
