#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=16:00:00
#SBATCH --job-name=run_infer_SNPE_rsd_mock
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
srun python run_inference_hod.py rsd all SNPE mock s1_s2 30 3
srun python run_inference_hod.py rsd all SNPE mock s1_s2 40 3
srun python run_inference_hod.py rsd all SNPE mock s1_s2 50 3
srun python run_inference_hod.py rsd all SNPE mock s1_s2 60 3
srun python run_inference_hod.py rsd all SNPE mock s1_s2 30 5
srun python run_inference_hod.py rsd all SNPE mock s1_s2 30 7
srun python run_inference_hod.py rsd all SNPE mock s1_s2 30 9
srun python run_inference_hod.py rsd all SNPE mock s1_s2 20 2
srun python run_inference_hod.py rsd all SNPE mock s1_s2 50 7
srun python run_inference_hod.py rsd all SNPE mock s1_s2 60 9
srun python run_inference_hod.py rsd all SNPE mock s1_s2 20 9
