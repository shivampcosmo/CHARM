#!/bin/bash
#SBATCH --job-name=charm_train_full_v2
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -C h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/train_full_v2_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/train_full_v2_%j.err

#
# Full training run v2: focal loss binary head (no Bayesian prior correction).
#
# Resume from checkpoint:
#   sbatch run_scripts/run_train_full_v2.sh --resume <path/to/checkpoint.pth>
#   (the --resume arg is forwarded to run_charm_joint_ddp.py)
#
# To disable W&B: add --no_wandb below.
# To use W&B:     set WANDB_API_KEY in your environment or run `wandb login` once.
#
# NOTE: no calibrate_binary_prior.py step needed after training — inference
# with focal loss uses binary_model.inverse() directly without any
# Bayesian prior correction.

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT_v2.yaml

mkdir -p $REPO/logs

source /etc/profile.d/modules.sh
module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

cd $REPO

echo "Job $SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES  node=$(hostname)  $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

EXTRA_ARGS="${@}"

master_node=$SLURMD_NODENAME
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=0

srun python `which torchrun` \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        charm/run_charm_joint_ddp.py \
        --config "$CONFIG" \
        $EXTRA_ARGS

echo "Training finished at $(date)"
