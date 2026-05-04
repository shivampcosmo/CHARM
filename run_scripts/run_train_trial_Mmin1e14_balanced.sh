#!/bin/bash
#SBATCH --job-name=charm_train_trial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -C h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/train_trial_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/train_trial_%j.err

# Training run: CHARM joint model, Mmin = 1e14 Msun/h, 4x A100-80GB.
# n_shards=4 in config → must use exactly 4 GPUs.
#
# Submit:
#   sbatch run_scripts/run_train_trial_Mmin1e14.sh
#
# Resume from checkpoint:
#   sbatch run_scripts/run_train_trial_Mmin1e14.sh --resume <path/to/checkpoint.pth>
#   (the --resume arg is forwarded to run_charm_joint_ddp.py)
#
# To disable W&B: add --no_wandb below.
# To use W&B:     set WANDB_API_KEY in your environment or run `wandb login` once.

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT_trial_Mmin1e14_balanced.yaml

mkdir -p $REPO/logs

source /etc/profile.d/modules.sh
module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

cd $REPO

echo "Job $SLURM_JOB_ID  node $(hostname)  $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Forward any extra CLI args (e.g. --resume, --no_wandb) passed to sbatch
EXTRA_ARGS="${@}"

torchrun \
    --standalone \
    --nproc_per_node=4 \
    charm/run_charm_joint_ddp.py \
    --config "$CONFIG" \
    $EXTRA_ARGS

echo "Training finished at $(date)"
