#!/bin/bash
#SBATCH --job-name=charm_v2vel_ft10
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -C h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --output=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/train_full_v2vel_ft10_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/CHARM_v2/CHARM/logs/train_full_v2vel_ft10_%j.err

# Slurm dependencies require the upstream job ID at submission time. To run
# this only after finetune9 finishes, submit via:
#   run_scripts/submit_train_full_v2vel_finetune9_then_10.sh

set -euo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
CONFIG=$REPO/run_configs/TRAIN_CHARM_JOINT_v2vel_finetune10.yaml

mkdir -p $REPO/logs/wandb

source /etc/profile.d/modules.sh
module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi
PYTHON_EXEC=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python

cd $REPO

echo "Job $SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES  node=$(hostname)  $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $PYTHON_EXEC"
"$PYTHON_EXEC" -c "import sys, wandb; print('Python executable:', sys.executable); print('wandb:', getattr(wandb, '__file__', None), 'has_init=', hasattr(wandb, 'init'))"

EXTRA_ARGS="${@}"

master_node=$SLURMD_NODENAME
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=0

export PYTHONPATH=$REPO:${PYTHONPATH:-}
export WANDB_DIR=$REPO/logs

srun "$PYTHON_EXEC" -m torch.distributed.run \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        charm/run_charm_joint_v2vel_ddp.py \
        --config "$CONFIG" \
        $EXTRA_ARGS

echo "Fine-tune finished at $(date)"
