#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/ceph/users/spandey/CHARM_v2/CHARM}
PYTHON_EXEC=${PYTHON_EXEC:-/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python}

SIM_START=${SIM_START:-1900}
SIM_END=${SIM_END:-2000}
WORKERS=${WORKERS:-25}
THREADS_PER_WORKER=${THREADS_PER_WORKER:-1}
DEVICE=${DEVICE:-cpu}

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

run_step() {
    echo
    printf '[%s] Running:' "$(date '+%F %T')"
    printf ' %q' "$@"
    echo
    "$@"
}

run_test_inference() {
    local config=$1

    run_step "$PYTHON_EXEC" charm/inferers/run_test_inference_v2.py \
        --config "$config" \
        --sim_start "$SIM_START" \
        --sim_end "$SIM_END" \
        --num_workers "$WORKERS" \
        --threads_per_worker "$THREADS_PER_WORKER" \
        --device "$DEVICE" \
        --overwrite
}

plot_inference_ratios() {
    local config=$1

    run_step "$PYTHON_EXEC" charm/plotters/plot_inference_ratios_v2.py \
        --config "$config" \
        --sim_start "$SIM_START" \
        --sim_end "$SIM_END" \
        --workers "$WORKERS"
}

run_test_inference run_configs/TRAIN_CHARM_JOINT_v2vel_finetune2.yaml
plot_inference_ratios run_configs/TRAIN_CHARM_JOINT_v2vel_finetune2.yaml

run_test_inference run_configs/TRAIN_CHARM_JOINT_v2vel_finetune3.yaml
plot_inference_ratios run_configs/TRAIN_CHARM_JOINT_v2vel_finetune3.yaml

echo
echo "All inference and plotting jobs finished at $(date '+%F %T')."
