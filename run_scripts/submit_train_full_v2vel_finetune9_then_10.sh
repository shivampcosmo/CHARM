#!/bin/bash
set -euo pipefail

# Submit finetune9, then submit finetune10 with a Slurm dependency so ft10
# starts only after ft9 has finished. Uses afterany so ft10 is released after
# ft9 completes regardless of success/failure; switch to afterok below if ft10
# should run only after a successful ft9.

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
FT9_SCRIPT=$REPO/run_scripts/run_train_full_v2vel_finetune9.sh
FT10_SCRIPT=$REPO/run_scripts/run_train_full_v2vel_finetune10.sh
DEPENDENCY_TYPE=${DEPENDENCY_TYPE:-afterany}

cd "$REPO"

ft9_job_id=$(sbatch --parsable "$FT9_SCRIPT")
ft10_job_id=$(sbatch --parsable --dependency="${DEPENDENCY_TYPE}:${ft9_job_id}" "$FT10_SCRIPT")

echo "Submitted finetune9:  $ft9_job_id"
echo "Submitted finetune10: $ft10_job_id"
echo "finetune10 dependency: ${DEPENDENCY_TYPE}:${ft9_job_id}"
