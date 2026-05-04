#!/bin/bash
# run_process_halos.sh
# --------------------
# Master submitter for the production halo-processing pipeline
# (Mmin = 5e12 Msun/h, lgMmin ≈ 12.70).
#
# All 2000 Quijote LH simulations:
#   sims 0-1799    → training
#   sims 1800-1899 → validation
#   sims 1900-1999 → test (never touched during training)
#
# This is a PLAIN BASH SCRIPT, not an sbatch script. It submits ONE
# sbatch_workers/halos_chunk.sbatch array job (200 sims) plus ONE tiny
# relay job chained via --dependency=afterany. The relay runs when the
# array finishes and re-invokes this script with START_CHUNK+1, keeping
# at most 2 jobs in the SLURM queue at any time (respects
# QOSMaxSubmitJobPerUser limits).
#
# `afterany` (not `afterok`) is intentional: a failed task in chunk N
# does NOT block chunk N+1, since sims are independent and the worker
# is idempotent (process_halos_quijote_v2.py skips sims whose per-sim
# HDF5 already exists). Re-run failed sims afterwards via a small
# array, e.g. `sbatch --array=42,107 sbatch_workers/halos_chunk.sbatch`.
#
# Usage:
#   bash prep_data/run_process_halos.sh                  # all 10 chunks (sims 0-1999)
#   START_CHUNK=4 bash prep_data/run_process_halos.sh    # resume from chunk 4 onward
#   START_CHUNK=2 END_CHUNK=5 bash prep_data/run_process_halos.sh   # subrange
#
# After submission, monitor with:
#   squeue -u $USER -t PD,R   # pending + running
#   tail -f logs/halos_<jobid>_*.out

set -uo pipefail

REPO=/mnt/ceph/users/spandey/CHARM_v2/CHARM
WORKER=$REPO/prep_data/sbatch_workers/halos_chunk.sbatch
CHUNK_SIZE=200
N_CHUNKS=10                       # 10 × 200 = 2000 sims total

START_CHUNK=${START_CHUNK:-0}
END_CHUNK=${END_CHUNK:-$(( N_CHUNKS - 1 ))}

# ── Pre-flight checks ────────────────────────────────────────────────
if [[ ! -f "$WORKER" ]]; then
    echo "ERROR: worker script not found: $WORKER" >&2
    exit 1
fi
if (( START_CHUNK < 0 || END_CHUNK >= N_CHUNKS || START_CHUNK > END_CHUNK )); then
    echo "ERROR: invalid chunk range [$START_CHUNK, $END_CHUNK]; must satisfy" >&2
    echo "       0 <= START_CHUNK <= END_CHUNK < $N_CHUNKS" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found on PATH — run this on a SLURM submit host" >&2
    exit 1
fi

mkdir -p $REPO/logs

CHUNK=$START_CHUNK
OFFSET=$(( CHUNK * CHUNK_SIZE ))
ISIM_LO=$OFFSET
ISIM_HI=$(( OFFSET + CHUNK_SIZE - 1 ))

echo "================================================================"
echo "  CHARM halo-processing (self-chaining relay, max 2 jobs queued)"
echo "  chunk $CHUNK  sims $ISIM_LO-$ISIM_HI   remaining: $((END_CHUNK - CHUNK)) more after"
echo "  worker: $WORKER"
echo "  $(date)"
echo "================================================================"

# ── Submit this chunk ────────────────────────────────────────────────
JOBID=$(sbatch --parsable \
               --export=ALL,OFFSET=$OFFSET \
               $WORKER)
rc=$?
JOBID=${JOBID%%;*}

if [[ $rc -ne 0 || -z "$JOBID" ]]; then
    echo "  chunk $CHUNK: SUBMIT FAILED (sbatch exit=$rc)"
    echo "  retry with:"
    echo "    START_CHUNK=$CHUNK bash prep_data/run_process_halos.sh"
    exit $rc
fi

printf "  chunk %d  sims %4d-%4d   jobid %-12s (runs immediately)\n" \
    "$CHUNK" "$ISIM_LO" "$ISIM_HI" "$JOBID"

# ── Submit relay job that will trigger the next chunk ────────────────
if (( CHUNK < END_CHUNK )); then
    NEXT=$(( CHUNK + 1 ))
    RELAY_CMD="START_CHUNK=$NEXT END_CHUNK=$END_CHUNK bash $REPO/prep_data/run_process_halos.sh"
    RELAY_JOBID=$(sbatch --parsable \
        --job-name=charm_relay \
        --dependency=afterany:$JOBID \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=512M --time=00:10:00 \
        -p cmbas \
        --output=$REPO/logs/relay_%j.out \
        --error=$REPO/logs/relay_%j.err \
        --wrap="$RELAY_CMD")
    rc_relay=$?
    RELAY_JOBID=${RELAY_JOBID%%;*}

    if [[ $rc_relay -ne 0 || -z "$RELAY_JOBID" ]]; then
        echo "  WARNING: relay submit failed (exit=$rc_relay)"
        echo "  chunk $CHUNK ($JOBID) will still run; resume manually afterward with:"
        echo "    START_CHUNK=$NEXT END_CHUNK=$END_CHUNK bash prep_data/run_process_halos.sh"
    else
        printf "  relay     jobid %-12s (submits chunk %d after %s finishes)\n" \
            "$RELAY_JOBID" "$NEXT" "$JOBID"
        echo "----------------------------------------------------------------"
        echo "  queue footprint: 1 array job + 1 relay at a time"
        echo "  squeue -u \$USER -t PD,R"
        echo "  scancel $JOBID $RELAY_JOBID   # abort (running chunk completes normally)"
    fi
fi

echo "================================================================"
