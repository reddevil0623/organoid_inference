#!/usr/bin/env bash
# Stage A1: fan the simulation grid out across hosts.
#
# Splits [0, N_SIMS) into per-host chunks proportional to each host's job
# count and launches one detached screen session per host. The cache and
# manifests live on shared storage, so every host feeds one evidence set.
#
# Usage:  ABC_REJ_REMOTE_DIR=/path/to/project bash scripts/launch_abc_rej_gen.sh
#
# Re-running is the resume action: a host with a live screen is left alone,
# and generate_sims skips indices whose metadata already exists.

set -e

REMOTE_DIR="${ABC_REJ_REMOTE_DIR:?set ABC_REJ_REMOTE_DIR to the project path on the remote hosts}"
RESULTS_REL="outputs/ABC_REJ"
SCRATCH_BASE="${ABC_REJ_SCRATCH:-/scratch/abc_rej}"
N_SIMS=3000

# (host, jobs). Counts target ~0.625 of physical cores, leaving capacity for
# other users of these shared machines: 64-core hosts 40, 48-core 30, 36-core 22.
HOSTS=(
    "wolverine    40"
    "nightcrawler 40"
    "nocturne     40"
    "vanisher     40"
    "forge        30"
    "cyclops      30"
    "shadowcat    30"
    "dazzler      30"
    "mystique     30"
    "polaris      22"
)

TOTAL_JOBS=0
for entry in "${HOSTS[@]}"; do
    set -- $entry
    TOTAL_JOBS=$(( TOTAL_JOBS + $2 ))
done
echo "Total jobs across hosts: ${TOTAL_JOBS}"

set -- ${HOSTS[0]}
CONTROLLER=$1
ssh "${CONTROLLER}" "mkdir -p ${REMOTE_DIR}/${RESULTS_REL}/cache/manifests \
                              ${REMOTE_DIR}/${RESULTS_REL}/observations \
                              ${REMOTE_DIR}/${RESULTS_REL}/distances \
                              ${REMOTE_DIR}/${RESULTS_REL}/posteriors"

# sampeuler is a project module, not a pip package, hence the PYTHONPATH.
echo "Checking deps on each host..."
for entry in "${HOSTS[@]}"; do
    set -- $entry
    host=$1
    if ! ssh -o ConnectTimeout=10 "${host}" \
        "source ${REMOTE_DIR}/.venv/bin/activate && \
         PYTHONPATH=${REMOTE_DIR}/scripts \
         python -c 'import numpy, scipy, sampeuler, pandas' 2>&1" \
        > /dev/null; then
        echo "  MISSING DEPS on ${host}: need numpy, scipy, pandas in ${REMOTE_DIR}/.venv"
        exit 1
    fi
done
echo "  deps OK on all hosts."

N_HOSTS=${#HOSTS[@]}
LAST_IDX=$(( N_HOSTS - 1 ))
cum=0
i=0
for entry in "${HOSTS[@]}"; do
    set -- $entry
    host=$1
    jobs=$2
    span=$(( N_SIMS * jobs / TOTAL_JOBS ))
    start=$cum
    end=$(( cum + span ))
    # Last host absorbs the rounding residual.
    if [[ "${i}" -eq "${LAST_IDX}" ]]; then
        end=${N_SIMS}
    fi
    cum=${end}
    i=$(( i + 1 ))

    cmd="cd ${REMOTE_DIR} && \
         source .venv/bin/activate && \
         python -u -m scripts.abc_rej.generate_sims \
             --n-sims ${N_SIMS} \
             --start ${start} --end ${end} \
             --jobs ${jobs} \
             --scratch ${SCRATCH_BASE} \
             2>&1 | tee -a ${RESULTS_REL}/gen_${host}.log; exec bash"

    # Resume guard: skip a host that is still working its slice.
    if ssh -o ConnectTimeout=10 "${host}" \
        "screen -ls 2>/dev/null | grep -q '[.]abc_rej_gen'" 2>/dev/null; then
        echo "  ${host}: chunk [${start}, ${end})  already running, skipped"
        continue
    fi
    if ssh -o ConnectTimeout=10 "${host}" "screen -dmS abc_rej_gen bash -c '${cmd}'"; then
        echo "  ${host}: chunk [${start}, ${end})  launched (jobs=${jobs})"
    else
        echo "  ${host}: chunk [${start}, ${end})  unreachable, skipped; re-run later to resume"
    fi
done
echo
echo "All chunks dispatched. Reattach: ssh <host> -t screen -r abc_rej_gen"
