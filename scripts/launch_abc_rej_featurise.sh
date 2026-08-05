#!/usr/bin/env bash
# Stage A2: pool-sample the cache, then featurise the sampled frames.
#
# Phase 1 runs --plan on one host, building the global frame pool from the
# Stage A1 metadata and writing per-host work lists to shared storage.
# Phase 2 launches a --featurise screen on each host for its own list.
#
# Usage:  ABC_REJ_REMOTE_DIR=/path/to/project bash scripts/launch_abc_rej_featurise.sh
#         ABC_REJ_N_POOL=60000 bash scripts/launch_abc_rej_featurise.sh
#
# Re-running is safe: --plan rewrites the work lists, and --featurise skips
# sims whose cached frames already match.

set -e

REMOTE_DIR="${ABC_REJ_REMOTE_DIR:?set ABC_REJ_REMOTE_DIR to the project path on the remote hosts}"
RESULTS_REL="outputs/ABC_REJ"

PLAN_EXTRA=""
[[ -n "${ABC_REJ_N_POOL:-}" ]] && PLAN_EXTRA="--n-pool-samples ${ABC_REJ_N_POOL}"

HOSTS=(
    "wolverine    46"
    "nightcrawler 41"
    "nocturne     25"
    "forge        31"
    "cyclops      31"
    "shadowcat    31"
    "dazzler      31"
    "mystique     31"
    "vanisher     30"
    "polaris      23"
)

echo "Checking deps on each host..."
for entry in "${HOSTS[@]}"; do
    set -- $entry
    host=$1
    if ! ssh -o ConnectTimeout=10 "${host}" \
        "source ${REMOTE_DIR}/.venv/bin/activate && \
         PYTHONPATH=${REMOTE_DIR}/scripts \
         python -c 'import numpy, scipy, sampeuler, pandas' 2>&1" \
        > /dev/null; then
        echo "  MISSING DEPS on ${host}: need numpy, scipy, pandas in the venv"
        exit 1
    fi
done
echo "  deps OK on all hosts."

set -- ${HOSTS[0]}
CONTROLLER_HOST=$1

echo
echo "=== Stage A2 planner (on ${CONTROLLER_HOST}) ==="
ssh "${CONTROLLER_HOST}" "cd ${REMOTE_DIR} && \
    source .venv/bin/activate && \
    python -u -m scripts.abc_rej.sample_and_featurise --plan ${PLAN_EXTRA} \
        2>&1 | tee ${RESULTS_REL}/plan.log"

echo
echo "=== Dispatching Stage A2 workers ==="
for entry in "${HOSTS[@]}"; do
    set -- $entry
    host=$1
    jobs=$2
    cmd="cd ${REMOTE_DIR} && \
         source .venv/bin/activate && \
         python -u -m scripts.abc_rej.sample_and_featurise --featurise \
             --jobs ${jobs} \
             2>&1 | tee -a ${RESULTS_REL}/feat_${host}.log; exec bash"
    ssh "${host}" "screen -dmS abc_rej_feat bash -c '${cmd}'"
    echo "  ${host}: featurise screen launched (jobs=${jobs})"
done

echo
echo "All featurise workers dispatched."
echo "Reattach:  ssh <host> -t screen -r abc_rej_feat"
echo "Next:      bash scripts/launch_abc_rej_dist.sh"
