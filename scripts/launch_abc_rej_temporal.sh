#!/usr/bin/env bash
# Build the temporal (n-frame trajectory) cache across hosts.
#
# Each host featurises the sims it owns, even-sampling N frames per sim across
# the whole trajectory and writing traj_<idx>.npz to the shared cache. Raw
# frames are read from each host's local scratch; the single-frame pool cache
# is left untouched in its own directory.
#
# Usage:  ABC_REJ_REMOTE_DIR=/path/to/project bash scripts/launch_abc_rej_temporal.sh
#         ABC_REJ_TEMPORAL_N=15 bash scripts/launch_abc_rej_temporal.sh

set -e

REMOTE_DIR="${ABC_REJ_REMOTE_DIR:?set ABC_REJ_REMOTE_DIR to the project path on the remote hosts}"
RESULTS_REL="outputs/ABC_REJ"
OUT_REL="${RESULTS_REL}/cache_temporal"
N_FRAMES="${ABC_REJ_TEMPORAL_N:-10}"

# Job counts kept below the Stage A2 values to hold loadtxt memory down.
HOSTS=(
    "wolverine    30"
    "nightcrawler 30"
    "nocturne     20"
    "forge        25"
    "cyclops      25"
    "shadowcat    25"
    "dazzler      25"
    "mystique     25"
    "vanisher     25"
    "polaris      20"
)

echo "Checking deps on each host..."
for entry in "${HOSTS[@]}"; do
    set -- $entry
    host=$1
    if ! ssh -o ConnectTimeout=10 "${host}" \
        "source ${REMOTE_DIR}/.venv/bin/activate && \
         PYTHONPATH=${REMOTE_DIR}/scripts \
         python -c 'import numpy, scipy, sampeuler' 2>&1" \
        > /dev/null; then
        echo "  MISSING DEPS on ${host}: need numpy, scipy in the venv"
        exit 1
    fi
done
echo "  deps OK on all hosts."

set -- ${HOSTS[0]}
ssh "$1" "mkdir -p ${REMOTE_DIR}/${OUT_REL}"

echo
echo "=== Dispatching temporal featurise workers (n_frames=${N_FRAMES}) ==="
for entry in "${HOSTS[@]}"; do
    set -- $entry
    host=$1
    jobs=$2
    cmd="cd ${REMOTE_DIR} && \
         source .venv/bin/activate && \
         python -u -m scripts.abc_rej.featurise_temporal \
             --n-frames ${N_FRAMES} --out-dir ${OUT_REL} --jobs ${jobs} \
             2>&1 | tee ${RESULTS_REL}/temporal_${host}.log; exec bash"
    ssh "${host}" "screen -dmS abc_rej_temporal bash -c '${cmd}'"
    echo "  ${host}: temporal featurise launched (jobs=${jobs})"
done

echo
echo "All temporal workers dispatched."
echo "Next (on a CUDA host, per observation index):"
echo "  python scripts/abc_rej_distance_cuda.py --mode temporal --fakexp-idx <i> \\"
echo "      --cache-dir ${OUT_REL} --cache-glob 'traj_[0-9]*.npz' \\"
echo "      --n-frames ${N_FRAMES} --time-sampling relative --temporal-agg sum"
