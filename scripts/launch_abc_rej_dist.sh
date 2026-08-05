#!/usr/bin/env bash
# Stage B: fan per-observation distance computation out across hosts.
#
# One observation per host. Edit ASSIGNMENTS for the observation indices you
# want scored; the default covers the three synthetic targets.
#
# Usage:  ABC_REJ_REMOTE_DIR=/path/to/project bash scripts/launch_abc_rej_dist.sh
#         ABC_REJ_DISTANCE=l2 bash scripts/launch_abc_rej_dist.sh

set -e

REMOTE_DIR="${ABC_REJ_REMOTE_DIR:?set ABC_REJ_REMOTE_DIR to the project path on the remote hosts}"
RESULTS_REL="outputs/ABC_REJ"

# 'hungarian' is the Wasserstein-1 distance between raw ECT curves; 'l2' is
# the cheaper batched Euclidean baseline on vectorised SampEuler images.
DISTANCE="${ABC_REJ_DISTANCE:-hungarian}"

# (host, observation index).
ASSIGNMENTS=(
    "wolverine    0"
    "nightcrawler 1"
    "nocturne     2"
)

set -- ${ASSIGNMENTS[0]}
ssh "$1" "mkdir -p ${REMOTE_DIR}/${RESULTS_REL}/distances \
                   ${REMOTE_DIR}/${RESULTS_REL}/observations"

for entry in "${ASSIGNMENTS[@]}"; do
    set -- $entry
    host=$1
    obs=$2

    cmd="cd ${REMOTE_DIR} && \
         source .venv/bin/activate && \
         python -u -m scripts.abc_rej.compute_distances \
             --obs-index ${obs} \
             --distance ${DISTANCE} \
             2>&1 | tee -a ${RESULTS_REL}/dist_obs${obs}_${DISTANCE}.log; exec bash"

    ssh "${host}" "screen -dmS abc_rej_dist_obs${obs}_${DISTANCE} bash -c '${cmd}'"
    echo "  ${host}: distances for observation ${obs}  (distance=${DISTANCE})"
done
echo
echo "Distance jobs dispatched. Reattach:"
echo "    ssh <host> -t screen -r abc_rej_dist_obs<idx>_${DISTANCE}"
echo "Then plot:"
echo "    python -m scripts.abc_rej.plot_posterior_3p --help"
