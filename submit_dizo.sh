#!/bin/bash
# Submit DiZO training to the dlab cluster on an H100.
# Job name encodes: user / method / timestamp for easy WandB filtering.
#
# Usage:
#   ./submit_dizo.sh

set -euo pipefail

GASPAR="lichen"
METHOD="dizo"

GPUS=1
NODE="${NODE:-h100}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="chengheng-${METHOD}-${TIMESTAMP}"
PROJECT="dlab-${GASPAR}"
IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"

# Source environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ">>> Submitting ${JOB_NAME}"
echo "    Cluster project : ${PROJECT}"
echo "    Node pool       : ${NODE} (${GPUS} GPUs)"
echo "    Method          : ${METHOD}"
echo "    Timestamp       : ${TIMESTAMP}"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment RUN_NAME="${JOB_NAME}" \
  --command -- bash -c 'git clone https://github.com/NilBiescas/OptML_zero.git && cd OptML_zero && bash run_dizo.sh'

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:    runai logs -f ${JOB_NAME} -p ${PROJECT}
Status:         runai describe job ${JOB_NAME} -p ${PROJECT}
List all jobs:  runai list jobs -p ${PROJECT}
Stop the job:   runai delete job ${JOB_NAME} -p ${PROJECT}
WandB run:      https://wandb.ai/chengheng/lozo-generative-training

EOF
