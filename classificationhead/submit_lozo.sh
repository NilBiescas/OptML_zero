#!/bin/bash
# Preemptible 4-GPU job that runs the LOZO training.
#
# Usage:
#   ./submit_lozo.sh

set -euo pipefail

GASPAR="nil"

GPUS=4
NODE="${NODE:-a100-40g}"
JOB_NAME="${GASPAR}-lozo-$(date +%H%M%S)"
PROJECT="vilab-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
# Source environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

echo ">>> Submitting ${JOB_NAME} (${GPUS} GPUs, preemptible training job)"

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
  --environment GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  --command -- bash -c 'git clone https://${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && cd OptML_zero/classificationhead && bash run_lozo.sh'

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:    runai logs -f ${JOB_NAME} -p ${PROJECT}
Status:         runai describe job ${JOB_NAME} -p ${PROJECT}
List jobs:      runai list jobs
Stop the job:   runai delete job ${JOB_NAME} -p ${PROJECT}

EOF
