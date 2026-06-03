#!/bin/bash
# Preemptible 1-GPU job that runs the consolidated train.py on CB.
# Targets an H100 GPU for maximum performance.

set -euo pipefail

GASPAR="nil"

GPUS=1
NODE="${NODE:-h100}"
PROJECT="dhlab-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

# Source environment variables for WandB/HF tokens
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default to SubZero if no argument provided
CONFIG_NAME="${1:-subzero}"
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

echo ">>> Submitting preemptible training job for ${CONFIG_NAME} on H100"

JOB_NAME="${GASPAR}-${CONFIG_NAME}-h100-$(date +%H%M%S)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment RUN_NAME="${JOB_NAME}" \
  --environment WANDB_ENTITY="pilligua" \
  --environment WANDB_PROJECT="Zero-Order-Opt" \
  --environment GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  --command -- bash -c "
    ln -sf /usr/bin/python3 /usr/bin/python && \
    git clone -b nil_branch https://\${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && \
    cd OptML_zero && \
    pip install -r SubZero/requirements.txt && \
    pip install -r PseuZO/requirements.txt && \
    CUDA_VISIBLE_DEVICES=0 python train.py --config ${CONFIG_FILE}
  "

echo ">>> Job submitted: ${JOB_NAME}"
echo ">>> Using config: ${CONFIG_FILE}"

cat <<EOF

To monitor your job:
List jobs:      runai list jobs
Stream logs:    runai logs -f ${JOB_NAME} -p ${PROJECT}
EOF
