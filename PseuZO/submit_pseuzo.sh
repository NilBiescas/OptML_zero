#!/bin/bash
# Preemptible 1-GPU job that runs PseuZO training on CB.

set -euo pipefail

GASPAR="nil"

GPUS=1
NODE="${NODE:-a100-40g}"
PROJECT="dhlab-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

# Source environment variables
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
elif [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ">>> Submitting preemptible training job for PseuZO on SST2 and RTE (${GPUS} GPUs)"

for DATASET in RTE SST2; do
  DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
  JOB_NAME="${GASPAR}-pseuzo-${DATASET_LOWER}-$(date +%H%M%S)"

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
    --environment WANDB_NAME="${JOB_NAME}" \
    --environment GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
    --command -- bash -c "ln -sf /usr/bin/python3 /usr/bin/python && git clone -b nil_branch https://\${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && cd OptML_zero/PseuZO && pip install -r requirements.txt && TRAIN=1536 DEV=512 EVAL=512 CUDA_VISIBLE_DEVICES=0 TRAINER=pzo MODEL=facebook/opt-1.3b TASK=${DATASET} MODE=ft LR=1e-7 EPS=1e-3 BS=16 STEPS=20000 EVAL_STEPS=500 bash mezo.sh"

  echo ">>> Job submitted: ${JOB_NAME}"
done

cat <<EOF

>>> All jobs submitted successfully!

To monitor your jobs, you can use:
List jobs:      runai list jobs
Stream logs:    runai logs -f <job-name> -p ${PROJECT}
EOF
