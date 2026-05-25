#!/bin/bash
# Preemptible 4-GPU job that runs the LOZO training.
#
# Usage:
#   ./submit_lozo.sh

set -euo pipefail

GASPAR="nil"

GPUS=1
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

echo ">>> Submitting 3 preemptible training jobs for SNLI, MNLI, and RTE (${GPUS} GPUs each)"

for DATASET in snli mnli rte; do
  JOB_NAME="${GASPAR}-lozo-${DATASET}-$(date +%H%M%S)"

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
    --command -- bash -c "git clone https://\${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && cd OptML_zero/LOZO/data && bash download_dataset.sh && cd ../medium_models && pip install --upgrade transformers datasets evaluate pandas scipy scikit-learn && python tools/generate_k_shot_data.py --mode k-shot-1k-test --k 16 && cd ../../classificationhead && accelerate launch --num_processes 4 train.py --config config_${DATASET}.yaml"

  echo ">>> Job submitted: ${JOB_NAME}"
done

cat <<EOF

>>> All 3 jobs submitted successfully!

To monitor your jobs, you can use:
List jobs:      runai list jobs
Stream logs:    runai logs -f <job-name> -p ${PROJECT}
EOF
