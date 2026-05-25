#!/bin/bash
# Preemptible job that runs the ORIGINAL authors' LOZO code for SNLI, MNLI, and RTE.
#
# Usage:
#   ./submit.sh

set -euo pipefail

GASPAR="nil"

GPUS=1 # The authors' codebase typically uses 1 GPU by default
NODE="${NODE:-a100-40g}"
PROJECT="vilab-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

# Source environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f ../../.env ]; then
    export $(grep -v '^#' ../../.env | xargs)
fi

echo ">>> Submitting 3 preemptible training jobs using the authors' original code for SNLI, MNLI, and RTE"

for TASK in SNLI MNLI RTE; do
  JOB_NAME="${GASPAR}-orig-lozo-$(echo $TASK | tr '[:upper:]' '[:lower:]')-$(date +%H%M%S)"

  # The command does the following:
  # 1. Clones the repository
  # 2. Downloads the datasets into LOZO/data
  # 3. Generates the 16-shot splits
  # 4. Runs the original lozo.sh training script for the specific task
  COMMAND_STR="git clone https://\${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && cd OptML_zero/LOZO/data && bash download_dataset.sh && cd ../medium_models && pip install --upgrade transformers datasets evaluate pandas scipy scikit-learn && python tools/generate_k_shot_data.py --mode k-shot-1k-test --k 16 && TASK=${TASK} K=16 SEED=42 BS=64 LR=1e-7 EPS=1e-3 MODEL=roberta-large RANK=4 STEP_INTERVAL=100 bash lozo.sh"

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
    --command -- bash -c "${COMMAND_STR}"

  echo ">>> Job submitted: ${JOB_NAME}"
  sleep 1 # To ensure unique timestamp in JOB_NAME
done

cat <<EOF

>>> All 3 jobs submitted successfully!

To monitor your jobs, you can use:
List jobs:      runai list jobs
Stream logs:    runai logs -f <job-name> -p ${PROJECT}
EOF
