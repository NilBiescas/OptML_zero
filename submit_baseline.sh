#!/bin/bash
# Submits a baseline (first-order) training job to RCP.
#
# Usage:
#   CONFIG_FILE=config_adam.yaml NAME_TAG=adamw ./submit_baseline.sh
#   CONFIG_FILE=config_sgd.yaml  NAME_TAG=sgd   ./submit_baseline.sh

set -euo pipefail

GASPAR="pilligua"
PROJECT="cvlab-pilligua"
GIT_REPO_SSH="git@github.com:mpilligua/OptML_zero.git"
GIT_REPO_DIR="OptML_zero"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

GPUS=4
NODE="${NODE:-a100-40g}"
CONFIG_FILE="${CONFIG_FILE:-config_adam.yaml}"
NAME_TAG="${NAME_TAG:-baseline}"
JOB_NAME="${GASPAR}-${NAME_TAG}-$(date +%H%M%S)"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ">>> Submitting ${JOB_NAME} (${GPUS} GPUs, ${NAME_TAG}, config=${CONFIG_FILE})"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment WANDB_ENTITY="nilbiescas3" \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment RUN_NAME="${JOB_NAME}" \
  --environment CHECKPOINT_DIR="/scratch/${JOB_NAME}" \
  --environment DEPLOY_KEY_B64="${DEPLOY_KEY_B64}" \
  --environment GIT_REPO_SSH="${GIT_REPO_SSH}" \
  --environment CONFIG_FILE="${CONFIG_FILE}" \
  --command -- bash -c 'set -e; mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "$DEPLOY_KEY_B64" | base64 -d > ~/.ssh/id_ed25519 && chmod 600 ~/.ssh/id_ed25519 && ssh-keyscan -t ed25519 github.com 2>/dev/null >> ~/.ssh/known_hosts && GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone "$GIT_REPO_SSH" && cd '"${GIT_REPO_DIR}"' && bash run_baseline.sh'

cat <<EOF

>>> Job submitted: ${JOB_NAME}
Stream logs:    runai logs -f ${JOB_NAME} -p ${PROJECT}
Status:         runai describe job ${JOB_NAME} -p ${PROJECT}
Stop the job:   runai delete job ${JOB_NAME} -p ${PROJECT}

EOF
