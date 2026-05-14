#!/bin/bash
# Preemptible 4-GPU job that runs Sparse-MeZO training on EPFL RCP.
#
# Usage:
#   ./submit_sparse_mezo.sh
#
# Before first run, edit the placeholders marked TODO below to point at YOUR
# EPFL GASPAR account / RCP project / GitHub fork.

set -euo pipefail

# TODO: replace with your EPFL GASPAR (the username part of your @epfl.ch email).
GASPAR="pilligua"

# TODO: replace with your RCP project (e.g. "cs-439-students", "vilab-<gaspar>", ...).
# Ask your TA or check `runai list projects` on the RCP login node.
PROJECT="cvlab-pilligua"

# SSH clone URL of your fork. Pod authenticates with the deploy key stored
# in .env as DEPLOY_KEY_B64 (the public half is configured on the repo).
GIT_REPO_SSH="git@github.com:mpilligua/OptML_zero.git"
GIT_REPO_DIR="OptML_zero"

# Container image — keep unless your TA gave you a different one.
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

GPUS=4
NODE="${NODE:-a100-40g}"
JOB_NAME="${GASPAR}-sparse-mezo-$(date +%H%M%S)"
CONFIG_FILE="${CONFIG_FILE:-config_sparse_mezo.yaml}"

# Source environment variables (.env file is gitignored — see .env.example)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ">>> Submitting ${JOB_NAME} (${GPUS} GPUs, Sparse-MeZO, config=${CONFIG_FILE})"

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
  --command -- bash -c 'set -e; mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "$DEPLOY_KEY_B64" | base64 -d > ~/.ssh/id_ed25519 && chmod 600 ~/.ssh/id_ed25519 && ssh-keyscan -t ed25519 github.com 2>/dev/null >> ~/.ssh/known_hosts && GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" git clone "$GIT_REPO_SSH" && cd '"${GIT_REPO_DIR}"' && bash run_sparse_mezo.sh'

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:    runai logs -f ${JOB_NAME} -p ${PROJECT}
Status:         runai describe job ${JOB_NAME} -p ${PROJECT}
List jobs:      runai list jobs
Stop the job:   runai delete job ${JOB_NAME} -p ${PROJECT}

EOF
