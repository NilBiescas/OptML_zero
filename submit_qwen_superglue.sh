#!/bin/bash
# Submit one qwen-superglue harness job (1xH100 preemptible) on dlab-lichen.
#
# Usage:
#   METHOD=conmezo TASK=copa ./submit_qwen_superglue.sh
#   METHOD=zo_muon TASK=multirc ./submit_qwen_superglue.sh
#
# METHOD is the config stem under configs/ (conmezo|fzoo|dizo|zo_muon|...).
# TASK is the SuperGLUE task wired in tasks.py (copa|multirc).
#
# WandB: the harness logs every metric to project "Zero-Order-Opt" with no
# hardcoded entity, so runs land in whichever account owns WANDB_API_KEY.
# To collect all team runs in ONE project we use the SHARED team key (Maria's
# mpilligua entity). Provide it via .env or the environment:
#     WANDB_API_KEY=<maria's key>
#     WANDB_ENTITY=mpilligua        # optional; only if the project lives under her entity
# The key is passed through to the pod and never echoed.

set -euo pipefail

GASPAR="lichen"
METHOD="${METHOD:-conmezo}"
TASK="${TASK:-copa}"
GPUS=1
NODE="${NODE:-h100}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="chengheng-${METHOD}-${TASK}-${TIMESTAMP}"
PROJECT="dlab-${GASPAR}"
IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
BRANCH="${BRANCH:-cheng-zo-optimizers}"
CONFIG_FILE_INNER="configs/${METHOD}.yaml"

# Load shared WandB credentials (Maria's) + any HF token from .env if present.
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is empty. Put Maria's team key in .env or export it." >&2
    echo "       (runs must land in the shared mpilligua 'Zero-Order-Opt' project)" >&2
    exit 1
fi

echo ">>> Submitting ${JOB_NAME}  (config=${CONFIG_FILE_INNER}, task=${TASK})"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --preemptible \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-}" \
  --environment WANDB_PROJECT="Zero-Order-Opt" \
  --environment RUN_OWNER="cheng" \
  --environment METHOD="${METHOD}" \
  --environment TASK="${TASK}" \
  --environment BRANCH="${BRANCH}" \
  --command -- bash -c 'set -e
    apt-get update && apt-get install -y --no-install-recommends git
    git clone https://github.com/NilBiescas/OptML_zero.git
    cd OptML_zero
    git checkout "${BRANCH}"
    pip install --quiet -r requirements.txt hf_transfer
    python train.py --config "configs/${METHOD}.yaml" --task "${TASK}" --owner cheng
  '

cat <<EOF

>>> Job submitted: ${JOB_NAME}
Stream logs: runai logs -f ${JOB_NAME} -p ${PROJECT}
Delete job:  runai delete job ${JOB_NAME} -p ${PROJECT}
WandB:       project Zero-Order-Opt (entity = owner of WANDB_API_KEY, or \$WANDB_ENTITY)
EOF
