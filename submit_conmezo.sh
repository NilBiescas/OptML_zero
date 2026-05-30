#!/bin/bash
# Submit ConMeZO replication on the dlab cluster (1xH100 preemptible).
# Paper-faithful: configs/conmezo/opt1.3b_sst2.yaml with the cone bug
# fix (cos/sin swap, paper Algorithm 1 / Eq. 6).
#
# Usage:
#   ./submit_conmezo.sh
#   CONFIG_FILE=configs/conmezo/opt1.3b_rte.yaml ./submit_conmezo.sh   # other task

set -euo pipefail

GASPAR="lichen"
METHOD="conmezo"
GPUS=1
NODE="${NODE:-h100}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="chengheng-${METHOD}-${TIMESTAMP}"
PROJECT="dlab-${GASPAR}"
IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
BRANCH="${BRANCH:-chenghengli}"
CONFIG_FILE_INNER="${CONFIG_FILE:-configs/conmezo/opt1.3b_sst2.yaml}"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ">>> Submitting ${JOB_NAME}"
echo "    Cluster project : ${PROJECT}"
echo "    Node pool       : ${NODE} (${GPUS} GPUs, preemptible)"
echo "    Branch          : ${BRANCH}"
echo "    Config          : ${CONFIG_FILE_INNER}"

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
  --environment WANDB_PROJECT="optml-zero-h100" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-}" \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment HF_USER="${HF_USER:-chenghengli}" \
  --environment RUN_NAME="${JOB_NAME}" \
  --environment CONFIG_FILE="${CONFIG_FILE_INNER}" \
  --environment BRANCH="${BRANCH}" \
  --command -- bash -c 'set -e
    apt-get update && apt-get install -y --no-install-recommends git
    git clone https://github.com/NilBiescas/OptML_zero.git
    cd OptML_zero
    git checkout "${BRANCH}"
    pip install --quiet transformers datasets accelerate wandb pyyaml huggingface_hub hf_transfer
    mkdir -p ~/.cache/huggingface && echo "${HF_TOKEN}" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
    accelerate launch --num_processes 1 train.py --config "${CONFIG_FILE}"
  '

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:    runai logs -f ${JOB_NAME} -p ${PROJECT}
Status:         runai describe job ${JOB_NAME} -p ${PROJECT}
List all jobs:  runai list jobs -p ${PROJECT}
Stop the job:   runai delete job ${JOB_NAME} -p ${PROJECT}
WandB project:  https://wandb.ai/${WANDB_ENTITY:-chenghengli-epfl}/optml-zero-h100

EOF
