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
# runai job names must be alphanumeric + '-' only: sanitize underscores
# (e.g. config stem "zo_muon" -> name token "zo-muon"). CONFIG_FILE still uses
# the original METHOD so it resolves configs/zo_muon.yaml.
METHOD_SAFE="${METHOD//_/-}"
JOB_NAME="chengheng-${METHOD_SAFE}-${TASK}-${TIMESTAMP}"
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
  --existing-pvc claimname=dlab-scratch,path=/scratch \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-pilligua}" \
  --environment WANDB_PROJECT="Zero-Order-Opt" \
  --environment RUN_OWNER="chengheng" \
  --environment RUN_TAGS="${RUN_TAGS:-}" \
  --environment METHOD="${METHOD}" \
  --environment TASK="${TASK}" \
  --environment BRANCH="${BRANCH}" \
  --command -- bash -c 'set -e
    apt-get update && apt-get install -y --no-install-recommends git
    git clone https://github.com/NilBiescas/OptML_zero.git
    cd OptML_zero
    git checkout "${BRANCH}"
    # Upgrade torch to >=2.7 (cu126) FIRST: the image ships torch 2.6.0, but the
    # current transformers (needed for Qwen3.5 / Qwen3_5ForCausalLM) imports
    # torch.float8_e8m0fnu, which only exists in torch>=2.7. Without this the
    # run dies at model load with AttributeError: module torch has no attribute
    # float8_e8m0fnu / ModuleNotFoundError: Qwen3_5ForCausalLM.
    pip install --quiet "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
    pip install --quiet -r requirements.txt hf_transfer
    # Checkpoints (best + rolling last) live on the durable dlab-scratch PVC at
    # /scratch/chengheng/zo-ckpts/<owner>-<method>-<task>/. The harness
    # auto-resumes from last/ if present, continuing the SAME wandb run and the
    # last step — so a preempted + relaunched job picks up where it left off.
    mkdir -p /scratch/chengheng/zo-ckpts
    python train.py --config "configs/${METHOD}.yaml" --task "${TASK}" --owner chengheng \
      --ckpt-dir /scratch/chengheng/zo-ckpts
  '

cat <<EOF

>>> Job submitted: ${JOB_NAME}
Stream logs: runai logs -f ${JOB_NAME} -p ${PROJECT}
Delete job:  runai delete job ${JOB_NAME} -p ${PROJECT}
WandB:       project Zero-Order-Opt (entity = owner of WANDB_API_KEY, or \$WANDB_ENTITY)
EOF
