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
# Model id (HF). Default keeps the original single-model behaviour; pass
# MODEL=Qwen/Qwen2.5-0.5B for the second model. Forwarded to train.py as --model.
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
# Optional run-suffix (e.g. "short") -> distinct wandb run + ckpt + tag.
SUFFIX="${SUFFIX:-}"
GPUS=1
NODE="${NODE:-h100}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
# runai job names must be alphanumeric + '-' only: sanitize underscores
# (e.g. config stem "zo_muon" -> name token "zo-muon"). CONFIG_FILE still uses
# the original METHOD so it resolves configs/zo_muon.yaml.
METHOD_SAFE="${METHOD//_/-}"
# Model token for the job name: strip org, lowercase, dots -> dashes
# (e.g. "Qwen/Qwen2.5-0.5B" -> "qwen2-5-0-5b") so two models get distinct jobs.
MODEL_SAFE="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr '.' '-')"
JOB_NAME="chengheng-${METHOD_SAFE}-${TASK}-${MODEL_SAFE}${SUFFIX:+-${SUFFIX}}-${TIMESTAMP}"
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
  --existing-pvc claimname=dlab-scratch,path=/dlab-scratch \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-pilligua}" \
  --environment WANDB_PROJECT="Zero-Order-Opt" \
  --environment WANDB_MODE="${WANDB_MODE:-online}" \
  --environment RUN_OWNER="chengheng" \
  --environment RUN_TAGS="${RUN_TAGS:-}" \
  --environment METHOD="${METHOD}" \
  --environment TASK="${TASK}" \
  --environment MODEL="${MODEL}" \
  --environment SUFFIX="${SUFFIX}" \
  --environment BRANCH="${BRANCH}" \
  --environment LR="${LR:-}" \
  --environment MAX_STEPS="${MAX_STEPS:-}" \
  --environment EVAL_STEPS="${EVAL_STEPS:-}" \
  --environment EXTRA_ARGS="${EXTRA_ARGS:-}" \
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
    # Checkpoints go to the dlab-scratch PVC (claimname=dlab-scratch, 100Ti,
    # mounted at /dlab-scratch) -- NOT the home PVC, which has only a ~100GB
    # user quota that filled up with fp32 Qwen checkpoints (3.2GB each) and
    # truncated a checkpoint write mid-flight, leaving an empty training_meta
    # that crash-looped every auto-resume. dlab-scratch has terabytes free.
    # Run training as lichen (uid 316680): NFS root_squash maps uid 0 -> nobody,
    # but uid 316680 maps to lichen, who OWNS
    # /mnt/dlab/scratch/dlabscratch1/chengheng/zo-optim and can write there.
    # Probe the in-pod mount path (PVC root may or may not include the
    # dlabscratch1/ prefix); fall back to ephemeral local disk so the run NEVER
    # dies at the checkpoint step. The harness auto-resumes from
    # <ckpt>/<owner>-<method>-<task>/last/ (or best/) if a VALID meta exists,
    # continuing the SAME wandb run + last step. HF_HOME -> /tmp keeps the
    # model download off any PVC quota.
    set +e
    groupadd -g 30204 lichen 2>/dev/null || true
    # /dlab-scratch/dlabscratch1 is mode drwxrws--T owned by group 60220
    # (dlab_AppGrpU): only members of gid 60220 can traverse + write. On the
    # login node lichen has this supplementary group, but a fresh useradd in the
    # pod would only give the primary group (30204), so lichen must be added to
    # 60220 explicitly or every dlab-scratch write is Permission-denied.
    groupadd -g 60220 dlabgrp 2>/dev/null || true
    id -u lichen >/dev/null 2>&1 || useradd -u 316680 -g 30204 -G 60220 -d /tmp/lhome -M -s /bin/bash lichen
    usermod -aG 60220 lichen 2>/dev/null || true
    chown -R 316680:30204 "$(pwd)" 2>/dev/null
    CKPT=""
    for d in /dlab-scratch/dlabscratch1/chengheng/zo-optim /dlab-scratch/chengheng/zo-optim /dlab-scratch/zo-optim; do
      su -p lichen -c "mkdir -p $d" 2>/dev/null && su -p lichen -c "test -w $d" && CKPT=$d && break
    done
    if [ -z "$CKPT" ]; then CKPT=/workspace/zo-ckpts; mkdir -p "$CKPT"; chown -R 316680:30204 "$CKPT"; echo "[ckpt-dir] WARNING: dlab-scratch not writable; using EPHEMERAL $CKPT (no preemption survival)"; fi
    echo "[ckpt-dir] using: $CKPT (run as lichen)"
    set -e
    mkdir -p /tmp/lhome && chown 316680:30204 /tmp/lhome
    su -p lichen -c "export HOME=/tmp/lhome PATH=/opt/conda/bin:/usr/bin:/bin HF_HOME=/tmp/hf HF_HUB_ENABLE_HF_TRANSFER=1 && cd $(pwd) && /opt/conda/bin/python train.py --config configs/${METHOD}.yaml --task ${TASK} --owner chengheng --ckpt-dir $CKPT ${MODEL:+--model $MODEL} ${LR:+--lr $LR} ${MAX_STEPS:+--max-steps $MAX_STEPS} ${EVAL_STEPS:+--eval-steps $EVAL_STEPS} ${SUFFIX:+--run-suffix $SUFFIX} ${EXTRA_ARGS:-}"
  '

cat <<EOF

>>> Job submitted: ${JOB_NAME}
Stream logs: runai logs -f ${JOB_NAME} -p ${PROJECT}
Delete job:  runai delete job ${JOB_NAME} -p ${PROJECT}
WandB:       project Zero-Order-Opt (entity = owner of WANDB_API_KEY, or \$WANDB_ENTITY)
EOF
