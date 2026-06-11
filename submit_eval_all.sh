#!/bin/bash
set -euo pipefail

DATASET=${1:-multirc}
GASPAR="nil"

GPUS=1
NODE="${NODE:-h100}"
PROJECT="${PROJECT:-vilab-${GASPAR}}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

MODELS=("eval_qwen_0_5b" "eval_qwen_0_8b" "eval_qwen_2b")

for CFG in "${MODELS[@]}"; do
    JOB_NAME="${GASPAR}-${CFG}-${DATASET}-$(date +%H%M%S)"
    echo ">>> Submitting zero-shot evaluation for ${CFG} on ${DATASET}..."
    
    runai submit \
      --name "${JOB_NAME}" \
      -p "${PROJECT}" \
      --image "${IMAGE}" \
      --gpu "${GPUS}" \
      --large-shm \
      --node-pools "${NODE}" \
      --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
      --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
      --environment WANDB_ENTITY="pilligua" \
      --environment HF_TOKEN="${HF_TOKEN:-}" \
      --environment RUN_OWNER="${GASPAR}" \
      --environment RUN_NAME="${JOB_NAME}" \
      --environment WANDB_NAME="${JOB_NAME}" \
      --environment GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
      --command -- bash -c "ln -sf /usr/bin/python3 /usr/bin/python && \
git clone -b nil_branch https://\${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && \
cd OptML_zero && \
pip install \"transformers>=5.2.0\" \"huggingface_hub>=0.30\" \"datasets>=5.0.0\" && \
python train.py --config configs/${CFG}.yaml --task ${DATASET}"

    sleep 2
done

echo ">>> All zero-shot evaluation jobs submitted successfully!"
