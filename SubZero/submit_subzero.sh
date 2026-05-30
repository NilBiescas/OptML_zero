#!/bin/bash
# Preemptible 1-GPU job that runs SubZero training on CB.

set -euo pipefail

GASPAR="nil"

GPUS=1
NODE="${NODE:-a100-40g}"
PROJECT="vilab-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

# Source environment variables
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
elif [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ">>> Submitting preemptible training job for SubZero on SST2 (${GPUS} GPUs)"

for DATASET in SST2; do
  JOB_NAME="${GASPAR}-subzero-sst2-$(date +%H%M%S)"

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
    --environment GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
    --command -- bash -c "ln -sf /usr/bin/python3 /usr/bin/python && git clone -b nil_branch https://\${GITHUB_TOKEN}@github.com/NilBiescas/OptML_zero.git && cd OptML_zero/SubZero/large_models && pip install -r ../requirements.txt && CUDA_VISIBLE_DEVICES=0 python run.py --task_name=${DATASET} --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-${DATASET}-ft-subzero --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --eval_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=500 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1025 --num_dev=512 --train_as_classification --perturbation_mode=two_side --trainer=subzero_sgd --train_set_seed=0 --lr_scheduler_type=constant --learning_rate=1e-7 --zo_eps=1e-3 --weight_decay=0 --gauss_rank=24 --update_interval=1000"

  echo ">>> Job submitted: ${JOB_NAME}"
done

cat <<EOF

>>> All jobs submitted successfully!

To monitor your jobs, you can use:
List jobs:      runai list jobs
Stream logs:    runai logs -f <job-name> -p ${PROJECT}
EOF
