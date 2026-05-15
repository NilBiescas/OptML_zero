#!/bin/bash
set -euo pipefail

mkdir -p hf_cache wandb
export HF_HOME="$(pwd)/hf_cache"
export WANDB_DIR="$(pwd)/wandb"

ln -sf "$(command -v python3)" /usr/local/bin/python || true
git config --global --add safe.directory "$(pwd)"

echo ">>> git pull in $(pwd)"
git pull --ff-only || echo "git pull failed — continuing with current files"

echo ">>> Installing dependencies"
pip install --upgrade transformers datasets accelerate evaluate pyyaml wandb huggingface_hub

echo ">>> Starting MeZO smoke test (3 epochs, config_mezo_smoke.yaml)"
accelerate launch --num_processes 1 train.py --config config_mezo_smoke.yaml

echo ">>> MeZO smoke test complete."
