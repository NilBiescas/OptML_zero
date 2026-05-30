#!/bin/bash
set -euo pipefail

# Launcher for FZOO (arXiv:2506.09034). Defaults to OPT-1.3B / SST-2.
# Override with: CONFIG_FILE=configs/fzoo/opt13b_sst2.yaml ./run_fzoo.sh

CONFIG_FILE="${CONFIG_FILE:-configs/fzoo/opt1.3b_sst2.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

mkdir -p hf_cache wandb
export HF_HOME="$(pwd)/hf_cache"
export WANDB_DIR="$(pwd)/wandb"

ln -sf "$(command -v python3)" /usr/local/bin/python || true
git config --global --add safe.directory "$(pwd)"

echo ">>> git pull in $(pwd)"
git pull --ff-only || echo "git pull failed — continuing with current files"

echo ">>> Installing dependencies"
pip install --upgrade transformers datasets accelerate evaluate pyyaml wandb huggingface_hub

echo ">>> Launching FZOO with config=${CONFIG_FILE}, num_processes=${NUM_PROCESSES}"
accelerate launch --num_processes "${NUM_PROCESSES}" train.py --config "${CONFIG_FILE}"

echo ">>> FZOO run complete."
