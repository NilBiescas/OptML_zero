#!/bin/bash
# Runs INSIDE the pod — DiZO training on Qwen3.5-0.8B / banking77.
# Called by submit_dizo.sh via runai submit --command.

set -euo pipefail

mkdir -p runs/dizo hf_cache wandb
export HF_HOME="$(pwd)/hf_cache"
export WANDB_DIR="$(pwd)/wandb"

ln -sf "$(command -v python3)" /usr/local/bin/python || true

git config --global --add safe.directory "$(pwd)"

mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keyscan -t ed25519 github.com 2>/dev/null >> ~/.ssh/known_hosts
sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts

if [[ -f ~/.ssh/id_ed25519 ]]; then
    chmod 600 ~/.ssh/id_ed25519
    cat > ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF
    chmod 600 ~/.ssh/config
    echo ">>> SSH configured"
fi

echo ">>> git pull in $(pwd)"
git pull --ff-only || echo "git pull failed — continuing with current files"

echo ">>> Installing dependencies"
pip install --upgrade transformers datasets accelerate evaluate pyyaml wandb huggingface_hub

echo ">>> Starting DiZO training (config_dizo.yaml)"
accelerate launch --num_processes 1 train.py --config config_dizo.yaml

echo ">>> DiZO training complete."
