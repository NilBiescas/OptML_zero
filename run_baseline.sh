#!/bin/bash
# Generic pod entrypoint for baseline runs (AdamW, SGD, ...). Reads CONFIG_FILE from env.

set -euo pipefail

mkdir -p runs/baseline hf_cache wandb
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
    echo ">>> SSH configured: using ~/.ssh/id_ed25519 for github.com"
fi

echo ">>> git pull in $(pwd)"
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" git pull --ff-only || echo "git pull failed (or repo offline) — continuing with current files"

echo ">>> Installing dependencies"
pip install --upgrade transformers datasets accelerate evaluate pyyaml

CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
echo ">>> Starting baseline training (${CONFIG_FILE})"
accelerate launch --num_processes 4 train.py --config "${CONFIG_FILE}"

echo ">>> Baseline training complete."
