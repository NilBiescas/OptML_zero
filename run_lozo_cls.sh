#!/bin/bash
# Runs INSIDE the pod to execute classification language modeling LOZO training.
# Called by submit_lozo_cls.sh via runai submit --command.

set -euo pipefail

# Create local directories for cache and logs in current directory
mkdir -p runs/cls hf_cache wandb
export HF_HOME="$(pwd)/hf_cache"
export WANDB_DIR="$(pwd)/wandb"

# Make `python` resolve in the image
ln -sf "$(command -v python3)" /usr/local/bin/python || true

# --- bootstrap so `git pull` works in a fresh pod dynamically ---
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
git pull --ff-only || echo "git pull failed (or repo offline) — continuing with current files"

echo ">>> Installing dependencies for Classification LOZO"
pip install --upgrade transformers datasets accelerate evaluate pyyaml

echo ">>> Starting Classification LOZO training (classificationhead/config.yaml)"
accelerate launch --num_processes 4 train_cls.py --config classificationhead/config.yaml

echo ">>> Classification LOZO Training complete."
