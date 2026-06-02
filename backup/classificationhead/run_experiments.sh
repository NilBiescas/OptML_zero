#!/bin/bash
set -euo pipefail

echo "=== Running SNLI (Few-shot LOZO) ==="
export RUN_NAME="lozo-snli-k16"
accelerate launch --num_processes 1 train.py --config config_snli.yaml

echo "=== Running MNLI (Few-shot LOZO) ==="
export RUN_NAME="lozo-mnli-k16"
accelerate launch --num_processes 1 train.py --config config_mnli.yaml

echo "=== Running RTE (Few-shot LOZO) ==="
export RUN_NAME="lozo-rte-k16"
accelerate launch --num_processes 1 train.py --config config_rte.yaml

echo "All experiments completed!"
