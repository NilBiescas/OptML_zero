#!/bin/bash
# Fast download from Hugging Face instead of Princeton
pip install -U huggingface_hub

# Get the username dynamically based on the HF_TOKEN
HF_USER=$(python -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ.get('HF_TOKEN')).whoami()['name'])")

echo "Downloading dataset from: ${HF_USER}/LM-BFF-datasets"

# Download the 'original' folder from the HF dataset directly into the pod's data/original directory
huggingface-cli download --repo-type dataset "${HF_USER}/LM-BFF-datasets" original --local-dir original --token "${HF_TOKEN:-}"
