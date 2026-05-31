#!/bin/bash
# Get the username dynamically based on the HF_TOKEN
HF_USER=$(python -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ.get('HF_TOKEN')).whoami()['name'])")

echo "Downloading dataset from: ${HF_USER}/LM-BFF-datasets"

# Download the dataset (which contains the 'original' folder) into the current directory
huggingface-cli download --repo-type dataset "${HF_USER}/LM-BFF-datasets" --local-dir . --token "${HF_TOKEN:-}"
