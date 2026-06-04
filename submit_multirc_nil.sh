#!/bin/bash
# Submits all 3 methods (LOZO, PseuZO, SubZero) for a given dataset
# Usage: bash submit_multirc_nil.sh [dataset_name]
# Defaults to multirc if no dataset is provided.

set -euo pipefail

DATASET=${1:-multirc}

echo ">>> Submitting all 3 jobs for dataset: ${DATASET}..."

bash submit_lozo.sh "${DATASET}"
sleep 2
bash submit_pseuzo.sh "${DATASET}"
sleep 2
bash submit_subzero.sh "${DATASET}"

cat <<EOF

>>> All 3 jobs for ${DATASET} submitted successfully!

To monitor your jobs, you can use:
List jobs:      runai list jobs
Stream logs:    runai logs -f <job-name> -p dhlab-nil
EOF
