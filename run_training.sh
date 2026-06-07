#!/bin/bash
# Run a ZO training job on the cluster.
#
# Usage:
#   bash run_training.sh --config configs/hizoo.yaml --task multirc
#   bash run_training.sh --config configs/mezo.yaml  --task copa   --node-pool a100-40g
#   bash run_training.sh --config configs/adam.yaml  --task multirc --image ic-registry.epfl.ch/cvlab/afan/pytorch:torch200
#
# Optional flags:
#   --node-pool  <pool>    default: h100
#   --image      <image>   default: registry.rcp.epfl.ch/course-cs-552/base-vllm:v1
#                          (has torch 2.8 + transformers 4.57 — supports Qwen3.5)
#                          Use ic-registry.epfl.ch/cvlab/afan/pytorch:torch200 only
#                          for methods that don't need Qwen3.5 (torch 2.0 only).
#   --owner      <owner>   default: maria
#   --dry-run             print the runai command without submitting
#
# Image notes:
#   base-vllm:v1   → torch 2.8, transformers 4.57, supports Qwen3.5-0.8B  ← DEFAULT
#   pytorch:torch200 → torch 2.0, transformers 4.48, Qwen2.5 only

set -e

# ---- defaults ----
CONFIG=""
TASK=""
NODE_POOL="h100"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
OWNER="maria"
DRY_RUN=0

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2";    shift 2 ;;
    --task)      TASK="$2";      shift 2 ;;
    --node-pool) NODE_POOL="$2"; shift 2 ;;
    --image)     IMAGE="$2";     shift 2 ;;
    --owner)     OWNER="$2";     shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift   ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$TASK" ]]; then
  echo "Usage: $0 --config <yaml> --task <multirc|copa> [--node-pool h100|a100-40g] [--image <img>] [--owner maria|nil|cheng] [--dry-run]"
  exit 1
fi

# ---- derive method name from config filename (replace _ with -) ----
METHOD=$(basename "$CONFIG" .yaml | tr '_' '-')

# ---- job name ----
JOB="pilligua-${METHOD}-${TASK}-$(date +%H%M%S)"

# ---- load secrets ----
source /mnt/cvlab/scratch/cvlab/home/pilligua/.secrets 2>/dev/null || true

LOG="/scratch/cvlab/home/pilligua/OML/logs/${JOB}.log"

# base-vllm:v1: torch 2.8 system + transformers 5.10.dev0 in packages_newt (for qwen3_5)
# pytorch:torch200: torch 2.0 system + packages_rcp2 for transformers 4.48
if [[ "$IMAGE" == *"pytorch:torch200"* ]]; then
  PYTHONPATH_VAL="/scratch/cvlab/home/pilligua/OML/packages_rcp2"
else
  # packages_newt FIRST to override system 4.57 with 5.10.dev0 (has qwen3_5)
  PYTHONPATH_VAL="/scratch/cvlab/home/pilligua/OML/packages_newt"
fi
PYTHONNOUSERSITE_VAL="1"

RUNAI_CMD=(
  runai submit "$JOB"
  -p cvlab-pilligua
  --image "$IMAGE"
  --gpu 1 --cpu 4 --memory 64G
  --node-pool "$NODE_POOL"
  --large-shm
  --pvc cvlab-scratch:/scratch
  --environment HF_TOKEN="$HF_TOKEN"
  --environment WANDB_API_KEY="$WANDB_API_KEY"
  --environment HF_HOME=/scratch/cvlab/home/pilligua/hf_cache
  --environment HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  --run-as-uid 315423 \
  --run-as-gid 30204 \
  ${PYTHONPATH_VAL:+--environment PYTHONPATH="$PYTHONPATH_VAL"} \
  ${PYTHONNOUSERSITE_VAL:+--environment PYTHONNOUSERSITE="$PYTHONNOUSERSITE_VAL"} \
  --command -- bash /scratch/cvlab/home/pilligua/OML/logs/${JOB}.sh
)

echo "========================================"
echo "  Job:      $JOB"
echo "  Config:   $CONFIG"
echo "  Task:     $TASK"
echo "  Owner:    $OWNER"
echo "  Pool:     $NODE_POOL"
echo "  Image:    $IMAGE"
echo "  Log:      $LOG"
echo "========================================"

mkdir -p /mnt/cvlab/scratch/cvlab/home/pilligua/OML/logs

# Stable per-job checkpoint dir (survives preemption — RunAI reuses $JOB).
# On restart the runner globs for an existing `last/` checkpoint under this
# dir and auto-resumes; train.py reuses the same WandB run + step counters.
CKPT_DIR="/scratch/cvlab/home/pilligua/OML/checkpoints/${JOB}"

# Write the runner script to disk — avoids bash -c quoting issues
RUNNER=/mnt/cvlab/scratch/cvlab/home/pilligua/OML/logs/${JOB}.sh
cat > "$RUNNER" << RUNNER_EOF
#!/bin/bash
set -e
source /scratch/cvlab/home/pilligua/.secrets 2>/dev/null || true
export HF_HOME=/scratch/cvlab/home/pilligua/hf_cache
export HUGGINGFACE_HUB_TOKEN=\$HF_TOKEN
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH_VAL
mkdir -p /scratch/cvlab/home/pilligua/OML/logs

# Auto-resume: if a prior 'last' checkpoint exists for this job, resume from it.
CKPT_DIR=$CKPT_DIR
RESUME_ARG=""
LAST_CKPT=\$(ls -d \$CKPT_DIR/*/last 2>/dev/null | head -1)
if [ -n "\$LAST_CKPT" ] && [ -f "\$LAST_CKPT/training_meta.json" ]; then
  echo "[resume] found checkpoint at \$LAST_CKPT — resuming"
  RESUME_ARG="--resume-from \$LAST_CKPT"
else
  echo "[resume] no checkpoint found — starting fresh"
fi

cd /scratch/cvlab/home/pilligua/OML/OptML_zero
python3 train.py \
    --config $CONFIG \
    --task $TASK \
    --owner $OWNER \
    --eval-batch-size 8 \
    --ckpt-dir \$CKPT_DIR \
    \$RESUME_ARG \
    2>&1 | tee -a $LOG
echo "[DONE] \$?"
RUNNER_EOF
chmod 755 "$RUNNER"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] runner script:"
  cat "$RUNNER"
  echo ""
  echo "[dry-run] would run:"
  printf '  %q\n' "${RUNAI_CMD[@]}"
  exit 0
fi

"${RUNAI_CMD[@]}"
echo ""
echo "Submitted: $JOB"
echo "Monitor:   tail -f /mnt/cvlab/scratch/cvlab/home/pilligua/OML/logs/${JOB}.log"
