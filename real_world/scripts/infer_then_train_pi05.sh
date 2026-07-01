#!/bin/bash

set -euo pipefail
source ./scripts/config.sh

# ============================================================
# USER INPUTS
# ============================================================
CKPT=exp/<add your checkpoint folder name>/checkpoints/final.pt

# ============================================================
# DERIVED PATHS — No need to edit
# ============================================================
RUN_TAG=$(basename $(dirname $(dirname "$CKPT")))
OUTPUT_SUBDIR="$TASK"
DECIMAL_PLACES=1
# Single source of truth — must match convert step below and pi05 input.
DATASET_REPO_ID="abhijnya/${OUTPUT_SUBDIR}_${DECIMAL_PLACES}dp"


echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true


# ---------- Step 1: infer + convert (qwen_rl env) ----------
conda activate qwen_rl

echo "=== [infer] Python: $(which python) ==="

python infer.py \
    --ckpt "$CKPT" \
    --preferences_dir "${PREFERENCES_DIR},${DEMOS_DIR}" \
    --output_dir "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_TAG}" \
    --task "$TASK"

"$OPENPI_PY" convert_custom_droid_to_lerobot.py \
    --args.scores_dir "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_TAG}" \
    --args.repo_name "$DATASET_REPO_ID" \
    --args.task_prompt "$TASK_PROMPT" \
    --args.score_type standardized \
    --args.decimal_places "$DECIMAL_PLACES"

conda deactivate


# ---------- Step 2: pi05 finetune on the repo just created ----------
cd "$OPENPI_DIR"

echo "=== [train] uv: $(which uv) ==="
echo "=== [train] dataset: $DATASET_REPO_ID ==="

EXP_NAME="pi05_${DATASET_REPO_ID}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_droid_finetune \
    --exp-name="$EXP_NAME" \
    --resume \
    --keep-train-state-only-latest \
    --data.repo_id="$DATASET_REPO_ID"