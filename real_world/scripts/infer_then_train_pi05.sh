#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --job-name=slurm_test
#SBATCH --output=slurm/%j.out # Ensure the 'slurm' folder exists!
#SBATCH --nodelist=iris-hgx-1,iris-hgx-2
#SBATCH --gres=gpu:1

set -euo pipefail


CKPT=exp/round1/2026-06-12_18-35-45_qwen_open_j15885981/checkpoints/step003000.pt
RUN_TAG=2026-06-12_18-35-45_qwen_open_j15885981
OUTPUT_SUBDIR=fold_pants_iter1_multi_qwen
DECIMAL_PLACES=1
ITER_TAG=iter0_2000
TASK=fold_pants
TASK_PROMPT="fold the shorts"


# Single source of truth — must match convert step below and pi05 input.
DATASET_REPO_ID="${OUTPUT_SUBDIR}_${DECIMAL_PLACES}dp_${ITER_TAG}"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------- Step 1: infer + convert (qwen310 env) ----------
cd "$REWARD_LEARNING_DIR"
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
conda activate qwen_rl

echo "=== [infer] Python: $(which python) ==="

python infer.py \
    --ckpt "$CKPT" \
    --preferences_dir "${PREFERENCES_DIR}" \
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