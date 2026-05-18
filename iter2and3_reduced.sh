#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint=hopper
#SBATCH --job-name=inf&pi05
#SBATCH --output=slurm/%j.out

# Combined haic launcher: runs infer.py + convert_custom_droid_to_lerobot.py,
# then trains pi05 on the LeRobot repo that was just created. Single source of
# truth for the repo_id so the two steps cannot drift.

set -euo pipefail

if [ -d /iris/u/marcelto ]; then
    REWARD_LEARNING_DIR=/iris/u/marcelto/reward_learning
    CONDA_ROOT=/iris/u/marcelto/miniconda3
    DATA_AM208=/iris/u/am208/droid-robot
    DATA_ABHIJNYA=/iris/u/abhijnya/droid-robot
    OUTPUT_ROOT=/iris/u/marcelto/reward_learning/infer_output
    export HF_LEROBOT_HOME=/iris/u/marcelto/data
    export HOME=/iris/u/marcelto
elif [ -d /hai/scratch/marcelto ]; then
    REWARD_LEARNING_DIR=/hai/scratch/marcelto/reward_learning
    CONDA_ROOT=/hai/scratch/marcelto/miniconda3
    DATA_AM208=/hai/scratch/marcelto/data/am208/droid-robot
    DATA_ABHIJNYA=/hai/scratch/marcelto/data/abhijnya/droid-robot
    OUTPUT_ROOT=/hai/scratch/marcelto/reward_learning/infer_output
    export HF_LEROBOT_HOME=/hai/scratch/marcelto/data
else
    echo "ERROR: neither /iris/u/marcelto nor /hai/scratch/marcelto present" >&2
    exit 1
fi

OPENPI_DIR="$REWARD_LEARNING_DIR/openpi"
OPENPI_PY="$OPENPI_DIR/.venv/bin/python"


# multi
WANDB_PROJ=2026-05-17_22-32-48_qwen_open_j77969
ITER=2500
CKPT=exp/${WANDB_PROJ}/checkpoints/step00${ITER}.pt
RUN_TAG=${WANDB_PROJ}_${ITER}_iter23_reduced
TASK=setup_table_reduced
OUTPUT_SUBDIR=${TASK}_iter23_multi_qwen
DECIMAL_PLACES=1
ITER_TAG=iter23_${ITER}
TASK_PROMPT="set up the table"

# single
WANDB_PROJ=2026-05-17_07-52-13_qwen_open_j77911
ITER=6000
CKPT=exp/${WANDB_PROJ}/checkpoints/step00${ITER}.pt
RUN_TAG=${WANDB_PROJ}_${ITER}_iter23_reduced
TASK=setup_table_reduced
OUTPUT_SUBDIR=${TASK}_iter23_single_qwen
DECIMAL_PLACES=1
ITER_TAG=iter23_${ITER}
TASK_PROMPT="set up the table"



# Single source of truth — must match convert step below and pi05 input.
DATASET_REPO_ID="marcelto/${OUTPUT_SUBDIR}_${DECIMAL_PLACES}dp_${ITER_TAG}"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------- Step 1: infer + convert (qwen310 env) ----------
cd "$REWARD_LEARNING_DIR"
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
conda activate qwen310

echo "=== [infer] Python: $(which python) ==="

python infer.py \
    --ckpt "$CKPT" \
    --preferences_dir "${DATA_AM208}/preferences_setup,${DATA_AM208}/demos/setup,${DATA_ABHIJNYA}/demos/table_setup" \
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
