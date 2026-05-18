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
#SBATCH --job-name=pi05_droid_ft
#SBATCH --output=slurm/%j.out

# haic launcher for pi05 full-finetune on a custom DROID LeRobot dataset.
# Mirrors infer_haic.sh: detects whether we're on iris or haic and resolves
# paths accordingly. Uses the openpi repo at $REWARD_LEARNING_DIR/openpi and
# its uv-managed environment.

if [ -d /iris/u/marcelto ]; then
    REWARD_LEARNING_DIR=/iris/u/marcelto/reward_learning
    export HF_LEROBOT_HOME=/iris/u/marcelto/data
    export HOME=/iris/u/marcelto
elif [ -d /hai/scratch/marcelto ]; then
    REWARD_LEARNING_DIR=/hai/scratch/marcelto/reward_learning
    export HF_LEROBOT_HOME=/hai/scratch/marcelto/data
else
    echo "ERROR: neither /iris/u/marcelto nor /hai/scratch/marcelto present" >&2
    exit 1
fi

OPENPI_DIR="$REWARD_LEARNING_DIR/openpi"
cd "$OPENPI_DIR"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== uv: $(which uv) ==="

# Must match the repo_name produced by convert_custom_droid_to_lerobot.py in
# infer_haic.sh: marcelto/${OUTPUT_SUBDIR}_1dp_iter2_5400
OUTPUT_SUBDIR=setup_table_multi_qwen
# OUTPUT_SUBDIR=setup_table_reduced_multi_qwen_discounted
# DATASET_REPO_ID="marcelto/setup_table_iter3_single_qwen_1dp_iter3_2000"
DATASET_REPO_ID="marcelto/setup_table_iter3_multi_qwen_1dp_iter3_3000"

EXP_NAME="pi05_${DATASET_REPO_ID}"

# Full finetune of pi05 on our custom DROID LeRobot dataset.
# Base config: pi05_droid_finetune (non-LoRA, action_dim=32, action_horizon=16,
# initialized from gs://openpi-assets/checkpoints/pi05_droid/params).
# Only the dataset repo_id is overridden here.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_droid_finetune \
    --exp-name="$EXP_NAME" \
    --resume \
    --keep-train-state-only-latest \
    --data.repo_id="$DATASET_REPO_ID"
