#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=reward_qwen
#SBATCH --nodelist=iris-hgx-1,iris-hgx-2
#SBATCH --output slurm/%j.out

# Detect cluster: prefer iris if mounted, otherwise fall back to haic.
if [ -d /iris/u/marcelto ]; then
    REWARD_LEARNING_DIR=/iris/u/marcelto/reward_learning
    CONDA_ROOT=/iris/u/marcelto/miniconda3
    DATA_AM208=/iris/u/am208/droid-robot
    DATA_ABHIJNYA=/iris/u/abhijnya/droid-robot
    export HOME=/iris/u/marcelto
elif [ -d /hai/scratch/marcelto ]; then
    REWARD_LEARNING_DIR=/hai/scratch/marcelto/reward_learning
    CONDA_ROOT=/hai/scratch/marcelto/miniconda3
    DATA_AM208=/hai/scratch/marcelto/data/am208
    DATA_ABHIJNYA=/hai/scratch/marcelto/data/abhijnya
else
    echo "ERROR: neither /iris/u/marcelto nor /hai/scratch/marcelto present" >&2
    exit 1
fi

cd "$REWARD_LEARNING_DIR"
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
conda activate qwen310


# Qwen3-VL reward model (following memer/QwenLM finetuning approach)
# - Vision frozen, MLP+LLM trainable
# - Gradient checkpointing enabled
# - lr=1e-5, cosine schedule (matching sft_qwen3_4b.sh)
# - batch_size=2 (single GPU, 4B model)
# python main.py \
#     --model qwen_open \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --log_interval 1 --save_interval 50 \
#     --preferences_dir "${DATA_AM208}/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/cross_preferences_setup,${DATA_AM208}/cross_preferences_setup" \
#     --task setup_table


# python main.py \
#     --model qwen_open_discounted \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 1 --epochs 1000 \
#     --lr 1e-5 \
#     --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --log_interval 1 --save_interval 50 \
#     --preferences_dir "${DATA_AM208}/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/cross_preferences_setup,${DATA_AM208}/cross_preferences_setup" \
#     --task setup_table

python main.py \
    --model qwen_open \
    --stride 20 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 1000 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --preload --preload_offsets 10 \
    --eval_interval 50 --vis_interval 999999 --log_interval 1 --save_interval 50 \
    --preferences_dir "${DATA_AM208}/preferences_setup" \
    --cross_preferences_dir "${DATA_ABHIJNYA}/cross_preferences_setup,${DATA_AM208}/cross_preferences_setup" \
    --task setup_table