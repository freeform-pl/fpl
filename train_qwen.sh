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

cd /iris/u/marcelto/reward_learning
export HOME=/iris/u/marcelto
eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"

# eval "$(/scr/marcelto/miniconda3/bin/conda shell.bash hook)"

conda activate qwen310


# Qwen3-VL reward model (following memer/QwenLM finetuning approach)
# - Vision frozen, MLP+LLM trainable
# - Gradient checkpointing enabled
# - lr=1e-5, cosine schedule (matching sft_qwen3_4b.sh)
# - batch_size=2 (single GPU, 4B model)
python main.py \
    --model qwen_open_discounted \
    --stride 20 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 1000 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --preload --preload_offsets 10 \
    --eval_interval 50 --vis_interval 999999 --log_interval 1 --save_interval 50 \
    --preferences_dir /iris/u/am208/droid-robot/preferences_setup \
    --cross_preferences_dir /iris/u/abhijnya/droid-robot/cross_preferences_setup,/iris/u/am208/droid-robot/cross_preferences_setup \
    --task setup_table


# python main.py \
#     --model qwen_open_discounted \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 1 --epochs 1000 \
#     --lr 1e-5 \
#     --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --log_interval 1 --save_interval 50 \
#     --preferences_dir /iris/u/am208/droid-robot/preferences_setup \
#     --cross_preferences_dir /iris/u/abhijnya/droid-robot/cross_preferences_setup,/iris/u/am208/droid-robot/cross_preferences_setup \
#     --task setup_table