#!/bin/bash

source ./scripts/config.sh
conda activate qwen_rl

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== Python: $(which python) ==="


python3 train_reward_model.py \
    --model qwen_open \
    --stride 10 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 1 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --preload --preload_offsets 10 \
    --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
    --preferences_dir "${PREFERENCES_DIR}" \
    --cross_preferences_dir "${CROSS_PREFERENCES_DIR}" \
    --task auto
