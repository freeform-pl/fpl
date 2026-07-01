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

source /iris/u/abhijnya/FPL/marcel/reward_learning/real_world/scripts/config.sh

cd "$REWARD_LEARNING_DIR"
conda activate qwen310

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
    --preferences_dir "/iris/u/abhijnya/FPL/test_pref_path" \
    --cross_preferences_dir "/iris/u/abhijnya/FPL/test_cross_pref_path" \
    --task auto
