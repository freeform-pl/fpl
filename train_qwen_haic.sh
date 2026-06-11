#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --constraint=hopper
#SBATCH --job-name=reward_qwen_disc
#SBATCH --output=slurm/%j.out

# haic launcher for the Qwen3-VL reward model (discounted variant).
# Same cluster detection so paths resolve correctly whether run via sbatch
# on haic, bash on haic, or copied to iris.

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

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== Python: $(which python) ==="

# Qwen3-VL reward model — discounted (per-frame scoring, sum over time)
# - Vision frozen, MLP+LLM trainable
# - Gradient checkpointing enabled
# - lr=1e-5, cosine schedule (matching sft_qwen3_4b.sh)
# python main.py \
#     --model qwen_open_cum \
#     --seed 3 \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task single

# python main.py \
#     --model qwen_discounted \
#     --seed 3 \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task setup_table

# python main.py \
#     --model qwen_open \
#     --seed 3 \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task single 

# python main.py \
#     --model qwen_open \
#     --seed 3 \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task setup_table

# python main.py \
#     --model qwen_open \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 500 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_burger" \
#     --cross_preferences_dir "${DATA_AM208}/droid-robot/cross_preferences_burger,${DATA_ABHIJNYA}/droid-robot/cross_preferences_burger" \
#     --task single

# python main.py \
#     --model qwen_open_cum \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 500 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_burger" \
#     --cross_preferences_dir "${DATA_AM208}/droid-robot/cross_preferences_burger,${DATA_ABHIJNYA}/droid-robot/cross_preferences_burger" \
#     --task auto

# python main.py \
#     --model qwen_open \
#     --stride 10 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 100 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 100 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_pick_and_place" \
#     --cross_preferences_dir "${DATA_AM208}/droid-robot/cross_preferences_pick_and_place","${DATA_ABHIJNYA}/droid-robot/cross_preferences_pick_and_place" \
#     --task single

#   rsync -avh --partial --progress --exclude='*_large*.hdf5' /hai/scratch/marcelto/reward_learning/infer_output/setup_table_iter3_open_cum_qwen/2026-05-25_23-29-41_qwen_open_cum_j79567_2000/ marcelto@iris-ws-6.stanford.edu:/iris/u/marcelto/reward_learning/infer_output/setup_table_iter3_open_cum_qwen/

# python main.py \
#     --model qwen_open \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task single


python main.py \
    --model qwen_open \
    --stride 60 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 1000 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --preload --preload_offsets 30 \
    --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
    --preferences_dir "${DATA_AM208}/droid-robot/preferences" \
    --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences,${DATA_AM208}/droid-robot/cross_preferences,${DATA_AM208}/droid-robot/cross_preferences_extra/" \
    --task single
