#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
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
python main.py \
    --model qwen_open_discounted \
    --stride 20 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 1000 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --preload --preload_offsets 10 \
    --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 50 \
    --preferences_dir "${DATA_AM208}/preferences_setup" \
    --cross_preferences_dir "${DATA_ABHIJNYA}/cross_preferences_setup,${DATA_AM208}/cross_preferences_setup" \
    --task setup_table

# python main.py \
#     --model qwen_open_discounted \
#     --stride 60 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 30 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 50 \
#     --preferences_dir /iris/u/am208/ \
#     --cross_preferences_dir /iris/u/abhijnya/,/iris/u/am208/droid-robot/cross_preferences \
#     --task fold_pants 

# python main.py \
#     --model qwen_open_discounted \
#     --stride 60 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 30 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 50 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences,${DATA_AM208}/droid-robot/cross_preferences" \
#     --task fold_pants
