#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --constraint=hopper
#SBATCH --job-name=reward_qwen_multi
#SBATCH --output=slurm/%j.out

# Multi-GPU (DDP) haic launcher for the Qwen3-VL reward model.

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

# Reduce CUDA fragmentation under DDP (recommended by the OOM error message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== Python: $(which python) ==="

# Number of GPUs for torchrun. Falls back to 1 if not running under SLURM.
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Launching torchrun with nproc_per_node=$NUM_GPUS"

# Use a random port to avoid collisions if multiple jobs share a node.
MASTER_PORT=$((20000 + RANDOM % 20000))

# Multi-GPU Qwen3-VL reward model training (DDP).
# - Each GPU processes --batch_size samples; global batch = batch_size * NUM_GPUS.
# - Vision frozen, MLP+LLM trainable
# - Gradient checkpointing enabled
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     main.py \
#     --model qwen_open \
#     --stride 20 --seq_len 20 --img_size 128 \
#     --batch_size 32 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task setup_table


# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     main.py \
#     --model qwen_open \
#     --stride 10 --seq_len 40 --img_size 128 \
#     --batch_size 16 --epochs 1000 \
#     --lr 1e-5 \
#     --equal_weight 0.0 \
#     --preload --preload_offsets 10 \
#     --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
#     --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
#     --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
#     --task setup_table


torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --model qwen_open \
    --stride 5 --seq_len 80 --img_size 128 \
    --batch_size 8 --epochs 1000 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --preload --preload_offsets 5 \
    --eval_interval 50 --vis_interval 999999 --small_vis_interval 200 --log_interval 1 --save_interval 1000 \
    --preferences_dir "${DATA_AM208}/droid-robot/preferences_setup" \
    --cross_preferences_dir "${DATA_ABHIJNYA}/droid-robot/cross_preferences_setup,${DATA_AM208}/droid-robot/cross_preferences_setup" \
    --task setup_table