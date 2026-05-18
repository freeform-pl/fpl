#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=hopper
#SBATCH --job-name=reward_qwen_debug
#SBATCH --output=slurm/%j.out

# Fast debug run for the multi-GPU Qwen3-VL reward model.
# - Skips all HDF5 / cross-pref / anchor loading via --debug_dummy
# - Tiny synthetic dataset → first DDP step in seconds, not minutes
# - No wandb, no visualizations
# Use to verify DDP setup, check OOM behavior, and iterate on hyperparams.

if [ -d /iris/u/marcelto ]; then
    REWARD_LEARNING_DIR=/iris/u/marcelto/reward_learning
    CONDA_ROOT=/iris/u/marcelto/miniconda3
    export HOME=/iris/u/marcelto
elif [ -d /hai/scratch/marcelto ]; then
    REWARD_LEARNING_DIR=/hai/scratch/marcelto/reward_learning
    CONDA_ROOT=/hai/scratch/marcelto/miniconda3
else
    echo "ERROR: neither /iris/u/marcelto nor /hai/scratch/marcelto present" >&2
    exit 1
fi

cd "$REWARD_LEARNING_DIR"
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
conda activate qwen310

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Launching torchrun with nproc_per_node=$NUM_GPUS"
MASTER_PORT=$((20000 + RANDOM % 20000))

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --debug_dummy \
    --debug_dummy_train 64 --debug_dummy_val 16 \
    --model qwen_open \
    --stride 20 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 2 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --eval_interval 999999 --vis_interval 999999 --small_vis_interval 999999 \
    --log_interval 1 --save_interval 999999 \
    --no_wandb \
    --task setup_table
