#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=hopper
#SBATCH --job-name=reward_qwen_debug_single
#SBATCH --output=slurm/%j.out

# Single-GPU stress test with the synthetic dummy dataset (no padding,
# always 20 full frames per trajectory — worst-case memory).
# Compare PeakMem here vs. the same batch size under DDP to confirm
# whether DDP overhead is what's pushing us over, or whether
# single-GPU is already at the ceiling.

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

# Plain `python` (no torchrun) so we get the true single-GPU baseline.
python main.py \
    --debug_dummy \
    --debug_dummy_train 64 --debug_dummy_val 16 \
    --model qwen_open \
    --stride 20 --seq_len 20 --img_size 128 \
    --batch_size 32 --epochs 1 \
    --lr 1e-5 \
    --equal_weight 0.0 \
    --eval_interval 999999 --vis_interval 999999 --small_vis_interval 999999 \
    --log_interval 1 --save_interval 999999 \
    --no_wandb \
    --task setup_table
