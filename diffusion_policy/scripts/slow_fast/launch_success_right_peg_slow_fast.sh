#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=success_right_peg_slow_fast
#SBATCH --nodelist=iris4,iris5,iris6,iris7,iris8
#SBATCH --output slurm/%j.out

# Success + right peg baseline: filter shared data, then train with demo_success config.
FILTERED_DATA_DIR="shared_data_slow_fast_right_peg"

export PIPELINE_DIR="pipeline_output_slow_fast_success_right_peg"
export WANDB_PROJECT="slow_fast_success_right_peg"
export COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
export SKIP_REWARD_MODEL=true
export IS_CONDITIONED_EVAL=false
export SHARED_DATA_DIR="${FILTERED_DATA_DIR}"
export RESUME_FROM_PHASE=4

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"

export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda activate robodiffrew2

# Step 1: Filter shared data to only successful right-peg episodes
echo "=== Filtering data to successful right-peg episodes ==="
python scripts/filter_right_peg_success.py \
    --rollout_npz "shared_data_slow_fast/rollouts.npz" \
    --demo_hdf5 "shared_data_slow_fast/scripted_data/demos.hdf5" \
    --output_dir "${FILTERED_DATA_DIR}"

# Step 2: Train with demo_success config (skip phases 0-3)
bash scripts/run_pipeline_slow_fast.sh
