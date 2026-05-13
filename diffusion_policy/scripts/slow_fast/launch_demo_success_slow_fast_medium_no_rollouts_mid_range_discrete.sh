#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=dsuc_sf_mid
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Demo-success baseline for slow/fast MEDIUM — no rollouts, 200 scripted demos, mid range, DISCRETE conditioning
export PIPELINE_DIR="pipeline_output_slow_fast_medium_no_rollouts_mid_range_discrete_demo_success"
export WANDB_PROJECT="slow_fast_medium_no_rollouts_mid_range_discrete_demo_success"
export COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
export SKIP_REWARD_MODEL=true
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true

# Left peg: speed [1, 4], Right peg: speed [1, 2]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

# 200 demos
export N_SCRIPTED=200
export SKIP_ROLLOUTS=false

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_mid_range"

# Per-axis eval conditioning targets
export REWARD_AXES="speed_reward,peg_reward"
export NUM_REWARD_DIMS=2

export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_slow_fast.sh
