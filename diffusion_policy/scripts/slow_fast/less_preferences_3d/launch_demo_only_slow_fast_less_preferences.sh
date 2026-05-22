#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=demo_sf_lp_3d
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Demo-only baseline for slow/fast MEDIUM — train base policy on demos, evaluate it directly
# 3D reward eval: speed_reward, smoothness, peg_reward (no preferences used here)
export PIPELINE_DIR="pipeline_output_slow_fast_less_preferences_3d_demo_only"
export WANDB_PROJECT="slow_fast_less_preferences_3d_demo_only"
export SKIP_REWARD_MODEL=true
export SKIP_POLICY_TRAINING=true
export IS_CONDITIONED_EVAL=false

# Left peg: speed [1, 4], Right peg: speed [1, 2]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

# 200 demos, skip rollout collection but still train base policy
export N_SCRIPTED=200
export SKIP_ROLLOUTS=false

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_mid_range"

# Per-axis eval conditioning targets (speed, smoothness, peg)
export REWARD_AXES="speed_reward,smoothness,peg_reward"
export NUM_REWARD_DIMS=3

export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_slow_fast.sh
