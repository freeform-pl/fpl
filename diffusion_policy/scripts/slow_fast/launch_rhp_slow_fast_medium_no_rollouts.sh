#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_sf_med_no_roll
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for slow/fast MEDIUM — no rollouts, 200 scripted demos only
export PIPELINE_DIR="pipeline_output_slow_fast_medium_no_rollouts_rhp"
export WANDB_PROJECT="slow_fast_medium_no_rollouts_rhp"
export IS_CONDITIONED_EVAL=true
export RESUME_FROM_PHASE=4

# Medium speed gap
export SPEED_FACTOR_LEFT=1
export SPEED_FACTOR_RIGHT=4.0

# 200 demos, no rollouts
export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts"

# Per-axis eval conditioning targets
export REWARD_AXES="speed_reward,peg_reward"
export EVAL_Z_POSITIVE="[0.99,0.99]"
export EVAL_Z_NEGATIVE="[0.98,-0.99]"
export NUM_REWARD_DIMS=2
export REWARD_EPOCHS=20
bash scripts/run_pipeline_slow_fast.sh
