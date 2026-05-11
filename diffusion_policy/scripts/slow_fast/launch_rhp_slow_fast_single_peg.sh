#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_sf_single_peg
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Sanity check: single peg (left only), speed=1, with rollouts, discrete conditioning
export PIPELINE_DIR="pipeline_output_slow_fast_single_peg_rhp"
export WANDB_PROJECT="slow_fast_single_peg_rhp"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=true

# Single peg (left only), speed factor 1
export SPEED_FACTOR_LEFT=1
export SPEED_FACTOR_RIGHT=1
export TARGET_PEG=left

# 200 demos, with rollouts
export N_SCRIPTED=200

# Separate shared data
export SHARED_DATA_DIR="shared_data_slow_fast_single_peg"

# Single reward axis: speed only (peg doesn't vary)
export REWARD_AXES="speed_reward"
export EVAL_Z_POSITIVE="[0.9]"
export EVAL_Z_NEGATIVE="[-0.9]"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=20
export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_slow_fast.sh
