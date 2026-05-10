#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_slow_fast_medium
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for slow/fast MEDIUM experiment (smaller speed gap)
export PIPELINE_DIR="pipeline_output_slow_fast_medium_rhp"
export WANDB_PROJECT="slow_fast_medium_rhp"
export IS_CONDITIONED_EVAL=true
export RESUME_FROM_PHASE=1

# Medium speed gap (default: left=0.6, right=4.0)
export SPEED_FACTOR_LEFT=1
export SPEED_FACTOR_RIGHT=4.0

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium"
export BASE_POLICY_DIR="base_policy_slow_fast_medium"

# Per-axis eval z-score conditioning
export EVAL_Z_POSITIVE="[0.9,0.9,0.9,0.9]"
export N_SCRIPTED=200

bash scripts/run_pipeline_slow_fast.sh
