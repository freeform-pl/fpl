#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sprf_sf_lp_3d
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Single-pref baseline for slow/fast — less preferences (100 pairs)
# 3D composite: speed_reward + smoothness + peg_reward
export PIPELINE_DIR="pipeline_output_slow_fast_less_preferences_3d_single_pref_500"
export WANDB_PROJECT="slow_fast_less_preferences_3d_single_pref_500"
export NUM_REWARD_DIMS=1
export REWARD_AXES="composite(speed_reward+smoothness+peg_reward)"

# Left peg: speed [1, 4], Right peg: speed [1, 2]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

# 200 demos, no rollouts
export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_mid_range"

# Per-axis eval z-score conditioning (length must match NUM_REWARD_DIMS=1)
export EVAL_Z_POSITIVE="[0.7]"
export REWARD_EPOCHS=100

# Less preferences: 100 pairs instead of all pairs
export N_PAIRS=500

export RESUME_FROM_PHASE=1

bash scripts/run_pipeline_slow_fast.sh
