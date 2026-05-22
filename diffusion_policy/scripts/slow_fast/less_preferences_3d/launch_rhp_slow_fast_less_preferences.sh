#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_sf_lp_3d
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for slow/fast — less preferences (100 pairs), CONTINUOUS conditioning
# 3D reward: speed_reward, smoothness, peg_reward
export PIPELINE_DIR="pipeline_output_slow_fast_less_preferences_3d_rhp_500"
export WANDB_PROJECT="slow_fast_less_preferences_3d_rhp_500"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# Left peg: speed [1, 4], Right peg: speed [1, 2]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

# 200 demos, no rollouts
export N_SCRIPTED=500
export SKIP_ROLLOUTS=true

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_mid_range"

# Per-axis eval conditioning targets (speed, smoothness, peg)
export REWARD_AXES="speed_reward,smoothness,peg_reward"
export EVAL_Z_POSITIVE="[0.9,0.5,0.9]"
export EVAL_Z_NEGATIVE="[0.5,0.2,0.8]"
export NUM_REWARD_DIMS=3
export REWARD_EPOCHS=100

# Less preferences: 100 pairs instead of all pairs
export N_PAIRS=500

# Iterative refinement: collect rollouts with diverse conditioning, retrain reward + policy
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
# speed_reward in [0.5, 0.9] x smoothness fixed at 0.9 x peg_reward fixed at 0.9
export CONDITIONING_TARGETS="0.5,0,0.9;0.6,0,0.9;0.7,0,0.9;0.8,0,0.9;0.9,0,0.9"

export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_slow_fast.sh
