#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=awr_sf_lp_3d
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# AWR baseline for slow/fast — less preferences (100 pairs), DISCRETE conditioning
# 3D reward: speed_reward, smoothness, peg_reward
export PIPELINE_DIR="pipeline_output_slow_fast_less_preferences_3d_awr_500"
export WANDB_PROJECT="slow_fast_less_preferences_3d_awr_500"
export COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true

# Left peg: speed [1, 4], Right peg: speed [1, 2]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

# 200 demos, no rollouts
export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_mid_range"

# Per-axis eval conditioning targets (speed, smoothness, peg)
export REWARD_AXES="speed_reward,smoothness,peg_reward"
export NUM_REWARD_DIMS=3
export REWARD_EPOCHS=100
export BASE_POLICY_EPOCHS=2000

# Less preferences: 100 pairs instead of all pairs
export N_PAIRS=500

export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_slow_fast.sh
