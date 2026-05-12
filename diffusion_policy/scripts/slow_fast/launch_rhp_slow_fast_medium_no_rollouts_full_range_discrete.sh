#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_sf_med_nr_fr
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for slow/fast MEDIUM — no rollouts, 200 scripted demos, DISCRETE conditioning
# Both pegs sample speed uniformly from [1, 4]
export PIPELINE_DIR="pipeline_output_slow_fast_medium_no_rollouts_full_range_discrete_rhp"
export WANDB_PROJECT="slow_fast_medium_no_rollouts_full_range_discrete_rhp"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=true

# Both pegs: speed uniformly sampled from [1, 4]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 4"

# 200 demos, no rollouts
export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

# Separate shared data for this variant
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_full_range"

# Per-axis eval conditioning targets
export REWARD_AXES="speed_reward,peg_reward"
export EVAL_Z_POSITIVE="[0.5,0.95]"
export EVAL_Z_NEGATIVE="[0.5,-0.95]"
export NUM_REWARD_DIMS=2
export REWARD_EPOCHS=20

# Iterative refinement: collect rollouts with diverse conditioning, retrain reward + policy
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
# speed_reward in [0.5, 0.9] x peg_reward fixed at 0.95
export CONDITIONING_TARGETS="0.5,0.95;0.6,0.95;0.7,0.95;0.8,0.95;0.9,0.95"

export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_slow_fast.sh
