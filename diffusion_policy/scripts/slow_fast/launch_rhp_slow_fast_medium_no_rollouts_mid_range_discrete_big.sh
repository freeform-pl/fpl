#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_sf_mid_big
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Same as mid_range_discrete but with a bigger policy network (12 layers, 512 emb, 8 heads)
export PIPELINE_DIR="pipeline_output_slow_fast_medium_no_rollouts_mid_range_discrete_big_rhp"
export WANDB_PROJECT="slow_fast_medium_no_rollouts_mid_range_discrete_big_rhp"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=true

# Left peg: speed [1, 4], Right peg: speed [1, 2]
export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

# 200 demos, no rollouts
export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

# Reuse same shared data as mid_range
export SHARED_DATA_DIR="shared_data_slow_fast_medium_no_rollouts_mid_range"

# Per-axis eval conditioning targets
export REWARD_AXES="speed_reward,peg_reward"
export EVAL_Z_POSITIVE="[0.9,0.95]"
export EVAL_Z_NEGATIVE="[0.5,0.95]"
export NUM_REWARD_DIMS=2
export REWARD_EPOCHS=20

# Bigger policy network
export EXTRA_POLICY_OVERRIDES="++policy.model.n_layer=12 ++policy.model.n_emb=512 ++policy.model.n_head=8"

# Iterative refinement
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.5,0.95;0.6,0.95;0.7,0.95;0.8,0.95;0.9,0.95"

export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_slow_fast.sh
