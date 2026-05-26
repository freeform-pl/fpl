#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp2_awr
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# AWR baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# Uses the same 5-D reward axes as RHP, but the AWR dataset averages them into
# a scalar advantage weight (exp(beta * mean_z)). No reward conditioning at eval.
export PIPELINE_DIR="pipeline_output_pickplace_2obj_fixed_awr"
export WANDB_PROJECT="pickplace_2obj_fixed_awr"
export BASE_POLICY_DIR="base_policy_pickplace_2obj_fixed"
export COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true

# 2-object variant: keep Bread + Can in the scene, clear Milk + Cereal.
export N_ACTIVE_OBJECTS=2
# 2 objects need ~700 control steps; 800 leaves slack for grasp retries.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"

# Preference-axis sampling for scripted demos
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=2
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 1000 demos, no rollouts
export N_SCRIPTED=300
export SKIP_ROLLOUTS=true

# export SHARED_DATA_DIR="shared_data_pickplace_2obj"
export SHARED_DATA_DIR="shared_data_pickplace_2obj_fixed"

# Same 5-D reward axes as RHP — AWR averages them into a scalar weight.
export REWARD_AXES="order_reward,bread_placed,can_placed,bread_drop,can_drop"
export NUM_REWARD_DIMS=5
export REWARD_EPOCHS=400
export COND_POLICY_EPOCHS=750
# Training seed for the AWR policy (Phase 4). Applied to both
# `training.seed` and `task.dataset.seed`. Leave unset to use YAML default (42).
export TRAINING_SEED=42
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

export N_PAIRS=70

# Phase 3 trains the reward model (scores.json) for THIS pipeline dir; phase 4
# trains the AWR policy off those scores. No iterative refinement for AWR.
export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_pickplace.sh
