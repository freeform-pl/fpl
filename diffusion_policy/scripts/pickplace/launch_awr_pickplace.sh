#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp4_awr
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# AWR baseline for the PickPlace 4-object benchmark.
# Active objects: Bread + Can + Milk + Cereal (right-first canonical order).
# Uses the same 9-D reward axes as RHP, but the AWR dataset averages them into
# a scalar advantage weight (exp(beta * mean_z)). No reward conditioning at eval.
export PIPELINE_DIR="pipeline_output_pickplace_4obj_fixed_awr"
export WANDB_PROJECT="pickplace_4obj_fixed_awr"
export BASE_POLICY_DIR="base_policy_pickplace_4obj_fixed"
export COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true

# 4-object variant: all four objects active in the scene.
export N_ACTIVE_OBJECTS=4
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=1000"

# Preference-axis sampling for scripted demos
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=4
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 1000 demos, no rollouts
export N_SCRIPTED=1000
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_pickplace_4obj_fixed"

# Same 9-D reward axes as RHP — AWR averages them into a scalar weight.
export REWARD_AXES="order_reward,bread_placed,can_placed,milk_placed,cereal_placed,bread_drop,can_drop,milk_drop,cereal_drop"
export NUM_REWARD_DIMS=9
export REWARD_EPOCHS=40
export COND_POLICY_EPOCHS=1500
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

export N_PAIRS=500

# Phase 3 trains the reward model (scores.json) for THIS pipeline dir; phase 4
# trains the AWR policy off those scores. No iterative refinement for AWR.
export RESUME_FROM_PHASE=3
bash scripts/run_pipeline_pickplace.sh
