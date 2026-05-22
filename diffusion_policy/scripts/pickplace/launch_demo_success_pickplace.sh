#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp4_demo_success
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Demo-success (success-only) baseline for the PickPlace 4-object benchmark.
# Active objects: Bread + Can + Milk + Cereal (right-first canonical order).
# Trains a plain flow transformer on the subset of demos that fully placed
# every active object. No reward model, no conditioning at eval.
export PIPELINE_DIR="pipeline_output_pickplace_4obj_fixed_demo_success"
export WANDB_PROJECT="pickplace_4obj_fixed_demo_success"
export BASE_POLICY_DIR="base_policy_pickplace_4obj_fixed"
export COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
export SKIP_REWARD_MODEL=true
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
export SKIP_ROLLOUTS=false

export SHARED_DATA_DIR="shared_data_pickplace_4obj_fixed"

# Reward axes are unused (SKIP_REWARD_MODEL=true) but kept for consistency
# with the env_runner's per-axis logging at eval time.
export REWARD_AXES="order_reward,bread_placed,can_placed,milk_placed,cereal_placed,bread_drop,can_drop,milk_drop,cereal_drop"
export NUM_REWARD_DIMS=9
export COND_POLICY_EPOCHS=1500
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

# Skip reward-model phase; jump straight to policy training on the filtered demos.
export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_pickplace.sh
