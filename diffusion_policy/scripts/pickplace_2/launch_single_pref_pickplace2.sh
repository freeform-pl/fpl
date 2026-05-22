#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp2_single_pref
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Single-pref baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# Composite scalar reward = average of order + per-object placed (bread/can) +
# per-object drop (bread/can).
export PIPELINE_DIR="pipeline_output_pickplace_2obj_fixed_single_pref"
export WANDB_PROJECT="pickplace_2obj_fixed_single_pref"
export BASE_POLICY_DIR="base_policy_pickplace_2obj_fixed"
export IS_CONDITIONED_EVAL=true
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

# Single composite scalar reward: average of order, per-object placed
# (bread/can), and per-object drop (bread/can).
export REWARD_AXES="composite(order_reward+bread_placed+can_placed+bread_drop+can_drop)"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=40
export COND_POLICY_EPOCHS=1500
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

# Eval z-score conditioning. Positive = best composite reward.
export EVAL_Z_POSITIVE="[0.9]"
export EVAL_Z_NEGATIVE="[-0.9]"

export N_PAIRS=500

# Iterative refinement: composite scalar has a single knob, so each iteration
# pins it to the positive target. Iteration count matches the RHP baseline.
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.9;0.9;0.9"

# Phase 3 trains the composite-scalar reward model for THIS pipeline dir.
# Cannot reuse RHP's scores.json (different dimensionality).
export RESUME_FROM_PHASE=3
bash scripts/run_pipeline_pickplace.sh
