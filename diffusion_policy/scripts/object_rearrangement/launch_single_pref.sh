#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp2_single_pref
#SBATCH --output slurm/%j.out

# Single-pref baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# Composite scalar reward = average of order + per-object placed (bread/can) +
# per-object drop (bread/can).
export PIPELINE_DIR="pipeline_output_object_rearrangement_single_pref"
export WANDB_PROJECT="object_rearrangement_single_pref"
export BASE_POLICY_DIR="base_policy_object_rearrangement"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# 2 objects need ~700 control steps; 800 leaves slack for grasp retries.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"

export SKIP_ROLLOUTS=true

# export SHARED_DATA_DIR="shared_data_object_rearrangement"
export SHARED_DATA_DIR="shared_data_object_rearrangement"

# Single composite scalar reward: average of order, per-object placed
# (bread/can), and per-object drop (bread/can).
export REWARD_AXES="composite(order_reward+bread_placed+can_placed+bread_drop+can_drop)"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=400
export COND_POLICY_EPOCHS=750
# Training seed for the single-pref policy (Phase 4). Applied to both
# `training.seed` and `task.dataset.seed`. Leave unset to use YAML default (42).
export TRAINING_SEED=62
# Conditioning-noise augmentation. Adds uniform [-AUGMENT_SCORE, +AUGMENT_SCORE]
# noise to the appended reward dims at sample time and re-rounds to the same
# 0.1 buckets — so each (state, action) pair sometimes gets re-labeled with an
# adjacent bucket. 0.0 disables.
export AUGMENT_SCORE=0.2
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE}"

# Eval z-score conditioning. Positive = best composite reward.
export EVAL_Z_POSITIVE="[0.8]"
export EVAL_Z_NEGATIVE="[-0.8]"

export N_PAIRS=70

# Phase 3 trains the composite-scalar reward model for THIS pipeline dir.
# Cannot reuse RHP's scores.json (different dimensionality).
export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_object_rearrangement.sh
