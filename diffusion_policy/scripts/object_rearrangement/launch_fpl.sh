#!/bin/bash

# FPL baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# 5D reward: order + bread_placed + can_placed + bread_drop + can_drop.
export PIPELINE_DIR="pipeline_output_object_rearrangement_fpl"
export WANDB_PROJECT="object_rearrangement_fpl"
export BASE_POLICY_DIR="base_policy_object_rearrangement"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# 2 objects need ~700 control steps; 800 leaves slack for grasp retries.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"

export SKIP_ROLLOUTS=true

# export SHARED_DATA_DIR="shared_data_object_rearrangement"
export SHARED_DATA_DIR="shared_data_object_rearrangement"

# 5D axes: order, per-object placed (bread/can), per-object drop (bread/can).
# ORDER MUST MATCH the values in EVAL_Z_* below.
export REWARD_AXES="order_reward,bread_placed,can_placed,bread_drop,can_drop"
export NUM_REWARD_DIMS=5
export REWARD_EPOCHS=400
export COND_POLICY_EPOCHS=750
# Training seed for the conditioned policy (Phase 4). Same value is applied
# to `training.seed` (model init, optimizer, dataloader shuffle) AND
# `task.dataset.seed` (train/val split). Change to get an independent run
# end-to-end without re-collecting data or re-training the base policy.
# Leave unset / empty to use the workspace YAML default (42).
export TRAINING_SEED=52
# Conditioning-noise augmentation. Adds uniform [-AUGMENT_SCORE, +AUGMENT_SCORE]
# noise to the appended reward dims at sample time and re-rounds to the same
# 0.1 buckets — so each (state, action) pair sometimes gets re-labeled with an
# adjacent bucket. 0.0 disables. 0.1 (a full bucket width) is the natural
# starting point; raise to broaden the conditioning support seen in training.
export AUGMENT_SCORE=0.2
# Score quantisation. True = round stored conditioning to 0.1 buckets (and
# re-round after augment noise). False = pass continuous z-scores through.
# Clipping to [-1, 1] always applies either way.
export ROUND_SCORES=False
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE} ++round_scores=${ROUND_SCORES}"

# Eval z-score conditioning. Positive = best on every axis, negative = worst.
export EVAL_Z_POSITIVE="[0.8,0.8,0.8,0.8,0.8]"
export EVAL_Z_NEGATIVE="[-0.8,0.7,0.8,0.8,0.8]"

export N_PAIRS=70

export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_object_rearrangement.sh
