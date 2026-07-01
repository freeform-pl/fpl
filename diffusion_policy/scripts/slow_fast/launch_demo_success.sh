#!/bin/bash

# Demo-success (success-only) baseline for slow_fast. Trains a plain flow
# transformer on the subset of demos+rollouts that succeeded. No reward
# model, no conditioning at eval.
export PIPELINE_DIR="pipeline_output_slow_fast_demo_success"
export WANDB_PROJECT="slow_fast_demo_success"
export BASE_POLICY_DIR="base_policy_slow_fast"
export COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
export SKIP_REWARD_MODEL=true
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true


# demo_success needs Phase 2 rollouts to filter successes; keep enabled.
export SKIP_ROLLOUTS=false

export SHARED_DATA_DIR="shared_data_slow_fast"

export REWARD_AXES="speed_reward,peg_reward"
export NUM_REWARD_DIMS=2
export BASE_POLICY_EPOCHS=1500
export COND_POLICY_EPOCHS=1500
# Match FPL/single_pref hyperparams for both Phase 1 (base) and Phase 4 (demo_success).
export BATCH_SIZE=1024
export LEARNING_RATE=2e-4
export BASE_BATCH_SIZE=1024
export BASE_LEARNING_RATE=2e-4
export TRAINING_SEED=42
export BASE_TRAINING_SEED=32
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++task.dataset.filter_to_right_peg=True"
# Phase 1 eval cadence needs its own override — EXTRA_POLICY_OVERRIDES only
# flows into Phase 4. Workspace YAML default for Phase 1 is every 50 epochs.
export EXTRA_BASE_POLICY_OVERRIDES="++training.rollout_every=100 ++training.checkpoint_every=100"

# Phase 1 trains base policy, Phase 2 collects rollouts, Phase 4 trains
# demo_success on filtered demos+rollouts. Phase 3 skipped (no reward model).
export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_slow_fast.sh
