#!/bin/bash

# Demo-success (success-only) baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# Trains a plain flow transformer on the subset of demos that fully placed
# every active object. No reward model, no conditioning at eval.
export PIPELINE_DIR="pipeline_output_object_rearrangement_demo_success"
export WANDB_PROJECT="object_rearrangement_demo_success"
export BASE_POLICY_DIR="base_policy_object_rearrangement"
export COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
export SKIP_REWARD_MODEL=true
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true

# 2 objects need ~700 control steps; 800 leaves slack for grasp retries.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"


export SKIP_ROLLOUTS=false

# export SHARED_DATA_DIR="shared_data_object_rearrangement"
export SHARED_DATA_DIR="shared_data_object_rearrangement"

# Reward axes are unused (SKIP_REWARD_MODEL=true) but kept for consistency
# with the env_runner's per-axis logging at eval time.
export REWARD_AXES="order_reward,bread_placed,can_placed,bread_drop,can_drop"
export NUM_REWARD_DIMS=5
export BASE_POLICY_EPOCHS=750
export COND_POLICY_EPOCHS=750
# Match FPL / single_pref hyperparameters (workspace YAML defaults for
# demo_success and the base-policy workspace are batch=256 / lr=1e-4; here
# we bump to batch=1024 / lr=2e-4 to keep the comparison apples-to-apples).
# Same override is applied to both Phase 1 (base policy) and Phase 4
# (demo_success policy) since demo_success trains both.
export BATCH_SIZE=1024
export LEARNING_RATE=2e-4
export BASE_BATCH_SIZE=1024
export BASE_LEARNING_RATE=2e-4
# Training seeds. demo_success trains both Phase 1 (base policy) and Phase 4
# (demo_success policy), so both seeds matter. Both default to 42 in the
# workspace YAMLs; override here to run additional independent seeds.
export TRAINING_SEED=62
export BASE_TRAINING_SEED=42
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

# Skip reward-model phase; jump straight to policy training on the filtered demos.
export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_object_rearrangement.sh
