#!/bin/bash

# Demo-only baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# Evaluates the pre-trained base policy directly — no reward model, no AWR
# weighting, no conditioning. Pure BC on the full demo set.
export PIPELINE_DIR="pipeline_output_object_rearrangement_demo_only"
export WANDB_PROJECT="object_rearrangement_demo_only"
export BASE_POLICY_DIR="base_policy_object_rearrangement"
export SKIP_REWARD_MODEL=true
export SKIP_POLICY_TRAINING=true
export IS_CONDITIONED_EVAL=false


# SKIP_ROLLOUTS=false keeps the BASE_CKPT discovery block active so Phase 5
# can find the pre-trained base policy. Phase 2 itself is skipped by RESUME.
export SKIP_ROLLOUTS=false

export SHARED_DATA_DIR="shared_data_object_rearrangement"

export BASE_POLICY_EPOCHS=750
export COND_POLICY_EPOCHS=750
# Training seed for the base policy (Phase 1 — the only training that
# happens here since SKIP_POLICY_TRAINING=true skips Phase 4). Applied to
# both `training.seed` and `task.dataset.seed`. Leave unset for YAML default.
export BASE_TRAINING_SEED=62
# Phase 1 (base policy) eval/checkpoint frequency. With SKIP_POLICY_TRAINING=true
# the base policy IS the policy we evaluate, so rolling out more often gives a
# denser learning curve. Affects only Phase 1.
export EXTRA_BASE_POLICY_OVERRIDES="++training.rollout_every=100 ++training.checkpoint_every=100"

# Per-axis eval logging — reads scripts/run_pipeline_object_rearrangement.sh's defaults.
export REWARD_AXES="order_reward,bread_placed,can_placed,bread_drop,can_drop"
export NUM_REWARD_DIMS=5

# Phase 5 only: evaluate BASE_CKPT directly. Phases 0–4 all skipped.
export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_object_rearrangement.sh
