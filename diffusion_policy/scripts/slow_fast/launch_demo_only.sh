#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_demo_only
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Demo-only baseline for slow_fast. Evaluates the base policy directly — no
# reward model, no AWR weighting, no conditioning. Pure BC on the full demos.
export PIPELINE_DIR="pipeline_output_slow_fast_demo_only"
export WANDB_PROJECT="slow_fast_demo_only"
export BASE_POLICY_DIR="base_policy_slow_fast"
export SKIP_REWARD_MODEL=true
export SKIP_POLICY_TRAINING=true
export IS_CONDITIONED_EVAL=false


# 200 demos, skip rollout collection but still train base policy.
export SKIP_ROLLOUTS=false

export SHARED_DATA_DIR="shared_data_slow_fast"

export REWARD_AXES="speed_reward,peg_reward"
export NUM_REWARD_DIMS=2
export BASE_POLICY_EPOCHS=1500
# BASE_BATCH_SIZE / BASE_LEARNING_RATE intentionally left unset to use the
# base-policy workspace YAML defaults (batch=256, lr=1e-4) — matches
# object_rearrangement demo_only which also uses the YAML defaults.
export BASE_TRAINING_SEED=42
# Phase 1 (base policy) eval/checkpoint frequency. With SKIP_POLICY_TRAINING=true
# the base policy IS the policy we evaluate, so rolling out more often gives a
# denser learning curve. Affects only Phase 1.
export EXTRA_BASE_POLICY_OVERRIDES="++training.rollout_every=100 ++training.checkpoint_every=100"

# Phase 1 trains base policy. Phase 4 skipped. Phase 5 evaluates BASE_CKPT.
export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_slow_fast.sh
