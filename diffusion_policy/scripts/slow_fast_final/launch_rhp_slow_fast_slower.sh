#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_rhp
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for slow_fast — mirrors the pickplace_2_final RHP structure:
# continuous conditioning, augment_score noise, raw z-scores (no rounding),
# exposed seed/batch/lr knobs.
# Left peg: speed [1, 4] (fast). Right peg: speed [1, 2] (slow).
export PIPELINE_DIR="pipeline_output_slow_fast_final_rhp_slower"
export WANDB_PROJECT="slow_fast_final_rhp"
export BASE_POLICY_DIR="base_policy_slow_fast_final"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="3 4"

# 200 demos, no rollouts (demos-only mode)
export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_slow_fast_final_slower"

# 2D axes: speed + peg
export REWARD_AXES="speed_reward,peg_reward"
export NUM_REWARD_DIMS=2
export REWARD_EPOCHS=200
export COND_POLICY_EPOCHS=1500
# BATCH_SIZE / LEARNING_RATE left unset → use reward_conditioned workspace
# YAML defaults (batch=1024, lr=2e-4) — matches pickplace_2_final RHP.
# Training seed (Phase 4 conditioned policy). Same value applied to
# `training.seed` AND `task.dataset.seed`.
export TRAINING_SEED=62
# Conditioning-noise augmentation. Uniform [-AUGMENT_SCORE, +AUGMENT_SCORE]
# applied to the appended reward dims at sample time so the policy doesn't
# overfit to specific bucket centres.
export AUGMENT_SCORE=0.2
# Score quantisation. False = pass continuous z-scores through (clipping to
# [-1, 1] always applies). True = round to 0.1 buckets at construction + after
# augment noise.
export ROUND_SCORES=False
# Rollout/eval frequency.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=20 ++training.checkpoint_every=20 ++augment_score=${AUGMENT_SCORE} ++round_scores=${ROUND_SCORES}"

# Eval z-score conditioning.
export EVAL_Z_POSITIVE="[0.9,0.8]"
export EVAL_Z_NEGATIVE="[0.9,-0.8]"

export N_PAIRS=100

# Iterative refinement: speed_reward in [0.5, 0.9], peg_reward fixed at 0.9.
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.5,0.9;0.6,0.9;0.7,0.9;0.8,0.9;0.9,0.9"

# Phase 3 trains the reward model in THIS pipeline dir.
export RESUME_FROM_PHASE=7
bash scripts/run_pipeline_slow_fast.sh
