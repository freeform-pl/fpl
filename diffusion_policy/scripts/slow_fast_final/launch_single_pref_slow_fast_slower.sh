#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_single_pref
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Single-pref baseline for slow_fast — composite scalar reward = mean of
# speed_reward + peg_reward. Mirrors pickplace_2_final single_pref structure.
export PIPELINE_DIR="pipeline_output_slow_fast_final_single_pref_slower"
export WANDB_PROJECT="slow_fast_final_single_pref"
export BASE_POLICY_DIR="base_policy_slow_fast_final"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_slow_fast_final_slower"

# Composite scalar: average of the 2 raw axes.
export REWARD_AXES="composite(speed_reward+peg_reward)"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=200
export COND_POLICY_EPOCHS=1500
# BATCH_SIZE / LEARNING_RATE left unset → use reward_conditioned workspace
# YAML defaults (batch=1024, lr=2e-4) — matches pickplace_2_final single_pref.
export TRAINING_SEED=62
export AUGMENT_SCORE=0.2
export ROUND_SCORES=False
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE} ++round_scores=${ROUND_SCORES}"

export EVAL_Z_POSITIVE="[0.8]"
export EVAL_Z_NEGATIVE="[-0.8]"

export N_PAIRS=100

# Iterative refinement: composite scalar has a single knob.
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.8;0.9;0.7"

export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_slow_fast.sh
