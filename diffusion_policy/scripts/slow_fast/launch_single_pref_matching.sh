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
# speed_reward + peg_reward. Mirrors object_rearrangement single_pref structure.
export PIPELINE_DIR="pipeline_output_slow_fast_single_pref_matching"
export WANDB_PROJECT="slow_fast_single_pref_matching"
export BASE_POLICY_DIR="base_policy_slow_fast"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false


export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_slow_fast"

# Composite scalar: average of the 2 raw axes.
export REWARD_AXES="composite(speed_reward+peg_reward)"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=200
export COND_POLICY_EPOCHS=1500
# BATCH_SIZE / LEARNING_RATE left unset → use reward_conditioned workspace
# YAML defaults (batch=1024, lr=2e-4) — matches object_rearrangement single_pref.
export TRAINING_SEED=82
export AUGMENT_SCORE=0.2
export ROUND_SCORES=False
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE} ++round_scores=${ROUND_SCORES}"

export EVAL_Z_POSITIVE="[0.8]"
export EVAL_Z_NEGATIVE="[-0.8]"

export N_PAIRS=200

export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_slow_fast.sh
