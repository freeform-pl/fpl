#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_recap
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RECAP-style baseline for slow_fast — Phase 3 trains V(s) on time-to-success,
# Phase 4 conditions the policy per-step on the standardized advantage
# A_t = V(s_{t+1}) - V(s_t) ∈ [-1, 1]. Eval queries a constant scalar.
export PIPELINE_DIR="pipeline_output_slow_fast_final_recap_slower"
export WANDB_PROJECT="slow_fast_final_recap"
export BASE_POLICY_DIR="base_policy_slow_fast_final"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# Swap Phase 3 to the value-function trainer.
export USE_VALUE_FUNCTION=true
export VALUE_FAIL_PENALTY=-100

export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_slow_fast_final_slower"

# Single advantage dim — REWARD_AXES is unused when USE_VALUE_FUNCTION=true
# but the pipeline still sources the var.
export REWARD_AXES="advantage"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=200
export COND_POLICY_EPOCHS=1500
export TRAINING_SEED=74
export AUGMENT_SCORE=0.1
export ROUND_SCORES=False
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE} ++round_scores=${ROUND_SCORES}"

# Eval at constant scalar conditioning: +1 = best advantage, -1 = worst.
export EVAL_Z_POSITIVE="[1.0]"
export EVAL_Z_NEGATIVE="[-1.0]"

# Iterative refinement on the single knob.
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="1.0;1.0;1.0"

export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_slow_fast.sh
