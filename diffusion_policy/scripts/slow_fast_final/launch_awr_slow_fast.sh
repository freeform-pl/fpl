#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_awr
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# AWR baseline for slow_fast. Uses the same 2-D reward axes as RHP but the
# AWR dataset averages them into a scalar advantage weight. No reward
# conditioning at eval.
export PIPELINE_DIR="pipeline_output_slow_fast_final_awr"
export WANDB_PROJECT="slow_fast_final_awr"
export BASE_POLICY_DIR="base_policy_slow_fast_final"
export COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=true

export SPEED_FACTOR_RANGE_LEFT="1 4"
export SPEED_FACTOR_RANGE_RIGHT="1 2"

export N_SCRIPTED=200
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_slow_fast_final"

export REWARD_AXES="speed_reward,peg_reward"
export NUM_REWARD_DIMS=2
export REWARD_EPOCHS=40
export COND_POLICY_EPOCHS=1500
# BATCH_SIZE / LEARNING_RATE intentionally left unset to use the AWR
# workspace YAML defaults (batch=256, lr=1e-4) — matches pickplace_2_final
# AWR which also uses the YAML defaults.
export TRAINING_SEED=42
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

export N_PAIRS=500

# Phase 3 trains reward model; Phase 4 trains AWR policy. No iterative refinement.
export RESUME_FROM_PHASE=3
bash scripts/run_pipeline_slow_fast.sh
