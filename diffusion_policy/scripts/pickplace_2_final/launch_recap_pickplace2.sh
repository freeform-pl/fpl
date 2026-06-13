#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp2_recap
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RECAP-style baseline for the PickPlace 2-object benchmark.
# Phase 3 trains a value function V(s) that regresses "time-to-success" with
# r_t = 0 at terminal-success, r_t = fail_penalty at terminal-failure, r_t = -1
# otherwise. The policy is then conditioned per-step on the standardized
# advantage A_t = V(s_{t+1}) - V(s_t) ∈ [-1, 1]. At eval we query a constant
# scalar in [-1, 1].
export PIPELINE_DIR="pipeline_output_pickplace_2obj_fixed_recap"
export WANDB_PROJECT="pickplace_2obj_fixed_recap"
export BASE_POLICY_DIR="base_policy_pickplace_2obj_fixed"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# Swap Phase 3 to the value-function trainer.
export USE_VALUE_FUNCTION=true
export VALUE_FAIL_PENALTY=-100

# 2-object variant: keep Bread + Can in the scene, clear Milk + Cereal.
export N_ACTIVE_OBJECTS=2
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"

# Preference-axis sampling for scripted demos (matches the other PP2 launchers).
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=2
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

export N_SCRIPTED=300
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_pickplace_2obj_fixed_v2"

# Single advantage dim from the value function — REWARD_AXES is unused when
# USE_VALUE_FUNCTION=true but the var is still required by the pipeline.
export REWARD_AXES="advantage"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=400
export COND_POLICY_EPOCHS=750

export TRAINING_SEED=74

# Augmentation on the per-step advantage at sample time (same role as for RHP /
# single_pref). 0.0 disables.
export AUGMENT_SCORE=0.1
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE}"

# Eval at constant scalar conditioning: +1 = best advantage, -1 = worst.
export EVAL_Z_POSITIVE="[1.0]"
export EVAL_Z_NEGATIVE="[-1.0]"

# Iterative refinement on the single advantage knob.
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="1.0;1.0;1.0"

export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_pickplace.sh
