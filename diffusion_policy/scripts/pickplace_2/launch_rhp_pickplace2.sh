#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp2_rhp
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# 5D reward: order + bread_placed + can_placed + bread_drop + can_drop.
export PIPELINE_DIR="pipeline_output_pickplace_2obj_fixed_rhp_raw"
export WANDB_PROJECT="pickplace_2obj_fixed_rhp"
export BASE_POLICY_DIR="base_policy_pickplace_2obj_fixed"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# 2-object variant: keep Bread + Can in the scene, clear Milk + Cereal.
export N_ACTIVE_OBJECTS=2
# 2 objects need ~700 control steps; 800 leaves slack for grasp retries.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"

# Preference-axis sampling for scripted demos
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=2
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 1000 demos, no rollouts
export N_SCRIPTED=500
export SKIP_ROLLOUTS=true

# export SHARED_DATA_DIR="shared_data_pickplace_2obj"
export SHARED_DATA_DIR="shared_data_pickplace_2obj_fixed_v2"

# 5D axes: order, per-object placed (bread/can), per-object drop (bread/can).
# ORDER MUST MATCH the values in CONDITIONING_TARGETS / EVAL_Z_* below.
# export REWARD_AXES="order_reward,bread_placed,can_placed,bread_drop,can_drop"
export REWARD_AXES="order_reward,bread_placed_raw,can_placed_raw,bread_drop_raw,can_drop_raw"
export NUM_REWARD_DIMS=5
export REWARD_EPOCHS=400
export COND_POLICY_EPOCHS=750
# Conditioning-noise augmentation. Adds uniform [-AUGMENT_SCORE, +AUGMENT_SCORE]
# noise to the appended reward dims at sample time and re-rounds to the same
# 0.1 buckets — so each (state, action) pair sometimes gets re-labeled with an
# adjacent bucket. 0.0 disables. 0.1 (a full bucket width) is the natural
# starting point; raise to broaden the conditioning support seen in training.
export AUGMENT_SCORE=0.2
# Score quantisation. True = round stored conditioning to 0.1 buckets (and
# re-round after augment noise). False = pass continuous z-scores through.
# Clipping to [-1, 1] always applies either way.
export ROUND_SCORES=False
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE} ++round_scores=${ROUND_SCORES}"

# Eval z-score conditioning. Positive = best on every axis, negative = worst.
export EVAL_Z_POSITIVE="[0.8,0.5,0.8,0.5,0.8]"
export EVAL_Z_NEGATIVE="[-0.8,0.7,0.8,-0.8,-0.8]"

export N_PAIRS=70

# Iterative refinement: sweep through how many objects placed while pinning
# order/drop positive. Rows are z-scores in REWARD_AXES order:
# [order, bread_p, can_p, bread_d, can_d].
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.8,-0.8,-0.8,0.8,0.8;0.8,0.8,-0.8,0.8,0.8;0.8,0.8,0.8,0.8,0.8"

export RESUME_FROM_PHASE=3
bash scripts/run_pipeline_pickplace.sh
