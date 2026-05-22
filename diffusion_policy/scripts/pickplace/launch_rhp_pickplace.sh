#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp4_rhp
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for the PickPlace 4-object benchmark.
# Active objects: Bread + Can + Milk + Cereal (right-first canonical order).
# 9D reward: order + per-object placed (4) + per-object drop (4).
export PIPELINE_DIR="pipeline_output_pickplace_4obj_fixed_rhp"
export WANDB_PROJECT="pickplace_4obj_fixed_rhp"
export BASE_POLICY_DIR="base_policy_pickplace_4obj_fixed"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# 4-object variant: all four objects active in the scene.
export N_ACTIVE_OBJECTS=4
# 4 objects need ~1400 control steps; 1000 is the per-object-proportional
# analogue of the 2-obj launcher's 500 — raise if rollouts time out.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=1000"

# Preference-axis sampling for scripted demos
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=4
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 1000 demos, no rollouts
export N_SCRIPTED=1000
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_pickplace_4obj_fixed"

# 9D axes (canonical right-first order: bread, can, milk, cereal).
# ORDER MUST MATCH the values in CONDITIONING_TARGETS / EVAL_Z_* below.
export REWARD_AXES="order_reward,bread_placed,can_placed,milk_placed,cereal_placed,bread_drop,can_drop,milk_drop,cereal_drop"
export NUM_REWARD_DIMS=9
export REWARD_EPOCHS=40
export COND_POLICY_EPOCHS=1500
# Conditioning-noise augmentation. Adds uniform [-AUGMENT_SCORE, +AUGMENT_SCORE]
# noise to the appended reward dims at sample time and re-rounds to the same
# 0.1 buckets — so each (state, action) pair sometimes gets re-labeled with an
# adjacent bucket. 0.0 disables.
export AUGMENT_SCORE=0.2
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE}"

# Eval z-score conditioning. Positive = best on every axis, negative = worst.
# Layout: [order, bread_p, can_p, milk_p, cereal_p, bread_d, can_d, milk_d, cereal_d].
export EVAL_Z_POSITIVE="[0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8]"
export EVAL_Z_NEGATIVE="[-0.8,0.8,0.8,0.8,0.8,-0.8,-0.8,-0.8,-0.8]"

export N_PAIRS=500

# Iterative refinement: sweep through how many objects placed (in canonical
# order: bread, can, milk, cereal) while pinning order / drop positive.
# Rows are z-scores in REWARD_AXES order.
export N_ITERATIONS=5
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.8,-0.8,-0.8,-0.8,-0.8,0.8,0.8,0.8,0.8;0.8,0.8,-0.8,-0.8,-0.8,0.8,0.8,0.8,0.8;0.8,0.8,0.8,-0.8,-0.8,0.8,0.8,0.8,0.8;0.8,0.8,0.8,0.8,-0.8,0.8,0.8,0.8,0.8;0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8"

export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_pickplace.sh
