#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp2_sp_rnd
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Single-pref baseline for the PickPlace 2-object benchmark.
# Active objects: Bread + Can (first two in the right-first canonical order).
# Composite scalar reward = average of order + per-object placed (bread/can) +
# per-object drop (bread/can).
export PIPELINE_DIR="pipeline_output_pickplace_2obj_random_single_pref"
export WANDB_PROJECT="pickplace_2obj_fixed_single_pref"
export BASE_POLICY_DIR="base_policy_pickplace_2obj_random"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=false

# 2-object variant: keep Bread + Can in the scene, clear Milk + Cereal.
export N_ACTIVE_OBJECTS=2
# 2 objects need ~700 control steps; 800 leaves slack for grasp retries.
export EXTRA_POLICY_OVERRIDES="++task.env_runner.max_steps=500"

# Preference-axis sampling for scripted demos — CONTINUOUS variants:
# - DROP_MODE=random + the new collector uniformly samples drop_height in
#   [CAREFUL_HEIGHT, DROP_HEIGHT_MAX], so the release-height distribution
#   is continuous (no bimodal careful/drop gap).
# - RELEASE_XY_NOISE adds per-object xy noise on the release position so
#   *_placed_raw varies continuously. 0.10 m ≈ bin half-width; placements
#   often land in-bin but occasionally miss, bridging the placed/not-placed
#   gap in the raw reward distribution.
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=2
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.10   # min height that still counts as "drop" (for the discrete label)
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export RELEASE_XY_NOISE=0.20
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 500 demos, no rollouts — re-collect with the new continuous sampling.
export N_SCRIPTED=500
export SKIP_ROLLOUTS=true

# Fresh shared_data dir shared with launch_rhp_pickplace2_random.sh — both
# random variants use the same continuously-sampled demo set so comparisons
# stay apples-to-apples.
export SHARED_DATA_DIR="shared_data_pickplace_2obj_random"

# Single composite scalar reward: average of order, per-object placed
# (bread/can), and per-object drop (bread/can).
# export REWARD_AXES="composite(order_reward+bread_placed+can_placed+bread_drop+can_drop)"
export REWARD_AXES="composite(order_reward_raw+bread_placed_raw+can_placed_raw+bread_drop_raw+can_drop_raw)"
export NUM_REWARD_DIMS=1
export REWARD_EPOCHS=400
export COND_POLICY_EPOCHS=750
# Conditioning-noise augmentation. Adds uniform [-AUGMENT_SCORE, +AUGMENT_SCORE]
# noise to the appended reward dims at sample time and re-rounds to the same
# 0.1 buckets — so each (state, action) pair sometimes gets re-labeled with an
# adjacent bucket. 0.0 disables.
export AUGMENT_SCORE=0.2
# Rollout/eval frequency (every N epochs). Larger = faster training, fewer checkpoints.
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100 ++augment_score=${AUGMENT_SCORE}"

# Eval z-score conditioning. Positive = best composite reward.
export EVAL_Z_POSITIVE="[0.8]"
export EVAL_Z_NEGATIVE="[-0.8]"

export N_PAIRS=70

# Iterative refinement: composite scalar has a single knob, so each iteration
# pins it to the positive target. Iteration count matches the RHP baseline.
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.8;0.9;0.7;0.6;0.5"

# Phase 0 re-collects demos with the continuous random sampling. The shared
# data dir is the same as the RHP-random launcher — run one of the two with
# RESUME_FROM_PHASE=0 first to collect the demos, then the other can resume
# from later phases (RESUME_FROM_PHASE=1 or 3) without re-collecting.
export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_pickplace.sh
