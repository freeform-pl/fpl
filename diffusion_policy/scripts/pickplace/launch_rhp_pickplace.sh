#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp_rhp
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# RHP baseline for the PickPlace 4-object benchmark.
# 6D reward (order + per-object placed for milk/bread/cereal/can + drop),
# DISCRETE conditioning, iterative refinement across diverse conditioning targets.
export PIPELINE_DIR="pipeline_output_pickplace_rhp"
export WANDB_PROJECT="pickplace_rhp"
export IS_CONDITIONED_EVAL=true
export DISCRETE_CONDITIONING=true

# Preference-axis sampling for scripted demos (mirrors single_pref launch)
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=4
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 200 demos, no rollouts
export N_SCRIPTED=1000
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_pickplace"

# 9D axes: order, per-object placed (4), per-object drop (4).
# ORDER MUST MATCH the values in CONDITIONING_TARGETS / EVAL_Z_* below.
export REWARD_AXES="order_reward,milk_placed,bread_placed,cereal_placed,can_placed,milk_drop,bread_drop,cereal_drop,can_drop"
export NUM_REWARD_DIMS=9
export REWARD_EPOCHS=40

# Eval z-score conditioning (normalized scores).
# Positive = best on every axis (canonical / all 4 placed / all careful).
# Negative = reversed / no objects placed / all dropped.
export EVAL_Z_POSITIVE="[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]"
export EVAL_Z_NEGATIVE="[-0.9,-0.9,-0.9,-0.9,-0.9,-0.9,-0.9,-0.9,-0.9]"

# Less preferences: 100 pairs instead of all pairs.
export N_PAIRS=500

# Iterative refinement: sweep through "how many objects placed" combinations
# while pinning order positive and all per-object drop at +0.9 (careful).
# Each row of CONDITIONING_TARGETS lists z-scores for the 9 axes in REWARD_AXES
# order: [order, milk_p, bread_p, cereal_p, can_p, milk_d, bread_d, cereal_d, can_d].
export N_ITERATIONS=3
export N_ITER_ROLLOUTS=200
export CONDITIONING_TARGETS="0.9,-0.9,-0.9,-0.9,-0.9,0.9,0.9,0.9,0.9;0.9,0.9,-0.9,-0.9,-0.9,0.9,0.9,0.9,0.9;0.9,0.9,0.9,-0.9,-0.9,0.9,0.9,0.9,0.9;0.9,0.9,0.9,0.9,-0.9,0.9,0.9,0.9,0.9;0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9"

export RESUME_FROM_PHASE=0
bash scripts/run_pipeline_pickplace.sh
