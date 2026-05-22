#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp_single_pref
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Single-pref baseline for the PickPlace 4-object benchmark.
# Composite scalar reward = average of order + per-object placed (milk/bread/
# cereal/can) + drop quality — all axes lumped into a single dim.
export PIPELINE_DIR="pipeline_output_pickplace_single_pref"
export WANDB_PROJECT="pickplace_single_pref"
export NUM_REWARD_DIMS=1
export REWARD_AXES="composite(order_reward+milk_placed+bread_placed+cereal_placed+can_placed+milk_drop+bread_drop+cereal_drop+can_drop)"

# Preference-axis settings (each demo rolls a side per axis)
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=4
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 200 demos, no rollouts (mirrors the peg less_preferences single_pref baseline)
export N_SCRIPTED=1000
export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_pickplace"

export EVAL_Z_POSITIVE="[0.7]"
export REWARD_EPOCHS=20

# Less preferences: 100 pairs instead of all pairs (mirrors single_pref baseline)
export N_PAIRS=1500

export RESUME_FROM_PHASE=1

bash scripts/run_pipeline_pickplace.sh
