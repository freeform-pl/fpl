#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp4_demo_only
#SBATCH --nodelist=iris7,iris8,iris10
#SBATCH --output slurm/%j.out

# Demo-only baseline for the PickPlace 4-object benchmark.
# Active objects: Bread + Can + Milk + Cereal (right-first canonical order).
# Evaluates the pre-trained base policy directly — no reward model, no AWR
# weighting, no conditioning. Pure BC on the full demo set.
export PIPELINE_DIR="pipeline_output_pickplace_4obj_fixed_demo_only"
export WANDB_PROJECT="pickplace_4obj_fixed_demo_only"
export BASE_POLICY_DIR="base_policy_pickplace_4obj_fixed"
export SKIP_REWARD_MODEL=true
export SKIP_POLICY_TRAINING=true
export IS_CONDITIONED_EVAL=false

# 4-object variant: all four objects active in the scene.
export N_ACTIVE_OBJECTS=4

# Preference-axis sampling for scripted demos (only used if Phase 0 runs)
export ORDER_MODE=random
export N_OBJECTS_MIN=1
export N_OBJECTS_MAX=4
export DROP_MODE=random
export DROP_HEIGHT_MIN=0.15
export DROP_HEIGHT_MAX=0.20
export CAREFUL_HEIGHT=0.04
export NOISE_MIN=0.0
export NOISE_MAX=0.05

# 1000 demos (already collected, just defines metadata).
export N_SCRIPTED=1000
# SKIP_ROLLOUTS=false keeps the BASE_CKPT discovery block active so Phase 5
# can find the pre-trained base policy. Phase 2 itself is skipped by RESUME.
export SKIP_ROLLOUTS=false

export SHARED_DATA_DIR="shared_data_pickplace_4obj_fixed"

# Per-axis eval logging — reads scripts/run_pipeline_pickplace.sh's defaults.
export REWARD_AXES="order_reward,bread_placed,can_placed,milk_placed,cereal_placed,bread_drop,can_drop,milk_drop,cereal_drop"
export NUM_REWARD_DIMS=9

# Phase 5 only: evaluate BASE_CKPT directly. Phases 0–4 all skipped.
export RESUME_FROM_PHASE=1
bash scripts/run_pipeline_pickplace.sh
