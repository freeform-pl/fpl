#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=slow_fast_pipeline
#SBATCH --nodelist=iris9,iris10
#SBATCH --output slurm/%j.out

set -e

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"

export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda activate robodiffrew2

# ============================================================
# Configuration
# ============================================================
N_SCRIPTED=50           # Number of scripted demos (left/right peg)
N_ROLLOUTS=200          # Number of rollouts from base policy
N_EVAL_ROLLOUTS=50
PIPELINE_DIR=${PIPELINE_DIR:-"pipeline_output_slow_fastv2"}
BASE_POLICY_EPOCHS=500
REWARD_EPOCHS=10
COND_POLICY_EPOCHS=500
WANDB_PROJECT=${WANDB_PROJECT:-"slow_fast_pipeline"}
NUM_REWARD_DIMS=${NUM_REWARD_DIMS:-3}       # speed, smoothness, peg
REWARD_AXES=${REWARD_AXES:-"speed_reward,smoothness,peg_reward"}
COND_CONFIG=${COND_CONFIG:-"train_reward_conditioned_flow_transformer_lowdim_workspace.yaml"}
SKIP_REWARD_MODEL=${SKIP_REWARD_MODEL:-false}
SKIP_POLICY_TRAINING=${SKIP_POLICY_TRAINING:-false}

# Noise range for scripted trajectory smoothness variation
NOISE_MIN=0.0
NOISE_MAX=0.12

# Speed factors: left peg = fast, right peg = slow
SPEED_FACTOR_LEFT=0.6
SPEED_FACTOR_RIGHT=2.0

# Per-axis eval z-score conditioning (optional, arrays like "[1.5,1.5,1.5,1.5]")
EVAL_Z_POSITIVE=${EVAL_Z_POSITIVE:-"[1.0,1.0,1.0]"}
EVAL_Z_NEGATIVE=${EVAL_Z_NEGATIVE:-}

EVAL_Z_OVERRIDES=""
if [ -n "${EVAL_Z_POSITIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} '++eval_z_positive=${EVAL_Z_POSITIVE}'"
fi
if [ -n "${EVAL_Z_NEGATIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} '++eval_z_negative=${EVAL_Z_NEGATIVE}'"
fi

# Resume from this phase (0=full run)
RESUME_FROM_PHASE=${RESUME_FROM_PHASE:-0}

SCRIPTED_DIR="${PIPELINE_DIR}/scripted_data"
SCRIPTED_HDF5="${SCRIPTED_DIR}/demos.hdf5"
ROLLOUT_PATH="${PIPELINE_DIR}/rollouts.npz"
REWARD_DIR="${PIPELINE_DIR}/reward_model"
SCORES_PATH="${REWARD_DIR}/scores.json"

# ============================================================
# Phase 0: Collect scripted demos (left=fast, right=slow)
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 0 ]; then
    echo "=== Phase 0: Collecting ${N_SCRIPTED} scripted demos (left=fast, right=slow) ==="
    python scripts/collect_initial_scripted_rollouts.py \
        -o "${SCRIPTED_DIR}" \
        -n ${N_SCRIPTED} \
        --noise_min ${NOISE_MIN} \
        --noise_max ${NOISE_MAX} \
        --speed_factor_left ${SPEED_FACTOR_LEFT} \
        --speed_factor_right ${SPEED_FACTOR_RIGHT}
else
    echo "=== Phase 0: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 1: Train base policy on scripted demos
# ============================================================
# if [ ${RESUME_FROM_PHASE} -le 1 ]; then
#     echo "=== Phase 1: Training base policy on scripted demos ==="
#     python train.py \
#         --config-name=train_flow_transformer_lowdim_workspace.yaml \
#         task=square_twopeg_lowdim \
#         task.dataset.dataset_path="${SCRIPTED_HDF5}" \
#         task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
#         training.num_epochs=${BASE_POLICY_EPOCHS} \
#         logging.project="${WANDB_PROJECT}"
# else
#     echo "=== Phase 1: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
# fi
echo "=== Phase 1: SKIPPED reusing pretrained ckpt ==="

# Find the best base policy checkpoint
BASE_CKPT="/iris/u/marcelto/reward_learning/diffusion_policy/data/outputs/2026.05.05/14.14.35_train_flow_transformer_lowdim_square_twopeg_lowdim/checkpoints/epoch=4600-test_mean_score=0.800.ckpt"
if [ -z "${BASE_CKPT}" ]; then
    echo "ERROR: Could not find base policy checkpoint"
    exit 1
fi
echo "Using base policy checkpoint: ${BASE_CKPT}"

# ============================================================
# Phase 2: Collect rollouts from base policy
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 2 ]; then
    echo "=== Phase 2: Collecting ${N_ROLLOUTS} rollouts from base policy ==="
    python scripts/collect_rollouts.py \
        --checkpoint "${BASE_CKPT}" \
        --n_rollouts ${N_ROLLOUTS} \
        --output_path "${ROLLOUT_PATH}" \
        --wandb_project "${WANDB_PROJECT}"
else
    echo "=== Phase 2: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 3: Train reward model on rollouts + score episodes
# ============================================================
if [ "${SKIP_REWARD_MODEL}" = "true" ]; then
    echo "=== Phase 3: SKIPPED (not needed for this baseline) ==="
elif [ ${RESUME_FROM_PHASE} -le 3 ]; then
    echo "=== Phase 3: Training reward model ==="
    python reward_model/train_reward_model.py \
        --rollout_data "${ROLLOUT_PATH}" \
        --demo_hdf5 "${SCRIPTED_HDF5}" \
        --output_dir "${REWARD_DIR}" \
        --epochs ${REWARD_EPOCHS} \
        --wandb_project "${WANDB_PROJECT}" \
        --reward_axes "${REWARD_AXES}"
else
    echo "=== Phase 3: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 4: Train reward-conditioned policy (5 reward dims)
# ============================================================
if [ "${SKIP_POLICY_TRAINING}" = "true" ]; then
    echo "=== Phase 4: SKIPPED (not needed for this baseline) ==="
elif [ ${RESUME_FROM_PHASE} -le 4 ]; then
    echo "=== Phase 4: Training policy ==="
    REWARD_OVERRIDES=""
    if [ "${SKIP_REWARD_MODEL}" != "true" ]; then
        REWARD_OVERRIDES="++num_reward_dims=${NUM_REWARD_DIMS} ++scores_path=${SCORES_PATH}"
    fi
    eval python train.py \
        --config-name="${COND_CONFIG}" \
        task=square_twopeg_lowdim \
        rollout_data_path="${ROLLOUT_PATH}" \
        demo_hdf5_path="${SCRIPTED_HDF5}" \
        training.num_epochs=${COND_POLICY_EPOCHS} \
        logging.project="${WANDB_PROJECT}" \
        hydra.run.dir="${PIPELINE_DIR}/policy_output" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        ++task.env_runner.use_twopeg_wrapper=True \
        ${REWARD_OVERRIDES} \
        ${EVAL_Z_OVERRIDES}
else
    echo "=== Phase 4: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the checkpoint from this pipeline's output
COND_CKPT_DIR="${PIPELINE_DIR}/policy_output/checkpoints"
if [ "${SKIP_POLICY_TRAINING}" = "true" ]; then
    COND_CKPT="${BASE_CKPT}"
elif [ -f "${COND_CKPT_DIR}/latest.ckpt" ]; then
    COND_CKPT="${COND_CKPT_DIR}/latest.ckpt"
else
    echo "ERROR: Could not find conditioned policy checkpoint at ${COND_CKPT_DIR}/latest.ckpt"
    exit 1
fi
echo "Using conditioned checkpoint: ${COND_CKPT}"

# ============================================================
# Phase 5: Evaluate conditioned policy
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 5 ]; then
    echo "=== Phase 5: Evaluation ==="
    EVAL_ARGS="--original_ckpt ${BASE_CKPT} --conditioned_ckpt ${COND_CKPT} --n_rollouts ${N_EVAL_ROLLOUTS} --output_dir ${PIPELINE_DIR}/eval --wandb_project ${WANDB_PROJECT}"
    if [ -f "${SCORES_PATH}" ]; then
        EVAL_ARGS="${EVAL_ARGS} --scores_path ${SCORES_PATH}"
    fi
    python scripts/eval_conditioned.py ${EVAL_ARGS}
else
    echo "=== Phase 5: SKIPPED ==="
fi

echo "=== Pipeline complete ==="
