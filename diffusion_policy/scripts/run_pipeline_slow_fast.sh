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

# Helper: find best checkpoint in a directory by test_mean_score in filename
find_best_ckpt() {
    local ckpt_dir="$1"
    local best_ckpt=""
    local best_score=""
    for f in "${ckpt_dir}"/epoch=*-test_mean_score=*.ckpt; do
        [ -f "$f" ] || continue
        score=$(echo "$f" | grep -oP 'test_mean_score=\K[0-9]+\.[0-9]+')
        [ -z "$score" ] && continue
        if [ -z "$best_score" ] || (( $(echo "$score > $best_score" | bc -l) )); then
            best_score="$score"
            best_ckpt="$f"
        fi
    done
    echo "$best_ckpt"
}

# ============================================================
# Configuration
# ============================================================
N_SCRIPTED=${N_SCRIPTED:-50}
N_ROLLOUTS=${N_ROLLOUTS:-200}
N_EVAL_ROLLOUTS=${N_EVAL_ROLLOUTS:-50}
PIPELINE_DIR=${PIPELINE_DIR:-"pipeline_output_slow_fastv2"}
BASE_POLICY_EPOCHS=${BASE_POLICY_EPOCHS:-5000}
REWARD_EPOCHS=${REWARD_EPOCHS:-50}
COND_POLICY_EPOCHS=${COND_POLICY_EPOCHS:-500}
# Diffusion-policy action chunking — matches pickplace defaults (h=16/o=2/a=8).
# n_action_steps=8 cuts inference frequency 8× at eval; horizon=16 gives the
# model more lookahead structure.
N_OBS_STEPS=${N_OBS_STEPS:-2}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
HORIZON=${HORIZON:-16}
# Eval rollout cap (`task.env_runner.max_steps`). 500 leaves room for the
# slowest scripted demos (which reach ~475 steps under speed_factor=4) while
# still discouraging endlessly-slow policies.
MAX_STEPS=${MAX_STEPS:-500}
WANDB_PROJECT=${WANDB_PROJECT:-"slow_fast_pipeline"}
NUM_REWARD_DIMS=${NUM_REWARD_DIMS:-4}       # speed, smoothness, peg
REWARD_AXES=${REWARD_AXES:-"success,speed_reward,smoothness,peg_reward"}
COND_CONFIG=${COND_CONFIG:-"train_reward_conditioned_flow_transformer_lowdim_workspace.yaml"}
SKIP_REWARD_MODEL=${SKIP_REWARD_MODEL:-false}
SKIP_POLICY_TRAINING=${SKIP_POLICY_TRAINING:-false}
IS_CONDITIONED_EVAL=${IS_CONDITIONED_EVAL:-true}
USE_BEST_CKPT=${USE_BEST_CKPT:-true}
SKIP_ROLLOUTS=${SKIP_ROLLOUTS:-false}
DISCRETE_CONDITIONING=${DISCRETE_CONDITIONING:-false}
TARGET_PEG=${TARGET_PEG:-random}
EXTRA_POLICY_OVERRIDES=${EXTRA_POLICY_OVERRIDES:-}
EXTRA_BASE_POLICY_OVERRIDES=${EXTRA_BASE_POLICY_OVERRIDES:-}
# Phase 4 (conditioned policy) hyperparameter overrides. Empty = use YAML default.
BATCH_SIZE=${BATCH_SIZE:-}
LEARNING_RATE=${LEARNING_RATE:-}
# Phase 1 (base policy) hyperparameter overrides — independent knobs.
BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-}
BASE_LEARNING_RATE=${BASE_LEARNING_RATE:-}
# Training seed overrides; same value goes to `training.seed` (model init,
# optimizer, dataloader) AND `task.dataset.seed` (train/val split + pair sampling).
TRAINING_SEED=${TRAINING_SEED:-}
BASE_TRAINING_SEED=${BASE_TRAINING_SEED:-}

# Noise range for scripted trajectory smoothness variation
NOISE_MIN=${NOISE_MIN:-0.0}
NOISE_MAX=${NOISE_MAX:-0.12}

# Per-axis eval conditioning targets (scores normalized to [-1, 1])
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

SHARED_DATA_DIR=${SHARED_DATA_DIR:-"shared_data_slow_fast"}
SCRIPTED_DIR="${SHARED_DATA_DIR}/scripted_data"
SCRIPTED_HDF5="${SCRIPTED_DIR}/demos.hdf5"
ROLLOUT_PATH="${SHARED_DATA_DIR}/rollouts.npz"
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
        --target_peg ${TARGET_PEG}
else
    echo "=== Phase 0: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 1: Train base policy on scripted demos
# ============================================================
BASE_POLICY_DIR=${BASE_POLICY_DIR:-"base_policy_slow_fast"}
BASE_CKPT_DIR="${BASE_POLICY_DIR}/checkpoints"
if [ "${SKIP_ROLLOUTS}" = "true" ]; then
    echo "=== Phase 1: SKIPPED (no rollouts mode — using demos only) ==="
elif [ ${RESUME_FROM_PHASE} -le 1 ]; then
    echo "=== Phase 1: Training base policy on scripted demos ==="
    BASE_OVERRIDES=""
    if [ -n "${BASE_BATCH_SIZE}" ]; then
        BASE_OVERRIDES="${BASE_OVERRIDES} dataloader.batch_size=${BASE_BATCH_SIZE} val_dataloader.batch_size=${BASE_BATCH_SIZE}"
    fi
    if [ -n "${BASE_LEARNING_RATE}" ]; then
        BASE_OVERRIDES="${BASE_OVERRIDES} optimizer.learning_rate=${BASE_LEARNING_RATE}"
    fi
    if [ -n "${BASE_TRAINING_SEED}" ]; then
        BASE_OVERRIDES="${BASE_OVERRIDES} training.seed=${BASE_TRAINING_SEED} task.dataset.seed=${BASE_TRAINING_SEED}"
    fi
    if [ -n "${EXTRA_BASE_POLICY_OVERRIDES}" ]; then
        BASE_OVERRIDES="${BASE_OVERRIDES} ${EXTRA_BASE_POLICY_OVERRIDES}"
    fi
    eval python train.py \
        --config-name=train_flow_transformer_lowdim_workspace.yaml \
        task=square_twopeg_lowdim \
        task.dataset.dataset_path="${SCRIPTED_HDF5}" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        training.num_epochs=${BASE_POLICY_EPOCHS} \
        training.resume=False \
        logging.resume=False \
        n_obs_steps=${N_OBS_STEPS} \
        n_action_steps=${N_ACTION_STEPS} \
        horizon=${HORIZON} \
        ++task.env_runner.max_steps=${MAX_STEPS} \
        logging.project="${WANDB_PROJECT}" \
        hydra.run.dir="${BASE_POLICY_DIR}" \
        ${BASE_OVERRIDES}
else
    echo "=== Phase 1: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the best base policy checkpoint (not needed if skipping rollouts)
if [ "${SKIP_ROLLOUTS}" != "true" ]; then
    BASE_CKPT=$(find_best_ckpt "${BASE_CKPT_DIR}")
    if [ -z "${BASE_CKPT}" ]; then
        if [ -f "${BASE_CKPT_DIR}/latest.ckpt" ]; then
            BASE_CKPT="${BASE_CKPT_DIR}/latest.ckpt"
        else
            echo "ERROR: Could not find base policy checkpoint in ${BASE_CKPT_DIR}"
            exit 1
        fi
    fi
    echo "Using base policy checkpoint: ${BASE_CKPT}"
fi

# ============================================================
# Phase 2: Collect rollouts from base policy
# ============================================================
if [ "${SKIP_ROLLOUTS}" = "true" ]; then
    echo "=== Phase 2: SKIPPED (no rollouts mode — demos only) ==="
    ROLLOUT_PATH="none"
elif [ ${RESUME_FROM_PHASE} -le 2 ]; then
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
    if [ "${USE_VALUE_FUNCTION}" = "true" ]; then
        echo "=== Phase 3: Training value function (RECAP-style) ==="
        python reward_model/train_value_function.py \
            --rollout_data "${ROLLOUT_PATH}" \
            --demo_hdf5 "${SCRIPTED_HDF5}" \
            --output_dir "${REWARD_DIR}" \
            --task slow_fast \
            --epochs ${REWARD_EPOCHS} \
            --wandb_project "${WANDB_PROJECT}" \
            --fail_penalty "${VALUE_FAIL_PENALTY:--100}"
    else
        echo "=== Phase 3: Training reward model ==="
        N_PAIRS_FLAG=""
        if [ -n "${N_PAIRS:-}" ]; then
            N_PAIRS_FLAG="--n_pairs ${N_PAIRS}"
        fi
        python reward_model/train_reward_model.py \
            --rollout_data "${ROLLOUT_PATH}" \
            --demo_hdf5 "${SCRIPTED_HDF5}" \
            --output_dir "${REWARD_DIR}" \
            --epochs ${REWARD_EPOCHS} \
            --wandb_project "${WANDB_PROJECT}" \
            --reward_axes "${REWARD_AXES}" \
            ${N_PAIRS_FLAG}
    fi
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
    EXTRA_OVERRIDES=""
    if [ "${SKIP_REWARD_MODEL}" != "true" ]; then
        EXTRA_OVERRIDES="++num_reward_dims=${NUM_REWARD_DIMS} ++scores_path=${SCORES_PATH}"
    fi
    if [ "${IS_CONDITIONED_EVAL}" = "true" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ++task.env_runner.use_twopeg_wrapper=True"
    fi
    if [ "${DISCRETE_CONDITIONING}" = "true" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ++discrete_conditioning=True"
    fi
    if [ -n "${BATCH_SIZE}" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} dataloader.batch_size=${BATCH_SIZE} val_dataloader.batch_size=${BATCH_SIZE}"
    fi
    if [ -n "${LEARNING_RATE}" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} optimizer.learning_rate=${LEARNING_RATE}"
    fi
    if [ -n "${TRAINING_SEED}" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} training.seed=${TRAINING_SEED} task.dataset.seed=${TRAINING_SEED}"
    fi
    if [ -n "${EXTRA_POLICY_OVERRIDES}" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ${EXTRA_POLICY_OVERRIDES}"
    fi
    eval python train.py \
        --config-name="${COND_CONFIG}" \
        task=square_twopeg_lowdim \
        rollout_data_path="${ROLLOUT_PATH}" \
        demo_hdf5_path="${SCRIPTED_HDF5}" \
        training.num_epochs=${COND_POLICY_EPOCHS} \
        training.resume=False \
        logging.resume=False \
        n_obs_steps=${N_OBS_STEPS} \
        n_action_steps=${N_ACTION_STEPS} \
        horizon=${HORIZON} \
        ++task.env_runner.max_steps=${MAX_STEPS} \
        logging.project="${WANDB_PROJECT}" \
        hydra.run.dir="${PIPELINE_DIR}/policy_output" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        ${EXTRA_OVERRIDES} \
        ${EVAL_Z_OVERRIDES}
else
    echo "=== Phase 4: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the checkpoint from this pipeline's output
COND_CKPT_DIR="${PIPELINE_DIR}/policy_output/checkpoints"
if [ "${SKIP_POLICY_TRAINING}" = "true" ]; then
    COND_CKPT="${BASE_CKPT}"
elif [ "${USE_BEST_CKPT}" = "true" ]; then
    COND_CKPT=$(find_best_ckpt "${COND_CKPT_DIR}")
    if [ -z "${COND_CKPT}" ]; then
        echo "WARNING: No best ckpt found, falling back to latest.ckpt"
        COND_CKPT="${COND_CKPT_DIR}/latest.ckpt"
    fi
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
    EVAL_ARGS="--ckpt ${COND_CKPT} --n_rollouts ${N_EVAL_ROLLOUTS} --output_dir ${PIPELINE_DIR}/eval --wandb_project ${WANDB_PROJECT} --num_reward_dims ${NUM_REWARD_DIMS}"
    EVAL_ARGS="${EVAL_ARGS} --n_videos ${N_EVAL_VIDEOS:-3}"
    if [ "${IS_CONDITIONED_EVAL}" = "true" ]; then
        EVAL_ARGS="${EVAL_ARGS} --is_conditioned"
    fi
    if [ -f "${SCORES_PATH}" ]; then
        EVAL_ARGS="${EVAL_ARGS} --scores_path ${SCORES_PATH}"
    fi
    if [ -n "${EVAL_Z_POSITIVE}" ]; then
        EVAL_ARGS="${EVAL_ARGS} --eval_z_positive ${EVAL_Z_POSITIVE}"
    fi
    if [ -n "${EVAL_Z_NEGATIVE}" ]; then
        EVAL_ARGS="${EVAL_ARGS} --eval_z_negative ${EVAL_Z_NEGATIVE}"
    fi
    python scripts/eval_conditioned.py ${EVAL_ARGS}
else
    echo "=== Phase 5: SKIPPED ==="
fi

echo "=== Pipeline complete ==="
