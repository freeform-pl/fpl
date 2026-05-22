#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pickplace_pipeline
#SBATCH --nodelist=iris9,iris10
#SBATCH --output slurm/%j.out

set -e

if [ -d /iris/u/marcelto/miniconda3 ]; then
    CONDA_ROOT=/iris/u/marcelto/miniconda3
elif [ -d /hai/scratch/marcelto/miniconda3 ]; then
    CONDA_ROOT=/hai/scratch/marcelto/miniconda3
else
    echo "ERROR: could not find miniconda on /iris or /hai" >&2
    exit 1
fi
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"

export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda activate robodiffrew2

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
N_SCRIPTED=${N_SCRIPTED:-200}
N_ROLLOUTS=${N_ROLLOUTS:-200}
N_EVAL_ROLLOUTS=${N_EVAL_ROLLOUTS:-50}
PIPELINE_DIR=${PIPELINE_DIR:-"pipeline_output_pickplace"}
BASE_POLICY_EPOCHS=${BASE_POLICY_EPOCHS:-5000}
REWARD_EPOCHS=${REWARD_EPOCHS:-50}
# PickPlace episodes are 1300+ steps but max_seq_len in the reward model is
# 512; stride=4 spans the whole trajectory at one token per 4 control steps.
REWARD_MODEL_STRIDE=${REWARD_MODEL_STRIDE:-4}
COND_POLICY_EPOCHS=${COND_POLICY_EPOCHS:-500}
WANDB_PROJECT=${WANDB_PROJECT:-"pickplace_pipeline"}
NUM_REWARD_DIMS=${NUM_REWARD_DIMS:-9}
# Axes available in rollouts.npz: success, speed_reward, smoothness,
#                                 order_reward, milk_placed/bread_placed/
#                                 cereal_placed/can_placed (per-object placed),
#                                 milk_drop/bread_drop/cereal_drop/can_drop
#                                 (per-object careful=+1 vs drop=-1, 0 if
#                                 not attempted), drop_reward (aggregated).
REWARD_AXES=${REWARD_AXES:-"order_reward,milk_placed,bread_placed,cereal_placed,can_placed,milk_drop,bread_drop,cereal_drop,can_drop"}
COND_CONFIG=${COND_CONFIG:-"train_reward_conditioned_flow_transformer_lowdim_workspace.yaml"}
SKIP_REWARD_MODEL=${SKIP_REWARD_MODEL:-false}
SKIP_POLICY_TRAINING=${SKIP_POLICY_TRAINING:-false}
IS_CONDITIONED_EVAL=${IS_CONDITIONED_EVAL:-true}
USE_BEST_CKPT=${USE_BEST_CKPT:-true}
SKIP_ROLLOUTS=${SKIP_ROLLOUTS:-true}     # PickPlace base policy is hard; default to demos-only.
DISCRETE_CONDITIONING=${DISCRETE_CONDITIONING:-false}
EXTRA_POLICY_OVERRIDES=${EXTRA_POLICY_OVERRIDES:-}

# Per-axis preference settings
ORDER_MODE=${ORDER_MODE:-random}                  # canonical | reversed | random
N_OBJECTS_MIN=${N_OBJECTS_MIN:-1}
N_OBJECTS_MAX=${N_OBJECTS_MAX:-4}
DROP_MODE=${DROP_MODE:-random}                    # careful | drop | random
DROP_HEIGHT_MIN=${DROP_HEIGHT_MIN:-0.15}
DROP_HEIGHT_MAX=${DROP_HEIGHT_MAX:-0.20}
CAREFUL_HEIGHT=${CAREFUL_HEIGHT:-0.04}
NOISE_MIN=${NOISE_MIN:-0.0}
NOISE_MAX=${NOISE_MAX:-0.05}
SPEED_FACTOR=${SPEED_FACTOR:-1.0}
MAX_STEPS=${MAX_STEPS:-2000}
QUADRANT_NOISE=${QUADRANT_NOISE:-0.03}
SETTLE_STEPS=${SETTLE_STEPS:-40}
MAX_GRASP_ATTEMPTS=${MAX_GRASP_ATTEMPTS:-3}
# Per-step xy jitter (m) on transit waypoints during scripted demo collection.
# Adds path-level diversity so the BC policy doesn't memorize a single route.
# Never applied during CLOSE / RELEASE / LOWER / regrasp.
PATH_JITTER=${PATH_JITTER:-0.05}
# Subset of the 4 PickPlace objects to keep in the scene. 4 = full task.
# 2 = first two in the right-first canonical order (Bread + Can). Inactive
# objects are cleared out of the bin by the env wrapper.
N_ACTIVE_OBJECTS=${N_ACTIVE_OBJECTS:-4}
# PickPlace episodes are long (~1300 control steps); a larger action chunk
# speeds up rollout/eval ~8x. Keep horizon >= n_obs_steps + n_action_steps - 1.
N_OBS_STEPS=${N_OBS_STEPS:-2}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
HORIZON=${HORIZON:-16}

# Eval z-score conditioning (length must match NUM_REWARD_DIMS)
EVAL_Z_POSITIVE=${EVAL_Z_POSITIVE:-"[1.0,1.0,1.0,1.0]"}
EVAL_Z_NEGATIVE=${EVAL_Z_NEGATIVE:-}

EVAL_Z_OVERRIDES=""
if [ -n "${EVAL_Z_POSITIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} '++eval_z_positive=${EVAL_Z_POSITIVE}'"
fi
if [ -n "${EVAL_Z_NEGATIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} '++eval_z_negative=${EVAL_Z_NEGATIVE}'"
fi

RESUME_FROM_PHASE=${RESUME_FROM_PHASE:-0}

SHARED_DATA_DIR=${SHARED_DATA_DIR:-"shared_data_pickplace"}
SCRIPTED_DIR="${SHARED_DATA_DIR}/scripted_data"
SCRIPTED_HDF5="${SCRIPTED_DIR}/demos.hdf5"
ROLLOUT_PATH="${SHARED_DATA_DIR}/rollouts.npz"
REWARD_DIR="${PIPELINE_DIR}/reward_model"
SCORES_PATH="${REWARD_DIR}/scores.json"

# Print resolved paths so it's obvious where each run is reading/writing.
echo "============================================================"
echo "PickPlace pipeline paths (n_active_objects=${N_ACTIVE_OBJECTS}):"
echo "  SHARED_DATA_DIR : ${SHARED_DATA_DIR}"
echo "  SCRIPTED_HDF5   : ${SCRIPTED_HDF5}"
echo "  ROLLOUT_PATH    : ${ROLLOUT_PATH}"
echo "  PIPELINE_DIR    : ${PIPELINE_DIR}"
echo "  REWARD_DIR      : ${REWARD_DIR}"
echo "  BASE_POLICY_DIR : ${BASE_POLICY_DIR:-base_policy_pickplace (default)}"
echo "============================================================"

# ============================================================
# Phase 0: Collect scripted demos
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 0 ]; then
    echo "=== Phase 0: Collecting ${N_SCRIPTED} scripted PickPlace demos ==="
    python scripts/collect_initial_scripted_rollouts_pickplace.py \
        -o "${SCRIPTED_DIR}" \
        -n ${N_SCRIPTED} \
        --order_mode "${ORDER_MODE}" \
        --n_objects_min ${N_OBJECTS_MIN} \
        --n_objects_max ${N_OBJECTS_MAX} \
        --drop_mode "${DROP_MODE}" \
        --drop_height_min ${DROP_HEIGHT_MIN} \
        --drop_height_max ${DROP_HEIGHT_MAX} \
        --careful_height ${CAREFUL_HEIGHT} \
        --noise_min ${NOISE_MIN} \
        --noise_max ${NOISE_MAX} \
        --speed_factor ${SPEED_FACTOR} \
        --max_steps ${MAX_STEPS} \
        --quadrant_noise ${QUADRANT_NOISE} \
        --settle_steps ${SETTLE_STEPS} \
        --max_grasp_attempts ${MAX_GRASP_ATTEMPTS} \
        --n_active_objects ${N_ACTIVE_OBJECTS} \
        --path_jitter ${PATH_JITTER}

    # Copy aggregated rollouts.npz to the shared data root for the reward model.
    cp "${SCRIPTED_DIR}/rollouts.npz" "${ROLLOUT_PATH}" || true
else
    echo "=== Phase 0: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 1: Train base policy on scripted demos (optional)
# ============================================================
# Derive a per-variant default from SHARED_DATA_DIR so 2-obj and 4-obj base
# policies don't write into the same dir when SKIP_ROLLOUTS=false.
BASE_POLICY_DIR=${BASE_POLICY_DIR:-"base_policy_${SHARED_DATA_DIR}"}
BASE_CKPT_DIR="${BASE_POLICY_DIR}/checkpoints"
if [ "${SKIP_ROLLOUTS}" = "true" ]; then
    echo "=== Phase 1: SKIPPED (no rollouts mode — using demos only) ==="
elif [ ${RESUME_FROM_PHASE} -le 1 ]; then
    echo "=== Phase 1: Training base policy on scripted demos ==="
    python train.py \
        --config-name=train_flow_transformer_lowdim_workspace.yaml \
        task=pickplace_4obj_lowdim \
        task.dataset.dataset_path="${SCRIPTED_HDF5}" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        task.env_runner.n_active_objects=${N_ACTIVE_OBJECTS} \
        training.num_epochs=${BASE_POLICY_EPOCHS} \
        training.resume=False \
        logging.resume=False \
        n_obs_steps=${N_OBS_STEPS} \
        n_action_steps=${N_ACTION_STEPS} \
        horizon=${HORIZON} \
        logging.project="${WANDB_PROJECT}" \
        hydra.run.dir="${BASE_POLICY_DIR}"
else
    echo "=== Phase 1: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

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
    echo "=== Phase 2: SKIPPED (demos only) ==="
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
    echo "=== Phase 3: SKIPPED ==="
elif [ ${RESUME_FROM_PHASE} -le 3 ]; then
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
        --stride ${REWARD_MODEL_STRIDE} \
        ${N_PAIRS_FLAG}
else
    echo "=== Phase 3: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 4: Train reward-conditioned policy
# ============================================================
if [ "${SKIP_POLICY_TRAINING}" = "true" ]; then
    echo "=== Phase 4: SKIPPED ==="
elif [ ${RESUME_FROM_PHASE} -le 4 ]; then
    echo "=== Phase 4: Training conditioned policy ==="
    EXTRA_OVERRIDES="++task.env_runner.use_pickplace_wrapper=True ++task.env_runner.n_active_objects=${N_ACTIVE_OBJECTS} ++task.dataset.n_active_objects=${N_ACTIVE_OBJECTS}"
    if [ "${SKIP_REWARD_MODEL}" != "true" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ++num_reward_dims=${NUM_REWARD_DIMS} ++scores_path=${SCORES_PATH}"
    fi
    if [ "${DISCRETE_CONDITIONING}" = "true" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ++discrete_conditioning=True"
    fi
    if [ -n "${EXTRA_POLICY_OVERRIDES}" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ${EXTRA_POLICY_OVERRIDES}"
    fi
    eval python train.py \
        --config-name="${COND_CONFIG}" \
        task=pickplace_4obj_lowdim \
        rollout_data_path="${ROLLOUT_PATH}" \
        demo_hdf5_path="${SCRIPTED_HDF5}" \
        training.num_epochs=${COND_POLICY_EPOCHS} \
        training.resume=False \
        logging.resume=False \
        n_obs_steps=${N_OBS_STEPS} \
        n_action_steps=${N_ACTION_STEPS} \
        horizon=${HORIZON} \
        logging.project="${WANDB_PROJECT}" \
        hydra.run.dir="${PIPELINE_DIR}/policy_output" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        ${EXTRA_OVERRIDES} \
        ${EVAL_Z_OVERRIDES}
else
    echo "=== Phase 4: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

COND_CKPT_DIR="${PIPELINE_DIR}/policy_output/checkpoints"
if [ "${SKIP_POLICY_TRAINING}" = "true" ]; then
    COND_CKPT="${BASE_CKPT:-}"
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
