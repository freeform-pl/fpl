#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=reward_cond_pipeline
#SBATCH --nodelist=iris4,iris5,iris6,iris7
#SBATCH --output slurm/%j.out

set -e

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"


export MUJOCO_PATH=~/.mujoco/mujoco210                                                                                             
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin                                                                    

conda activate robodiffrew2                                                       


# cd /iris/u/marcelto/reward_learning/diffusion_policy
# export HOME=/iris/u/marcelto
# eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
# conda activate robodiffrew2

# ============================================================
# Configuration — edit these paths before running
# ============================================================
POLICY_CKPT="/iris/u/marcelto/reward_learning/diffusion_policy/data/outputs/2026.05.02/12.21.55_train_flow_transformer_lowdim_square_lowdim/checkpoints/epoch=3600-test_mean_score=0.540.ckpt"
DEMO_HDF5="data/robomimic/datasets/square/mh/low_dim.hdf5"
N_ROLLOUTS=200
N_EVAL_ROLLOUTS=50
REWARD_EPOCHS=10
POLICY_EPOCHS=500

# Baseline: "rhp", "single_pref", "awr", "demo_success", "demo_only"
BASELINE=${BASELINE:-rhp}

SKIP_REWARD_MODEL=false
SKIP_POLICY_TRAINING=false
if [ "${BASELINE}" = "demo_only" ]; then
    SKIP_REWARD_MODEL=true
    SKIP_POLICY_TRAINING=true
    COND_CONFIG=""
fi

REWARD_AXES="success,speed_reward,smoothness"  # default axes

if [ "${BASELINE}" = "single_pref" ]; then
    REWARD_AXES="composite"
    NUM_REWARD_DIMS=1
    COND_CONFIG="train_reward_conditioned_flow_transformer_lowdim_workspace.yaml"
elif [ "${BASELINE}" = "awr" ]; then
    NUM_REWARD_DIMS=3
    COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
elif [ "${BASELINE}" = "demo_success" ]; then
    SKIP_REWARD_MODEL=true
    COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
else
    NUM_REWARD_DIMS=3
    COND_CONFIG="train_reward_conditioned_flow_transformer_lowdim_workspace.yaml"
fi

# Shared rollout data (collected once, reused across baselines)
SHARED_DIR="pipeline_output_shared"
ROLLOUT_PATH="${SHARED_DIR}/rollouts.npz"

PIPELINE_DIR="pipeline_output_${BASELINE}"
WANDB_PROJECT="reward_cond_${BASELINE}"

# Per-axis eval z-score conditioning (optional, arrays like "[1.5,1.5,1.5]")
EVAL_Z_POSITIVE=${EVAL_Z_POSITIVE:-}
EVAL_Z_NEGATIVE=${EVAL_Z_NEGATIVE:-}

# Build extra Hydra overrides for eval z-scores
EVAL_Z_OVERRIDES=""
if [ -n "${EVAL_Z_POSITIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} 'eval_z_positive=${EVAL_Z_POSITIVE}'"
fi
if [ -n "${EVAL_Z_NEGATIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} 'eval_z_negative=${EVAL_Z_NEGATIVE}'"
fi

# Resume from this phase (1=full run, 2=skip rollouts, 3=skip reward model, 4=skip policy training)
RESUME_FROM_PHASE=${RESUME_FROM_PHASE:-1}

REWARD_DIR="${PIPELINE_DIR}/reward_model"
SCORES_PATH="${REWARD_DIR}/scores.json"

# ============================================================
# Phase 1: Collect rollouts
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 1 ]; then
    if [ -f "${ROLLOUT_PATH}" ]; then
        echo "=== Phase 1: Rollouts already exist at ${ROLLOUT_PATH}, skipping collection ==="
    else
        echo "=== Phase 1: Collecting ${N_ROLLOUTS} rollouts ==="
        mkdir -p "${SHARED_DIR}"
        python scripts/collect_rollouts.py \
            --checkpoint "${POLICY_CKPT}" \
            --n_rollouts ${N_ROLLOUTS} \
            --output_path "${ROLLOUT_PATH}" \
            --wandb_project "${WANDB_PROJECT}"
    fi
else
    echo "=== Phase 1: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 2: Train reward model + score episodes
# ============================================================
if [ "${SKIP_REWARD_MODEL}" = "true" ]; then
    echo "=== Phase 2: SKIPPED (not needed for ${BASELINE}) ==="
elif [ ${RESUME_FROM_PHASE} -le 2 ]; then
    echo "=== Phase 2: Training reward model ==="
    python reward_model/train_reward_model.py \
        --rollout_data "${ROLLOUT_PATH}" \
        --demo_hdf5 "${DEMO_HDF5}" \
        --output_dir "${REWARD_DIR}" \
        --epochs ${REWARD_EPOCHS} \
        --wandb_project "${WANDB_PROJECT}" \
        --reward_axes "${REWARD_AXES}"
else
    echo "=== Phase 2: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 3: Train reward-conditioned policy
# ============================================================
if [ "${SKIP_POLICY_TRAINING}" = "true" ]; then
    echo "=== Phase 3: SKIPPED (not needed for ${BASELINE}) ==="
elif [ ${RESUME_FROM_PHASE} -le 3 ]; then
    echo "=== Phase 3: Training policy (${BASELINE}) ==="
    TRAIN_OUTPUT_DIR="${PIPELINE_DIR}/train_output_dir.txt"
    if [ "${BASELINE}" = "demo_success" ]; then
        python train.py \
            --config-name="${COND_CONFIG}" \
            task=square_lowdim \
            task.dataset_type=mh \
            rollout_data_path="${ROLLOUT_PATH}" \
            demo_hdf5_path="${DEMO_HDF5}" \
            training.num_epochs=${POLICY_EPOCHS} \
            logging.project="${WANDB_PROJECT}" \
            hydra.run.dir="${PIPELINE_DIR}/policy_outputv2"
    elif [ "${BASELINE}" = "awr" ]; then
        python train.py \
            --config-name="${COND_CONFIG}" \
            task=square_lowdim \
            task.dataset_type=mh \
            rollout_data_path="${ROLLOUT_PATH}" \
            scores_path="${SCORES_PATH}" \
            demo_hdf5_path="${DEMO_HDF5}" \
            training.num_epochs=${POLICY_EPOCHS} \
            logging.project="${WANDB_PROJECT}" \
            hydra.run.dir="${PIPELINE_DIR}/policy_outputv2"
    else
        eval python train.py \
            --config-name="${COND_CONFIG}" \
            task=square_lowdim \
            task.dataset_type=mh \
            num_reward_dims=${NUM_REWARD_DIMS} \
            rollout_data_path="${ROLLOUT_PATH}" \
            scores_path="${SCORES_PATH}" \
            demo_hdf5_path="${DEMO_HDF5}" \
            training.num_epochs=${POLICY_EPOCHS} \
            logging.project="${WANDB_PROJECT}" \
            hydra.run.dir="${PIPELINE_DIR}/policy_outputv2" \
            ${EVAL_Z_OVERRIDES}
    fi
else
    echo "=== Phase 3: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the checkpoint from this pipeline's output
COND_CKPT_DIR="${PIPELINE_DIR}/policy_output/checkpoints"
if [ "${BASELINE}" = "demo_only" ]; then
    COND_CKPT="${POLICY_CKPT}"
elif [ -f "${COND_CKPT_DIR}/latest.ckpt" ]; then
    COND_CKPT="${COND_CKPT_DIR}/latest.ckpt"
else
    echo "ERROR: Could not find policy checkpoint at ${COND_CKPT_DIR}/latest.ckpt"
    exit 1
fi
echo "Using checkpoint: ${COND_CKPT}"

# ============================================================
# Phase 4: Compare original vs conditioned policy
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 4 ]; then
    echo "=== Phase 4: Evaluation comparison ==="
    EVAL_ARGS="--original_ckpt ${POLICY_CKPT} --conditioned_ckpt ${COND_CKPT} --n_rollouts ${N_EVAL_ROLLOUTS} --output_dir ${PIPELINE_DIR}/eval --wandb_project ${WANDB_PROJECT}"
    if [ -f "${SCORES_PATH}" ]; then
        EVAL_ARGS="${EVAL_ARGS} --scores_path ${SCORES_PATH}"
    fi
    python scripts/eval_conditioned.py ${EVAL_ARGS}
else
    echo "=== Phase 4: SKIPPED ==="
fi

echo "=== Pipeline complete ==="
