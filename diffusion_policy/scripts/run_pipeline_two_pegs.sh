#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=two_pegs_pipeline
#SBATCH --nodelist=iris4,iris5,iris6,iris7
#SBATCH --output slurm/%j.out

set -e

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"

export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda activate robodiffrew2

# ============================================================
# Configuration
# ============================================================
N_SCRIPTED=50          # Number of scripted demos (left/right peg)
N_ROLLOUTS=200          # Number of rollouts from base policy
N_EVAL_ROLLOUTS=50
PIPELINE_DIR="pipeline_output_two_pegs"
BASE_POLICY_EPOCHS=5000
REWARD_EPOCHS=100
COND_POLICY_EPOCHS=5000
WANDB_PROJECT="two_pegs_pipeline"
NUM_REWARD_DIMS=4       # success, speed, smoothness, peg

# Noise range for scripted trajectory smoothness variation
NOISE_MIN=0.0
NOISE_MAX=0.12

# Per-axis eval z-score conditioning (optional, arrays like "[1.5,1.5,1.5,1.5]")
EVAL_Z_POSITIVE=${EVAL_Z_POSITIVE:-}
EVAL_Z_NEGATIVE=${EVAL_Z_NEGATIVE:-}

EVAL_Z_OVERRIDES=""
if [ -n "${EVAL_Z_POSITIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} 'eval_z_positive=${EVAL_Z_POSITIVE}'"
fi
if [ -n "${EVAL_Z_NEGATIVE}" ]; then
    EVAL_Z_OVERRIDES="${EVAL_Z_OVERRIDES} 'eval_z_negative=${EVAL_Z_NEGATIVE}'"
fi

# Resume from this phase (0=full run)
RESUME_FROM_PHASE=${RESUME_FROM_PHASE:-0}

SCRIPTED_DIR="${PIPELINE_DIR}/scripted_data"
SCRIPTED_HDF5="${SCRIPTED_DIR}/demos.hdf5"
ROLLOUT_PATH="${PIPELINE_DIR}/rollouts.npz"
REWARD_DIR="${PIPELINE_DIR}/reward_model"
SCORES_PATH="${REWARD_DIR}/scores.json"

# ============================================================
# Phase 0: Collect scripted demos (left/right peg, varying noise)
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 0 ]; then
    echo "=== Phase 0: Collecting ${N_SCRIPTED} scripted demos ==="
    python scripts/collect_initial_scripted_rollouts.py \
        -o "${SCRIPTED_DIR}" \
        -n ${N_SCRIPTED} \
        --noise_min ${NOISE_MIN} \
        --noise_max ${NOISE_MAX}
else
    echo "=== Phase 0: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 1: Train base policy on scripted demos
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 1 ]; then
    echo "=== Phase 1: Training base policy on scripted demos ==="
    python train.py \
        --config-name=train_flow_transformer_lowdim_workspace.yaml \
        task=square_twopeg_lowdim \
        task.dataset.dataset_path="${SCRIPTED_HDF5}" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        training.num_epochs=${BASE_POLICY_EPOCHS} \
        logging.project="${WANDB_PROJECT}"
else
    echo "=== Phase 1: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the best base policy checkpoint
BASE_CKPT=$(ls -t data/outputs/**/train_flow_transformer_lowdim_square_twopeg_lowdim/checkpoints/epoch=*-test_mean_score=*.ckpt 2>/dev/null | head -1)
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
if [ ${RESUME_FROM_PHASE} -le 3 ]; then
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
        --reward_axes "success,speed_reward,smoothness,peg_reward" \
        ${N_PAIRS_FLAG}
else
    echo "=== Phase 3: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 4: Train reward-conditioned policy (4 reward dims)
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 4 ]; then
    echo "=== Phase 4: Training reward-conditioned policy ==="
    eval python train.py \
        --config-name=train_reward_conditioned_flow_transformer_lowdim_workspace.yaml \
        task=square_twopeg_lowdim \
        num_reward_dims=${NUM_REWARD_DIMS} \
        rollout_data_path="${ROLLOUT_PATH}" \
        scores_path="${SCORES_PATH}" \
        demo_hdf5_path="${SCRIPTED_HDF5}" \
        training.num_epochs=${COND_POLICY_EPOCHS} \
        logging.project="${WANDB_PROJECT}" \
        hydra.run.dir="${PIPELINE_DIR}/policy_output" \
        task.env_runner.dataset_path="${SCRIPTED_HDF5}" \
        task.env_runner.use_twopeg_wrapper=True \
        ${EVAL_Z_OVERRIDES}
else
    echo "=== Phase 4: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the checkpoint from this pipeline's output
COND_CKPT_DIR="${PIPELINE_DIR}/policy_output/checkpoints"
if [ -f "${COND_CKPT_DIR}/latest.ckpt" ]; then
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
    python scripts/eval_conditioned.py \
        --original_ckpt "${BASE_CKPT}" \
        --conditioned_ckpt "${COND_CKPT}" \
        --scores_path "${SCORES_PATH}" \
        --n_rollouts ${N_EVAL_ROLLOUTS} \
        --output_dir "${PIPELINE_DIR}/eval" \
        --wandb_project "${WANDB_PROJECT}"
else
    echo "=== Phase 5: SKIPPED ==="
fi

echo "=== Pipeline complete ==="
