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
PIPELINE_DIR="pipeline_output"
REWARD_EPOCHS=100
POLICY_EPOCHS=5000
WANDB_PROJECT="reward_cond_pipeline"

# Resume from this phase (1=full run, 2=skip rollouts, 3=skip reward model, 4=skip policy training)
RESUME_FROM_PHASE=${RESUME_FROM_PHASE:-1}
RESUME_FROM_PHASE=3

ROLLOUT_PATH="${PIPELINE_DIR}/rollouts.npz"
REWARD_DIR="${PIPELINE_DIR}/reward_model"
SCORES_PATH="${REWARD_DIR}/scores.json"

# ============================================================
# Phase 1: Collect rollouts
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 1 ]; then
    echo "=== Phase 1: Collecting ${N_ROLLOUTS} rollouts ==="
    python scripts/collect_rollouts.py \
        --checkpoint "${POLICY_CKPT}" \
        --n_rollouts ${N_ROLLOUTS} \
        --output_path "${ROLLOUT_PATH}" \
        --wandb_project "${WANDB_PROJECT}"
else
    echo "=== Phase 1: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 2: Train reward model + score episodes
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 2 ]; then
    echo "=== Phase 2: Training reward model ==="
    python reward_model/train_reward_model.py \
        --rollout_data "${ROLLOUT_PATH}" \
        --demo_hdf5 "${DEMO_HDF5}" \
        --output_dir "${REWARD_DIR}" \
        --epochs ${REWARD_EPOCHS} \
        --wandb_project "${WANDB_PROJECT}"
else
    echo "=== Phase 2: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# ============================================================
# Phase 3: Train reward-conditioned policy
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 3 ]; then
    echo "=== Phase 3: Training reward-conditioned policy ==="
    python train.py \
        --config-name=train_reward_conditioned_flow_transformer_lowdim_workspace.yaml \
        task=square_lowdim \
        task.dataset_type=mh \
        rollout_data_path="${ROLLOUT_PATH}" \
        scores_path="${SCORES_PATH}" \
        demo_hdf5_path="${DEMO_HDF5}" \
        training.num_epochs=${POLICY_EPOCHS} \
        logging.project="${WANDB_PROJECT}"
else
    echo "=== Phase 3: SKIPPED (resuming from phase ${RESUME_FROM_PHASE}) ==="
fi

# Find the latest conditioned checkpoint
COND_CKPT=$(ls -t data/outputs/**/train_reward_conditioned_*/checkpoints/latest.ckpt 2>/dev/null | head -1)
if [ -z "${COND_CKPT}" ]; then
    echo "ERROR: Could not find conditioned policy checkpoint"
    exit 1
fi
echo "Using conditioned checkpoint: ${COND_CKPT}"

# ============================================================
# Phase 4: Compare original vs conditioned policy
# ============================================================
if [ ${RESUME_FROM_PHASE} -le 4 ]; then
    echo "=== Phase 4: Evaluation comparison ==="
    python scripts/eval_conditioned.py \
        --original_ckpt "${POLICY_CKPT}" \
        --conditioned_ckpt "${COND_CKPT}" \
        --scores_path "${SCORES_PATH}" \
        --n_rollouts ${N_EVAL_ROLLOUTS} \
        --output_dir "${PIPELINE_DIR}/eval" \
        --wandb_project "${WANDB_PROJECT}"
else
    echo "=== Phase 4: SKIPPED ==="
fi

echo "=== Pipeline complete ==="
