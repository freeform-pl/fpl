#!/bin/bash
#
# Per-rollout eval for the locally cached single_pref run (iu7gkh6h). Dumps
# first_success_step lists to
#   pipeline_output_slow_fast_final_single_pref_slower/eval/per_rollout_steps.jsonl
#
# Usage:
#   bash diffusion_policy/scripts/slow_fast_final/eval_per_rollout_single.sh
#   sbatch diffusion_policy/scripts/slow_fast_final/eval_per_rollout_single.sh
#
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_single_eval
#SBATCH --output=slurm/%j.out

set -eo pipefail

export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin"

CONDA_ROOT="/iris/u/marcelto/miniconda3"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate robodiffrew2

PYTHON="${CONDA_ROOT}/envs/robodiffrew2/bin/python"
"${PYTHON}" -c "import hydra; print(f'[env check] hydra {hydra.__version__} from {hydra.__file__}')"

PIPELINE_DIR="/iris/u/marcelto/reward_learning/diffusion_policy/pipeline_output_slow_fast_final_single_pref_slower"
EVAL_DIR="${PIPELINE_DIR}/eval"
SCORES_PATH="${PIPELINE_DIR}/reward_model/scores.json"
JSONL_OUT="${EVAL_DIR}/per_rollout_steps.jsonl"

# Eval config — keep consistent with launch_single_pref_slow_fast_slower.sh
N_ROLLOUTS=100
NUM_REWARD_DIMS=1
EVAL_Z_POSITIVE="[0.8]"
EVAL_Z_NEGATIVE="[-0.8]"
WANDB_PROJECT="slow_fast_final_single_pref"

CKPT=$(ls "${PIPELINE_DIR}/policy_output/checkpoints/"epoch=*-test_mean_score=*.ckpt \
       | awk -F'test_mean_score=' '{print $2 "\t" $0}' \
       | sort -k1,1 -rn | head -n1 | cut -f2-)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found under ${PIPELINE_DIR}/policy_output/checkpoints/"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

rm -f "${JSONL_OUT}"

cd /iris/u/marcelto/reward_learning/diffusion_policy

"${PYTHON}" scripts/eval_conditioned.py \
    --ckpt "${CKPT}" \
    --scores_path "${SCORES_PATH}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --num_reward_dims "${NUM_REWARD_DIMS}" \
    --eval_z_positive "${EVAL_Z_POSITIVE}" \
    --eval_z_negative "${EVAL_Z_NEGATIVE}" \
    --is_conditioned \
    --n_videos 0 \
    --output_dir "${EVAL_DIR}" \
    --wandb_project "${WANDB_PROJECT}"

echo
echo "Done. Per-rollout step dump at:"
echo "  ${JSONL_OUT}"
ls -la "${JSONL_OUT}" || true
