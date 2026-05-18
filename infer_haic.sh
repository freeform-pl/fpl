#!/bin/bash
#SBATCH --account=models
#SBATCH --partition=hai
#SBATCH --qos=models
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=hopper
#SBATCH --job-name=reward_infer
#SBATCH --output=slurm/%j.out

# haic launcher for infer.py. Mirrors train_qwen_discounted_haic.sh: detects
# whether we're on iris or haic and resolves data paths accordingly.

if [ -d /iris/u/marcelto ]; then
    REWARD_LEARNING_DIR=/iris/u/marcelto/reward_learning
    CONDA_ROOT=/iris/u/marcelto/miniconda3
    DATA_AM208=/iris/u/am208/droid-robot
    DATA_ABHIJNYA=/iris/u/abhijnya/droid-robot
    OUTPUT_ROOT=/iris/u/marcelto/reward_learning/infer_output
    export HF_LEROBOT_HOME=/iris/u/marcelto/data
    export HOME=/iris/u/marcelto
elif [ -d /hai/scratch/marcelto ]; then
    REWARD_LEARNING_DIR=/hai/scratch/marcelto/reward_learning
    CONDA_ROOT=/hai/scratch/marcelto/miniconda3
    DATA_AM208=/hai/scratch/marcelto/data/am208/droid-robot
    DATA_ABHIJNYA=/hai/scratch/marcelto/data/abhijnya/droid-robot
    OUTPUT_ROOT=/hai/scratch/marcelto/reward_learning/infer_output
    export HF_LEROBOT_HOME=/hai/scratch/marcelto/data
else
    echo "ERROR: neither /iris/u/marcelto nor /hai/scratch/marcelto present" >&2
    exit 1
fi

cd "$REWARD_LEARNING_DIR"
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
conda activate qwen310

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== Python: $(which python) ==="

CKPT=exp/2026-05-17_00-26-30_qwen_open_j77850/checkpoints/step002000.pt
RUN_TAG=2026-05-17_00-26-30_qwen_open_j77850_2000
OUTPUT_SUBDIR=setup_table_iter3_single_qwen
# Convert step runs in openpi's venv (matches the lerobot/datasets versions
# that openpi will read with at train time — qwen310 has a newer `datasets`
# that emits feature types openpi can't parse).
OPENPI_PY="$REWARD_LEARNING_DIR/openpi/.venv/bin/python"


python infer.py \
    --ckpt "$CKPT" \
    --preferences_dir "${DATA_AM208}/preferences_setup,${DATA_ABHIJNYA}/demos/table_setup" \
    --output_dir "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_TAG}"

"$OPENPI_PY" convert_custom_droid_to_lerobot.py \
    --args.scores_dir "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_TAG}" \
    --args.repo_name "marcelto/${OUTPUT_SUBDIR}_1dp_iter3_2000" \
    --args.task_prompt "set up the table" \
    --args.score_type standardized \
    --args.decimal_places 1

# CKPT=exp/2026-05-17_00-26-30_qwen_open_j77850/checkpoints/step002000.pt
# RUN_TAG=2026-05-17_00-26-30_qwen_open_j77850_2000
# OUTPUT_SUBDIR=setup_table_iter3_single_qwen
    

# python infer.py \
#     --ckpt "$CKPT" \
#     --preferences_dir "${DATA_AM208}/preferences_setup,${DATA_ABHIJNYA}/demos/table_setup" \
#     --output_dir "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_TAG}" \
#     --task setup_table_reduced

# "$OPENPI_PY" convert_custom_droid_to_lerobot.py \
#     --args.scores_dir "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_TAG}" \
#     --args.repo_name "marcelto/${OUTPUT_SUBDIR}_1dp_iter3_2000" \
#     --args.task_prompt "set up the table" \
#     --args.score_type standardized \
#     --args.decimal_places 1
