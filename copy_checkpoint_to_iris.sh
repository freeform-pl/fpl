#!/bin/bash
# Copy a single checkpoint from haic to iris-ws-7.
#
# Usage:
#   ./copy_checkpoint_to_iris.sh <folder_name> <checkpoint_number>
#
# Example:
#   ./copy_checkpoint_to_iris.sh setup_table_iter3_multi_qwen_1dp_iter3_3000 5000
#   ./copy_checkpoint_to_iris.sh plate_toast_iter0_multi_qwen_1dp_iter0_2000 30000
#   ./copy_checkpoint_to_iris.sh pick_and_place_iter1_multi_qwen_1dp_iter1_1700 30000
#   ./copy_checkpoint_to_iris.sh pick_and_place_iter1_single_qwen_1dp_iter1_single_1700 30000
#   ./copy_checkpoint_to_iris.sh setup_table_single_matching_1dp_iter1_3000 30000
#   ./copy_checkpoint_to_iris.sh burger_single_matching_1dp_iter1_1500 30000

# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/
# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/pick_and_place_iter1_single_qwen_v2_1dp_iter1_single_1700
# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/plate_toast_iter1_multi_qwen_1dp_iter1_5000
# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/plate_toast_iter1_single_qwen_1dp_iter1_5000
# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/pick_and_place_iter1_multi_qwen_v3_1dp_iter1_1700_new
# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/

#   ./copy_checkpoint_to_iris.sh pick_and_place_iter1_single_matching_1dp_iter1_single_matching_1500 25000
#   ./copy_checkpoint_to_iris.sh pick_and_place_iter1_single_qwen_v2_1dp_iter1_single_1700 30000
#   ./copy_checkpoint_to_iris.sh plate_toast_iter1_multi_qwen_1dp_iter1_5000 30000
#   ./copy_checkpoint_to_iris.sh plate_toast_iter1_single_qwen_1dp_iter1_5000 30000
# /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/
# Source:      /hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto/<folder>/<ckpt>
# Destination: iris-ws-7:/iris/u/marcelto/reward_learning_old/reward_learning/checkpoints/<folder>/<ckpt>
#
# Uses SSH ControlMaster so the password is only prompted ONCE.
# rsync -a is idempotent (resumes via --partial, skips already-transferred files).

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_name> <checkpoint_number>"
    echo "Example: $0 setup_table_iter3_multi_qwen_1dp_iter3_3000 5000"
    exit 1
fi

FOLDER="$1"
CKPT="$2"

SRC_BASE=/hai/scratch/marcelto/reward_learning/openpi/checkpoints/pi05_droid_finetune/pi05_marcelto
SRC_PATH="$SRC_BASE/$FOLDER/$CKPT"

DEST_USER_HOST=marcelto@iris-ws-7.stanford.edu
DEST_BASE=/iris/u/marcelto/reward_learning/checkpoints
DEST_PATH="$DEST_BASE/$FOLDER"

if [ ! -d "$SRC_PATH" ]; then
    echo "ERROR: source checkpoint does not exist: $SRC_PATH"
    echo "Available checkpoints in $SRC_BASE/$FOLDER:"
    ls "$SRC_BASE/$FOLDER" 2>/dev/null || echo "  (folder not found)"
    exit 1
fi

SOCK=/tmp/iris-cm-$$
cleanup() {
    ssh -S "$SOCK" -O exit "$DEST_USER_HOST" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Opening SSH master to $DEST_USER_HOST (enter password once) ==="
ssh -M -S "$SOCK" -fN "$DEST_USER_HOST"

RSH="ssh -S $SOCK"

echo ""
echo "=== Ensuring destination folder exists: $DEST_PATH ==="
ssh -S "$SOCK" "$DEST_USER_HOST" "mkdir -p '$DEST_PATH'"

echo ""
echo "=== Copying checkpoint $CKPT from $FOLDER ==="
echo "  src:  $SRC_PATH/"
echo "  dest: $DEST_USER_HOST:$DEST_PATH/$CKPT/"
rsync -avh --partial --progress -e "$RSH" "$SRC_PATH/" "$DEST_USER_HOST:$DEST_PATH/$CKPT/"

echo ""
echo "=== Done: $FOLDER/$CKPT copied to iris-ws-7 ==="
