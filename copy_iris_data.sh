#!/bin/bash
# Copy datasets from iris-ws-6 to /hai/scratch/marcelto/data.
# Excludes any file matching *_large*.hdf5.
# Uses SSH ControlMaster so the password is only prompted ONCE.
#
# Idempotent: rsync -a skips files that already exist with the same
# size + mtime (its default delta-transfer behavior). --partial keeps
# half-copied files around so the next run resumes them instead of
# starting over. Safe to re-run anytime.

set -e

SOCK=/tmp/iris-cm-$$
SRC=marcelto@iris-ws-6.stanford.edu
EXC='--exclude=*_large*.hdf5'
DEST=/hai/scratch/marcelto/data

cleanup() {
    ssh -S "$SOCK" -O exit "$SRC" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Opening SSH master to $SRC (enter password once) ==="
ssh -M -S "$SOCK" -fN "$SRC"

RSH="ssh -S $SOCK"
RSYNC="rsync -avh --partial --progress -e $RSH $EXC"

# Pairs of "src_remote_path  dest_local_path"
PAIRS=(
    "/iris/u/am208/droid-robot/preferences_setup/             $DEST/am208/droid-robot/preferences_setup/"
    "/iris/u/am208/droid-robot/cross_preferences_setup/       $DEST/am208/droid-robot/cross_preferences_setup/"
    "/iris/u/abhijnya/droid-robot/cross_preferences_setup/    $DEST/abhijnya/droid-robot/cross_preferences_setup/"
    "/iris/u/abhijnya/droid-robot/demos/table_setup/          $DEST/abhijnya/droid-robot/demos/table_setup/"
    "/iris/u/am208/droid-robot/preferences/                   $DEST/am208/droid-robot/preferences/"
    "/iris/u/am208/droid-robot/cross_preferences/             $DEST/am208/droid-robot/cross_preferences/"
    "/iris/u/am208/droid-robot/cross_preferences_burger/      $DEST/am208/droid-robot/cross_preferences_burger/"
    "/iris/u/abhijnya/droid-robot/cross_preferences/          $DEST/abhijnya/droid-robot/cross_preferences/"
    "/iris/u/am208/droid-robot/demos/test/                    $DEST/am208/droid-robot/demos/test/"
    "/iris/u/abhijnya/droid-robot/demos/test/                 $DEST/abhijnya/droid-robot/demos/test/"
    "/iris/u/am208/droid-robot/demos/setup/                   $DEST/am208/droid-robot/demos/setup/"
    "/iris/u/am208/droid-robot/demos/burger/                  $DEST/am208/droid-robot/demos/burger/"
    "/iris/u/am208/droid-robot/cross_preferences_pick_and_place/    $DEST/am208/droid-robot/cross_preferences_pick_and_place/"
    "/iris/u/am208/droid-robot/preferences_pick_and_place     $DEST/am208/droid-robot/preferences_pick_and_place/"
    "/iris/u/am208/droid-robot/demos/pick_and_place           $DEST/am208/droid-robot/demos/pick_and_place/"
)

i=1
total=${#PAIRS[@]}
for pair in "${PAIRS[@]}"; do
    read -r remote local <<< "$pair"
    mkdir -p "$local"
    echo ""
    echo "=== [$i/$total] $remote -> $local ==="
    rsync -avh --partial --progress -e "$RSH" $EXC "$SRC:$remote" "$local"
    i=$((i + 1))
done

echo ""
echo "=== All $total transfers complete ==="
