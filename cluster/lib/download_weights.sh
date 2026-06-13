#!/bin/bash
# Thin wrapper: activate conda env, run the Python downloader.
# Run from the LOGIN node (compute nodes typically lack outbound internet).
#
# Usage (from /scratch300/$USER/dflash_vlm/dflash-visual):
#   bash cluster/lib/download_weights.sh
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: run from /scratch300/\$USER/dflash_vlm/dflash-visual" >&2; exit 1; }
source ./cluster/env.sh

python cluster/lib/download_weights.py --out-dir "$DFLASH_PRETRAINED" "$@"
