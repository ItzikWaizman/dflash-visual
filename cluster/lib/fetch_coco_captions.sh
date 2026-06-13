#!/bin/bash
# Download COCO 2017 captions and emit JSONL files (~250 MB download, ~591K
# train + ~25K val captions). Runs the Python downloader.
#
# Usage (one-line sbatch, per context.md):
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=01:00:00 --cpus-per-task=2 --mem=8G -o output_logs/fetch_coco.out --job-name=dflash_coco --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/lib/fetch_coco_captions.sh
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: --chdir must point at repo root" >&2; exit 1; }
source ./cluster/env.sh

python cluster/lib/fetch_coco_captions.py --out-dir "$DFLASH_DATA"
