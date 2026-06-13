#!/bin/bash
# Download LlamaGen + FLAN-T5-XL pretrained weights into $DFLASH_PRETRAINED.
# Runs the Python downloader (huggingface_hub) under the project conda env.
#
# Usage (one-line sbatch, per context.md):
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=02:00:00 --cpus-per-task=2 --mem=8G -o output_logs/download_weights.out --job-name=dflash_download --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/lib/download_weights.sh
#
# Pass extra flags through to the Python script, e.g. --skip-t5:
#   sbatch ... ./cluster/lib/download_weights.sh --skip-t5
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: --chdir must point at the repo root (/scratch300/\$USER/dflash_vlm/dflash-visual)" >&2; exit 1; }
source ./cluster/env.sh

python cluster/lib/download_weights.py --out-dir "$DFLASH_PRETRAINED" "$@"
