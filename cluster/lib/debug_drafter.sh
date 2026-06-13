#!/bin/bash
# Debug-mode diagnostic: load trained t2i drafter + tiny batch, emit NDJSON
# to debug-a22afb.log AND stdout for hypothesis testing.
#
# Usage (one-line sbatch, per context.md):
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=00:30:00 --cpus-per-task=4 --mem=64G -o output_logs/debug_drafter_llamagen_xl_t2i_stage2.out --job-name=dbg_xlt2i --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/lib/debug_drafter.sh llamagen_xl_t2i_stage2
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: cluster/env.sh not found. Did sbatch get --chdir to the repo root?" >&2; exit 1; }
source ./cluster/env.sh

EXP="${1:?usage: $0 <experiment_name>}"
CONFIG="$DFLASH_CODE/cluster/experiments/${EXP}/config.json"
RUN_DIR="$DFLASH_RUNS/${EXP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

J="${SLURM_JOB_ID:-local}"

{
  echo "=== debug_drafter $EXP / job $J ==="
  echo "config: $CONFIG"
  nvidia-smi || true

  cd "$DFLASH_CODE"
  python cluster/lib/debug_drafter.py --config "$CONFIG" \
      --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED"

  echo "--- debug-a22afb.log contents ---"
  cat debug-a22afb.log || true
  echo "--- end debug-a22afb.log ---"
} 2>&1 | tee -a "$LOG_DIR/debug_drafter_${J}.log"
