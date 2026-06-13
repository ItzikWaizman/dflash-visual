#!/bin/bash
# Drafter training, single GPU. Depends on datagen.
# Usage:
#   sbatch -A <ACC> -p <PART> --qos=<QOS> --time=24:00:00 --gres=gpu:1 \
#          --cpus-per-task=4 --mem=64G \
#          --chdir <repo>/cluster -o <repo>/cluster/output_logs/train_%j.txt \
#          cluster/scripts/train.sh llamagen_xl_t2i_stage2
set -euo pipefail

EXP="${1:?usage: $0 <experiment_name>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../env.sh"

CONFIG="$DFLASH_CODE/cluster/configs/${EXP}.json"
RUN_DIR="$DFLASH_RUNS/${EXP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

J="${SLURM_JOB_ID:-local}"
LOG="$LOG_DIR/train_${J}.log"

{
  echo "=== train $EXP / job $J ==="
  echo "config: $CONFIG"
  nvidia-smi || true

  TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"
  cd "$DFLASH_CODE"
  if [ "$TASK" = "t2i" ]; then
      python train_drafter_t2i.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED"
  else
      python train_drafter.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED"
  fi
} 2>&1 | tee -a "$LOG"
