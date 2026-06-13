#!/bin/bash
# Datagen job. Designed to be submitted as a SLURM job array so multiple GPUs
# generate disjoint shards in parallel.
#
# Usage:
#   sbatch -A <ACC> -p <PART> --qos=<QOS> --time=20:00:00 \
#          --array=0-3 --gres=gpu:1 --cpus-per-task=4 --mem=32G \
#          --chdir <repo>/cluster -o <repo>/cluster/output_logs/datagen_%A_%a.txt \
#          cluster/scripts/datagen.sh llamagen_xl_t2i_stage2
set -euo pipefail

EXP="${1:?usage: $0 <experiment_name>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../env.sh"

CONFIG="$DFLASH_CODE/cluster/configs/${EXP}.json"
RUN_DIR="$DFLASH_RUNS/${EXP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR" "$RUN_DIR"

J="${SLURM_JOB_ID:-local}"
A="${SLURM_ARRAY_TASK_ID:-0}"
ASZ="${SLURM_ARRAY_TASK_COUNT:-1}"
LOG="$LOG_DIR/datagen_${J}_${A}.log"

# pick c2i vs t2i datagen entry point from the config's "task" field
TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"

{
  echo "=== datagen $EXP / task $TASK / array $A of $ASZ / job $J ==="
  echo "config: $CONFIG"
  echo "run_dir: $RUN_DIR"
  nvidia-smi || true

  cd "$DFLASH_CODE"
  if [ "$TASK" = "t2i" ]; then
      python generate_training_data_t2i.py \
          --config "$CONFIG" --run-dir "$RUN_DIR" \
          --pretrained "$DFLASH_PRETRAINED" --data-root "$DFLASH_DATA" \
          --array-id "$A" --array-size "$ASZ"
  elif [ "$TASK" = "c2i" ]; then
      python generate_training_data.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED" \
          --array-id "$A" --array-size "$ASZ"
  else
      echo "unknown task: $TASK" >&2
      exit 2
  fi
} 2>&1 | tee -a "$LOG"
