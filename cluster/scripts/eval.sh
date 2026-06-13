#!/bin/bash
# Evaluation, parallelizable across prompt/class subsets.
# Usage:
#   sbatch -A <ACC> -p <PART> --qos=<QOS> --time=4:00:00 \
#          --array=0-3 --gres=gpu:1 --cpus-per-task=2 --mem=32G \
#          --chdir <repo>/cluster -o <repo>/cluster/output_logs/eval_%A_%a.txt \
#          cluster/scripts/eval.sh llamagen_xl_t2i_stage2
set -euo pipefail

EXP="${1:?usage: $0 <experiment_name>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../env.sh"

CONFIG="$DFLASH_CODE/cluster/configs/${EXP}.json"
RUN_DIR="$DFLASH_RUNS/${EXP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

J="${SLURM_JOB_ID:-local}"
A="${SLURM_ARRAY_TASK_ID:-0}"
ASZ="${SLURM_ARRAY_TASK_COUNT:-1}"
LOG="$LOG_DIR/eval_${J}_${A}.log"

{
  echo "=== eval $EXP / array $A of $ASZ / job $J ==="
  TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"
  cd "$DFLASH_CODE"
  if [ "$TASK" = "t2i" ]; then
      python eval_real_drafter_t2i.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED" \
          --array-id "$A" --array-size "$ASZ"
  else
      python eval_real_drafter.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED" \
          --array-id "$A" --array-size "$ASZ"
  fi
} 2>&1 | tee -a "$LOG"
