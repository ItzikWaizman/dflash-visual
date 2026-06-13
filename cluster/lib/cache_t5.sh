#!/bin/bash
# T5 feature cache (t2i only) -- single-GPU prep job. Sampling array depends
# on this via SLURM --dependency=afterok:<this jobid>.
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: --chdir must point at repo root" >&2; exit 1; }
source ./cluster/env.sh

EXP="${1:?usage: $0 <experiment_name>}"
CONFIG="$DFLASH_CODE/cluster/experiments/${EXP}/config.json"
RUN_DIR="$DFLASH_RUNS/${EXP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR" "$RUN_DIR"

J="${SLURM_JOB_ID:-local}"

{
  echo "=== cache_t5 $EXP / job $J ==="
  echo "config: $CONFIG"
  nvidia-smi || true

  TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"
  echo "task: $TASK"
  if [ "$TASK" != "t2i" ]; then
      echo "[cache_t5] task is '$TASK', not 't2i' -- nothing to cache; exiting 0."
      exit 0
  fi

  cd "$DFLASH_CODE"
  python generate_training_data_t2i.py \
      --config "$CONFIG" --run-dir "$RUN_DIR" \
      --pretrained "$DFLASH_PRETRAINED" --data-root "$DFLASH_DATA" \
      --mode cache
} 2>&1 | tee -a "$LOG_DIR/cache_t5_${J}.log"
