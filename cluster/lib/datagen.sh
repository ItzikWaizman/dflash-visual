#!/bin/bash
# Datagen job. Submit as SLURM job array for parallel shard generation.
#
# Usage (one-line sbatch, per context.md):
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=20:00:00 --array=0-3 --cpus-per-task=4 --mem=48G -o output_logs/datagen_llamagen_xl_t2i_stage2_%A_%a.out --job-name=datagen_xlt2i --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/lib/datagen.sh llamagen_xl_t2i_stage2
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
mkdir -p "$LOG_DIR" "$RUN_DIR"

J="${SLURM_JOB_ID:-local}"
A="${SLURM_ARRAY_TASK_ID:-0}"
ASZ="${SLURM_ARRAY_TASK_COUNT:-1}"

{
  echo "=== datagen $EXP / array $A of $ASZ / job $J ==="
  echo "config: $CONFIG"
  echo "run_dir: $RUN_DIR"
  nvidia-smi || true

  TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"
  echo "task: $TASK"

  cd "$DFLASH_CODE"
  if [ "$TASK" = "t2i" ]; then
      python generate_training_data_t2i.py \
          --config "$CONFIG" --run-dir "$RUN_DIR" \
          --pretrained "$DFLASH_PRETRAINED" --data-root "$DFLASH_DATA" \
          --array-id "$A" --array-size "$ASZ"
  elif [ "$TASK" = "c2i" ]; then
      python generate_training_data.py \
          --config "$CONFIG" --run-dir "$RUN_DIR" \
          --pretrained "$DFLASH_PRETRAINED" \
          --array-id "$A" --array-size "$ASZ"
  else
      echo "unknown task: $TASK" >&2
      exit 2
  fi
} 2>&1 | tee -a "$LOG_DIR/datagen_${J}_${A}.log"
