#!/bin/bash
# Evaluation, parallelizable across prompt/class subsets via SLURM job array.
#
# Usage:
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner \
#          --time=4:00:00 --array=0-1 --gres=gpu:1 \
#          --cpus-per-task=2 --mem=32G \
#          --chdir /scratch300/$USER/dflash_visual/code \
#          -o /scratch300/$USER/dflash_visual/code/cluster/output_logs/eval_%A_%a.txt \
#          cluster/scripts/eval.sh llamagen_xl_t2i_stage2
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
A="${SLURM_ARRAY_TASK_ID:-0}"
ASZ="${SLURM_ARRAY_TASK_COUNT:-1}"

{
  echo "=== eval $EXP / array $A of $ASZ / job $J ==="
  TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"
  echo "task: $TASK"

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
} 2>&1 | tee -a "$LOG_DIR/eval_${J}_${A}.log"
