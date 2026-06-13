#!/bin/bash
# Drafter training, single GPU. Depends on datagen.
#
# Usage:
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner \
#          --time=24:00:00 --gres=gpu:1 \
#          --cpus-per-task=4 --mem=64G \
#          --chdir /scratch300/$USER/dflash_visual/code \
#          -o /scratch300/$USER/dflash_visual/code/cluster/output_logs/train_%j.txt \
#          cluster/scripts/train.sh llamagen_xl_t2i_stage2
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

source "$(dirname "$0")/../env.sh"

EXP="${1:?usage: $0 <experiment_name>}"
CONFIG="$DFLASH_CODE/cluster/experiments/${EXP}/config.json"
RUN_DIR="$DFLASH_RUNS/${EXP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

J="${SLURM_JOB_ID:-local}"

{
  echo "=== train $EXP / job $J ==="
  echo "config: $CONFIG"
  nvidia-smi || true

  TASK="$(python -c "import json; print(json.load(open('$CONFIG'))['task'])")"
  echo "task: $TASK"

  cd "$DFLASH_CODE"
  if [ "$TASK" = "t2i" ]; then
      python train_drafter_t2i.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED"
  else
      python train_drafter.py --config "$CONFIG" \
          --run-dir "$RUN_DIR" --pretrained "$DFLASH_PRETRAINED"
  fi
} 2>&1 | tee -a "$LOG_DIR/train_${J}.log"
