#!/bin/bash
# Experiment: LlamaGen-3B c2i @ 384x384 (576 image tokens, 1 class token).
# Cluster replica of the 5080 pilot. Direct comparison vs LANTERN's 3B ablation
# and a calibration point for the consumer-vs-cluster scaling story.
#
# Submit from the cluster login node:
#   ./cluster/experiments/llamagen_3b_c2i_384/run.sh
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../../env.sh"

EXP="llamagen_3b_c2i_384"
ACCT="${ACCT:-gpu-tad-wolf_v2}"
PART="${PART:-gpu-tad-pool}"
QOS="${QOS:-owner}"
DG_ARRAY="${DG_ARRAY:-0-3}"
EV_ARRAY="${EV_ARRAY:-0-1}"

LIB="$DFLASH_CODE/cluster/lib"
LOG_DIR="$DFLASH_RUNS/$EXP/logs"
mkdir -p "$LOG_DIR"

COMMON="-A $ACCT -p $PART --qos=$QOS --gres=gpu:1 --cpus-per-task=4 --chdir $DFLASH_CODE --job-name=${EXP}"

echo "[$EXP] submitting datagen (array=$DG_ARRAY)"
DG=$(sbatch --parsable $COMMON --time=18:00:00 --mem=48G \
            --array="$DG_ARRAY" \
            -o "$LOG_DIR/sbatch_datagen_%A_%a.out" \
            "$LIB/datagen.sh" "$EXP")
echo "  -> $DG"

echo "[$EXP] submitting train (depends on $DG)"
TR=$(sbatch --parsable $COMMON --time=24:00:00 --mem=64G \
            --dependency=afterok:"$DG" \
            -o "$LOG_DIR/sbatch_train_%j.out" \
            "$LIB/train.sh" "$EXP")
echo "  -> $TR"

echo "[$EXP] submitting eval (array=$EV_ARRAY, depends on $TR)"
EV=$(sbatch --parsable $COMMON --time=4:00:00 --mem=32G \
            --array="$EV_ARRAY" --dependency=afterok:"$TR" \
            -o "$LOG_DIR/sbatch_eval_%A_%a.out" \
            "$LIB/eval.sh" "$EXP")
echo "  -> $EV"

echo
echo "submitted: datagen=$DG train=$TR eval=$EV"
echo "watch:     squeue -u \$USER; tail -F $LOG_DIR/*.out"
