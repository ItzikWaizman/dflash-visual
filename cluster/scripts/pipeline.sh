#!/bin/bash
# Submit a full datagen -> train -> eval pipeline for one experiment, chained
# via SLURM dependencies so each stage waits for the previous.
#
# Edit ACCT / PART / QOS to match your account, then run from the cluster
# login node:
#   ./cluster/scripts/pipeline.sh llamagen_xl_t2i_stage2
set -euo pipefail

EXP="${1:?usage: $0 <experiment_name>}"
ACCT="${ACCT:-gpu-tad-wolf_v2}"
PART="${PART:-gpu-tad-pool}"
QOS="${QOS:-owner}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUTLOG="$REPO_ROOT/cluster/output_logs"
mkdir -p "$OUTLOG"

DG_ARRAY="${DG_ARRAY:-0-3}"   # 4 parallel datagen tasks
EV_ARRAY="${EV_ARRAY:-0-1}"   # 2 parallel eval tasks

common="-A $ACCT -p $PART --qos=$QOS --cpus-per-task=4 --gres=gpu:1 --chdir $REPO_ROOT"

echo "[pipeline] submitting datagen (array $DG_ARRAY)"
DG=$(sbatch --parsable $common --time=20:00:00 --mem=48G \
            --array="$DG_ARRAY" \
            -o "$OUTLOG/datagen_%A_%a.txt" --job-name="dg_${EXP}" \
            "$REPO_ROOT/cluster/scripts/datagen.sh" "$EXP")
echo "  -> datagen job $DG"

echo "[pipeline] submitting train (depends on datagen)"
TR=$(sbatch --parsable $common --time=24:00:00 --mem=64G \
            --dependency=afterok:"$DG" \
            -o "$OUTLOG/train_%j.txt" --job-name="tr_${EXP}" \
            "$REPO_ROOT/cluster/scripts/train.sh" "$EXP")
echo "  -> train job $TR"

echo "[pipeline] submitting eval (array $EV_ARRAY, depends on train)"
EV=$(sbatch --parsable $common --time=4:00:00 --mem=32G \
            --array="$EV_ARRAY" --dependency=afterok:"$TR" \
            -o "$OUTLOG/eval_%A_%a.txt" --job-name="ev_${EXP}" \
            "$REPO_ROOT/cluster/scripts/eval.sh" "$EXP")
echo "  -> eval job $EV"

echo
echo "submitted: datagen=$DG train=$TR eval=$EV"
echo "watch:   squeue -u \$USER --start; tail -F $OUTLOG/*.txt"
