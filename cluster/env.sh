# DFlash-project paths only. Sourced AFTER the user's
# /scratch300/$USER/env.sh + conda activate /scratch300/$USER/conda_envs/unlearning
# already set up CUDA, $USER, $SCRATCH, etc.

set -euo pipefail

: "${SCRATCH:=/scratch300/$USER}"
export DFLASH_ROOT="${DFLASH_ROOT:-${SCRATCH}/dflash_vlm}"
export DFLASH_CODE="${DFLASH_CODE:-${DFLASH_ROOT}/dflash-visual}"
export DFLASH_PRETRAINED="${DFLASH_ROOT}/pretrained"
export DFLASH_DATA="${DFLASH_ROOT}/data"
export DFLASH_RUNS="${DFLASH_ROOT}/runs"
mkdir -p "$DFLASH_PRETRAINED" "$DFLASH_DATA" "$DFLASH_RUNS"

# Keep target/T5 weight downloads on scratch, never in $HOME.
export HF_HOME="${DFLASH_PRETRAINED}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[dflash-env] DFLASH_ROOT=$DFLASH_ROOT  HF_HOME=$HF_HOME"
echo "[dflash-env] python: $(which python)  $(python -V 2>&1)"
echo "[dflash-env] SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>}"
