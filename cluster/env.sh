# Source this file at the top of every sbatch script to set up the
# environment + paths. Tolerant of being sourced multiple times.

set -euo pipefail

# Cluster-writable scratch root. Override by exporting SCRATCH before sourcing.
: "${SCRATCH:=/scratch300/$USER}"
export DFLASH_ROOT="${SCRATCH}/dflash_visual"
export DFLASH_CODE="${DFLASH_ROOT}/code"
export DFLASH_PRETRAINED="${DFLASH_ROOT}/pretrained"
export DFLASH_DATA="${DFLASH_ROOT}/data"
export DFLASH_RUNS="${DFLASH_ROOT}/runs"

mkdir -p "$DFLASH_PRETRAINED" "$DFLASH_DATA" "$DFLASH_RUNS"

# HuggingFace cache: keep target/T5 weights on scratch, never in $HOME.
export HF_HOME="${DFLASH_PRETRAINED}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Conda activation (override CONDA_ENV path if you put it elsewhere).
: "${CONDA_ENV:=${SCRATCH}/conda_envs/dflash_visual}"
module load anaconda 2>/dev/null || module --ignore_cache load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Reproducibility / perf knobs.
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[env] SCRATCH=$SCRATCH"
echo "[env] DFLASH_ROOT=$DFLASH_ROOT"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] CONDA_ENV=$CONDA_ENV  (python=$(python -V 2>&1))"
echo "[env] SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>}"
