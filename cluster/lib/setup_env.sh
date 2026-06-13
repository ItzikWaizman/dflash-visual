#!/bin/bash
# One-time cluster setup: install DFlash deps into the existing `unlearning`
# conda env, then download LlamaGen pretrained weights.
#
# Usage (one-line sbatch, per context.md):
#   sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=02:00:00 --cpus-per-task=2 --mem=16G -o output_logs/setup_env.out --job-name=dflash_setup --chdir /scratch300/$USER/dflash_vlm/dflash-visual/ ./cluster/lib/setup_env.sh
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

test -f cluster/env.sh || { echo "ERROR: cluster/env.sh not found. Did sbatch get --chdir to the repo root?" >&2; exit 1; }
source ./cluster/env.sh

echo "[setup] python: $(which python)  $(python -V)"
echo "[setup] pip:    $(which pip)"

# Install torch FIRST against the cu128 index (matches the 5080 box).
# If the cluster's `unlearning` env already has a working torch, this is a no-op.
pip install --upgrade pip
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[setup] installing torch (cu128 wheels)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    echo "[setup] torch already importable with CUDA -> skipping"
fi

# DFlash-specific deps.
pip install -r "$DFLASH_CODE/cluster/requirements.txt"

# Smoke-test the imports we actually use.
python - <<'PY'
import torch, numpy, transformers, huggingface_hub, ftfy, sentencepiece
print("[setup] torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("[setup] transformers", transformers.__version__)
try:
    import bitsandbytes as bnb
    print("[setup] bitsandbytes", bnb.__version__)
except Exception as e:
    print(f"[setup] bitsandbytes unavailable (ok, fall back to fp32 AdamW): {e}")
PY

# ---- pretrained weights ------------------------------------------------------
# Weight downloads are handled by a separate sbatch job (no GPU needed):
#   sbatch ... --chdir .../dflash-visual/ cluster/lib/download_weights.sh
mkdir -p "$DFLASH_PRETRAINED/llamagen" "$DFLASH_PRETRAINED/t5"
if compgen -G "$DFLASH_PRETRAINED/llamagen/*.pt" > /dev/null; then
    echo "[setup] LlamaGen weights present in $DFLASH_PRETRAINED/llamagen:"
    ls -lh "$DFLASH_PRETRAINED/llamagen"
else
    echo "[setup] LlamaGen weights NOT yet downloaded."
    echo "        Submit: sbatch ... --chdir <repo> cluster/lib/download_weights.sh"
fi

echo "[setup] DONE"
