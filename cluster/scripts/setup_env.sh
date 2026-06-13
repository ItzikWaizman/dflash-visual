#!/bin/bash
# One-time cluster setup: conda env, deps, pretrained weights.
# sbatch --gres=gpu:1 --time=2:00:00 -o setup.log cluster/scripts/setup_env.sh
set -euo pipefail

source "$(dirname "$0")/../env.sh"

if [ ! -d "$CONDA_ENV" ]; then
    echo "[setup] creating conda env at $CONDA_ENV"
    conda create -y -p "$CONDA_ENV" python=3.12
    conda activate "$CONDA_ENV"
fi

echo "[setup] installing python deps"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy transformers huggingface_hub safetensors
pip install bitsandbytes ftfy beautifulsoup4 sentencepiece
pip install Pillow tqdm

echo "[setup] downloading LlamaGen pretrained weights to $DFLASH_PRETRAINED/llamagen"
mkdir -p "$DFLASH_PRETRAINED/llamagen"
cd "$DFLASH_PRETRAINED/llamagen"
download() {
    [ -f "$2" ] && return 0
    echo "  fetching $2"
    wget -q --show-progress -O "$2" "$1"
}
# c2i targets + VQ tokenizer
download https://huggingface.co/peizesun/llamagen/resolve/main/c2i_3B_384.pt c2i_3B_384.pt
download https://huggingface.co/peizesun/llamagen/resolve/main/c2i_XXL_384.pt c2i_XXL_384.pt
download https://huggingface.co/peizesun/llamagen/resolve/main/vq_ds16_c2i.pt vq_ds16_c2i.pt
# t2i: XL Stage 2 (512) + VQ tokenizer trained on LAION
download https://huggingface.co/peizesun/llamagen/resolve/main/t2i_XL_stage2_512.pt t2i_XL_stage2_512.pt
download https://huggingface.co/peizesun/llamagen/resolve/main/vq_ds16_t2i.pt vq_ds16_t2i.pt

echo "[setup] T5 weights will lazy-download via T5Embedder into $DFLASH_PRETRAINED/t5"
mkdir -p "$DFLASH_PRETRAINED/t5"

echo "[setup] DONE"
